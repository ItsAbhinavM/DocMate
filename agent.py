import json
import os
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import (JsonOutputParser,
                                           PydanticOutputParser,
                                           StrOutputParser)
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from filehandler import query_vectorstore

load_dotenv()


# --- Define the structure for schema interpretation ---
class InterpretedSchema(BaseModel):
    """Structure representing the desired dataset schema and retrieval query."""

    schema_description: Dict[str, str] = Field(
        description="A dictionary where keys are column names and values are descriptions of the data expected in that column."
    )
    retrieval_query: Optional[str] = Field(
        description="Optimized query string for vector similarity search based on the user request."
    )
    ambiguity: Optional[str] = Field(
        description="If the user query is too vague or could be interpreted in multiple ways request explain briefly why clarification is needed",
        default=None,
    )
    refinement_instructions: Optional[str] = Field(
        description="If refining, specific instructions derived from the query on how to modify the previous dataset (e.g., 'filter out entries where status is closed', 'add currency symbol'). Null for initial generation.",
        default=None,
    )
    needs_statistics: Optional[bool]= Field(
            description="If the user needs to see the statistics of the overall datbase"
    )


class ExtractedItem(BaseModel):
    """Represents a single extracted data record conforming to a part of the schema."""

    data: Dict[str, Any] = Field(
        description="Extracted data points for a single item/record."
    )
    source_document_info: Optional[Dict] = Field(
        description="Metadata about the source document (e.g., source name, page number).",
        default=None,
    )


# --- Define the State for the Graph ---
class GraphState(TypedDict):
    # Inputs
    original_query: str
    run_mode: Literal["initial_generation", "refinement"]
    previous_dataset: Optional[List[Dict]]  # Loaded by caller for refinement

    # Tracking / Intermediate
    current_query: Optional[str]  # Query from a clarification event
    interpreted_schema: Optional[InterpretedSchema]
    retrieved_docs: Optional[List[Document]]  # Docs from vectorstore
    extracted_data_points: List[ExtractedItem]  # List of raw extractions
    error_message: Optional[str]
    needs_clarification: Optional[bool]
    current_iteration: int
    needs_statistics: Optional[bool]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
saver = MemorySaver()


# --- Define Graph Nodes ---
def root_node(state: GraphState):
    """Agent start point, increments the iteration count on each loop"""
    iteration = state.get("current_iteration", 0) + 1
    print(f"--- [Node: Increment Iteration] --- (Starting Iteration {iteration})")
    return {
        "current_iteration": iteration,
        "error_message": None,  # Clear previous errors on loop/restart
        "needs_clarification": False,  # Default to no clarification needed
    }


def interpret_query_node(state: GraphState):
    """Interprets the user query to define schema and retrieval query."""
    print("--- [Node: Interpret Query] ---")
    user_query = state.get("current_query") or state["original_query"]
    run_mode = state["run_mode"]

    parser = PydanticOutputParser(pydantic_object=InterpretedSchema)

    template = """Analyze the user's query based on the specified run mode to define a structured dataset schema, a search query (if needed), and refinement instructions (if applicable).

Run Mode: {run_mode}
User Query: {query}
"""

    if run_mode == "refinement":
        template += """The user wants to refine a dataset generated previously.
Analyze the query to determine:
1.  `schema_description`: The schema of the *final* dataset after refinement. This might be the same as the old schema or modified by the request. Keys = concise field names, values = clear descriptions.
2.  `retrieval_query`: A query for vector search *only if* the refinement requires fetching *new* information from documents. If the refinement only involves filtering/modifying the *existing* data (passed separately), this should be null.
3.  `refinement_instructions`: A clear, actionable instruction for a downstream process describing *how* to modify the previous dataset based on the user's query AND any newly extracted data (if retrieval_query is not null). Examples: "Filter previous data to keep only entries with status='active'", "Merge new data points, prioritizing newer sources for conflicting values", "Add a 'category' field based on the 'description'". If the query implies replacing the dataset entirely with new data, state that.
4.  `ambiguity`: Briefly explain *why* clarification is needed if the query is vague (e.g., "Refinement target unclear", "Filter criteria ambiguous"). Otherwise, null.
"""

    else:  # initial_generation
        template += """The user wants to generate a dataset from scratch.
Analyze the query to define:
1.  `schema_description`: A dictionary describing the columns/fields for the new dataset. Keys = concise field names, values = clear descriptions.
2.  `retrieval_query`: A query string optimized for finding relevant documents in a vector database. Focus on key entities, actions, and concepts. This should *not* be null for initial generation unless the query is completely nonsensical.
3.  `refinement_instructions`: Should be null for initial generation.
4.  `ambiguity`: Briefly explain *why* clarification is needed if the query is vague (e.g., "Query is too general", "Specific entities not mentioned"). Otherwise, null.
"""

    template += "\n{format_instructions}\n"

    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "run_mode"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    try:
        interpreted_output = chain.invoke({"query": user_query, "run_mode": run_mode})
        print(f"Ambiguity: {interpreted_output.ambiguity}")
        print(f"Interpreted Schema: {interpreted_output.schema_description}")
        print(
            f"  Refinement Instructions: {interpreted_output.refinement_instructions}"
        )
        print(f"Retrieval Query: {interpreted_output.retrieval_query}")

        # Basic validation
        if (
            run_mode == "initial_generation"
            and not interpreted_output.retrieval_query
            and not interpreted_output.ambiguity
        ):
            print(
                "[WARN] Initial generation mode but no retrieval query was generated. Setting ambiguity."
            )
            interpreted_output.ambiguity = (
                "Initial generation requires a search query, but none could be derived."
            )

        needs_clarification = bool(interpreted_output.ambiguity)

        return {
            "interpreted_schema": interpreted_output,
            "needs_clarification": needs_clarification,
            "error_message": None,
            # Clear potentially outdated fields from previous loops if clarification happened
            "retrieved_docs": (
                None if needs_clarification else state.get("retrieved_docs")
            ),
            "extracted_data_points": (
                [] if needs_clarification else state.get("extracted_data_points", [])
            ),
        }
    except Exception as e:
        print(f"[ERROR] Failed to interpret query: {e}")
        return {"error_message": f"Failed to interpret query: {str(e)}"}


def retrieve_documents_node(state: GraphState):
    """Retrieves documents from the vector store based on the interpreted query."""
    print("--- [Node: Retrieve Documents] ---")
    if state.get("error_message"):  # Skip if previous step failed
        return {}
    if not state.get("interpreted_schema"):
        return {"error_message": "Interpretation step did not produce schema/query."}

    retrieval_query = state["interpreted_schema"].retrieval_query
    try:
        documents = query_vectorstore(
            retrieval_query, k=10
        )  # k should be made variable based on query
        if not documents:
            print("[WARN] No documents found for the query.")
            return {
                "retrieved_docs": [],
                "extracted_data_points": [],
            }  # Proceed with empty list

        return {"retrieved_docs": documents, "error_message": None}
    except Exception as e:
        print(f"[ERROR] Failed during vector store query: {e}")
        return {"error_message": f"Failed during vector store query: {str(e)}"}


def extract_data_node(state: GraphState):
    """Extracts structured data from retrieved documents based on the schema."""
    print("--- [Node: Extract Data] ---")
    if state.get("error_message") or not state.get(
        "retrieved_docs"
    ):  # Skip if error or no docs
        # If no docs, ensure extracted_data_points is initialized
        if "extracted_data_points" not in state:
            return {"extracted_data_points": []}
        return {}  # Pass existing state if error occurred

    original_query = state["original_query"]
    schema_dict = state["interpreted_schema"].schema_description
    documents = state["retrieved_docs"]

    # Define the parser based *dynamically* on the interpreted schema for extraction
    # For simplicity, we'll request a JSON blob matching the schema keys.
    # A more robust approach might create a Pydantic model on the fly.

    extraction_prompt_template = """
Given the user's original query and a target data schema, extract relevant information ONLY from the provided document context.
Original User Query: {original_query}

Target Schema:
{schema_description}

Document Context:
---
{document_content}
---

Extract all data points matching the schema found within the document context. If a field is not present, omit it or use null.
Format the output as a JSON object where keys match the schema description. If multiple distinct items matching the schema are found, return a JSON list of objects. If no relevant information is found, return an empty JSON object or list.

AVOID using the triple backticks format, just print the RAW JSON object.

Extracted Data (JSON):
"""
    extraction_prompt = PromptTemplate.from_template(extraction_prompt_template)

    extraction_chain = extraction_prompt | llm | JsonOutputParser()

    extracted_items: List[ExtractedItem] = []

    for i, doc in enumerate(documents):
        print(
            f"Processing document {i+1}/{len(documents)} (Source: {doc.metadata.get('source', 'N/A')[:50]}...)"
        )
        try:
            # Prepare input for the extraction chain
            schema_desc_string = json.dumps(schema_dict, indent=2)
            input_data = {
                "original_query": original_query,
                "schema_description": schema_desc_string,
                "document_content": doc.page_content.strip(),
            }
            extracted_data = extraction_chain.invoke(input_data)

            # Attempt to parse the JSON output
            try:
                # Handle both single object and list of objects responses
                if isinstance(extracted_data, dict) and extracted_data:
                    item = ExtractedItem(
                        data=extracted_data, source_document_info=doc.metadata
                    )
                    extracted_items.append(item)
                    print(f"  Extracted: {item.data}")
                elif isinstance(extracted_data, list):
                    for entry in extracted_data:
                        if isinstance(entry, dict) and entry:
                            item = ExtractedItem(
                                data=entry, source_document_info=doc.metadata
                            )
                            extracted_items.append(item)
                            print(f"  Extracted (from list): {item.data}")
            except json.JSONDecodeError:
                print(
                    f"  [WARN] LLM output was not valid JSON: {extracted_json_str[:100]}..."
                )
            except Exception as parse_err:
                print(f"  [WARN] Could not parse extracted data: {parse_err}")

        except Exception as e:
            print(f"  [ERROR] Failed to extract from document {i+1}: {e}")
            # Continue to next document

    return {"extracted_data_points": extracted_items, "error_message": None}


def process_data_node(state: GraphState):

    print("--- [Node: Process Data] ---")
    if state.get("error_message"):
        return {}

    run_mode = state["run_mode"]
    newly_extracted_points = state.get("extracted_data_points", [])  # From current run
    previous_dataset = state.get("processed_dataset", [])
    target_schema = state.get("interpreted_schema").schema_description
    refinement_instructions = state.get(
        "interpreted_schema", {}
    ).refinement_instructions

    processed_dataset: List[Dict] = []

    if run_mode == "initial_generation":
        print("  Mode: Initial Generation - Creating dataset from extracted points.")
        # Basic synthesis: collect data dictionaries
        processed_dataset = [item.data for item in newly_extracted_points]
        print(f"  Synthesized dataset with {len(processed_dataset)} records.")

    elif run_mode == "refinement":
        print("  Mode: Refinement - Applying changes to previous dataset.")
        if not previous_dataset:
            print(
                "  [WARN] Refinement mode, but no previous dataset was provided. Using only newly extracted points."
            )
            processed_dataset = [item.data for item in newly_extracted_points]
        else:
            print(f"  Newly extracted points: {len(newly_extracted_points)}")
            print(f"  Refinement Instructions: {refinement_instructions}")

            refinement_prompt_template = """You are a data refinement assistant.
**Your Goal:** Refine a dataset based on user instructions, using any provided new data. Output a single JSON list of records that strictly follows the target schema.

**Inputs:**

--- Previous Dataset ---
{previous_dataset}
--- End Previous Dataset ---

--- Newly Extracted Data ---
{new_data}
--- End Newly Extracted Data ---

--- Target Schema ---
{target_schema}
--- End Target Schema ---

--- Refinement Instructions ---
{instructions}
--- End Refinement Instructions ---

**Guidelines:**

1. **Follow Instructions First:** Treat the instructions as your primary guide.
2. **Use Previous Dataset:** Start from it unless told to discard or replace.
3. **Integrate New Data:** Append, update, or merge as instructed. Deduplicate if needed, based on likely unique fields.
4. **Filter & Transform:** Apply all specified filters and data transformations.
5. **Match Schema:** Each record must:
   - Contain only keys in the target schema.
   - Include all required keys (use `null` if missing).
6. **Edge Cases:**
   - If both datasets are empty, return `[]`.
   - If one is empty, refine the other.

**Output:** Only return the final refined dataset as a valid JSON list. Do not use thriple backtick return the raw json.
            """
            refinement_prompt = PromptTemplate.from_template(refinement_prompt_template)
            refinement_chain = refinement_prompt | llm | JsonOutputParser()

            # Prepare the input dictionary
            refinement_input = {
                "previous_dataset": previous_dataset,
                "new_data": newly_extracted_points,
                "target_schema": target_schema,
                "instructions": (
                    refinement_instructions
                    if refinement_instructions
                    else "No specific refinement instructions provided. Merge new data with previous data, ensuring final schema compliance and attempting basic deduplication based on obvious keys."
                ),  # Provide default if needed
            }

            try:
                print("Calling LLM for refinement processing...")
                processed_dataset = refinement_chain.invoke(refinement_input)
                print(f"LLM refinement resulted in {len(processed_dataset)} records.")

            except OutputParserException as e:
                print(f"  [ERROR] Failed to parse LLM refinement output: {e}")
                # Log raw output for debugging
                raw_output = (refinement_prompt | llm | StrOutputParser()).invoke(
                    refinement_input
                )
                print(
                    f"  Raw LLM Output (Refinement):\n---\n{raw_output[:500]}...\n---"
                )
                # Handle error: Maybe return original + new, or fail.
                processed_dataset = previous_dataset + [
                    item.data for item in newly_extracted_points
                ]  # Naive fallback
                state["error_message"] = (
                    f"LLM refinement output parsing failed: {e}"  # Propagate soft error
                )

    return {"processed_dataset": processed_dataset, "error_message": None}

def generate_statistics_node(state: GraphState):
    """Generates statistics about the available documents."""
    print("--- [Node: Generate Statistics] ---")

    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64
    import os
    from filehandler import get_document_stats

    try:
        doc_stats = get_document_stats()
        documents = doc_stats["documents"]

        if not documents:
            raise ValueError("No documents found.")

        structured_counts = sum(1 for doc in documents if doc["structured"])
        unstructured_counts = doc_stats["total_docs"] - structured_counts

        token_counts_by_type = {}
        for doc in documents:
            token_counts_by_type.setdefault(doc["type"], []).append(doc["tokens"])

        total_token_by_type = {k: sum(v) for k, v in token_counts_by_type.items()}
        top_heavy_docs = sorted(documents, key=lambda d: d["tokens"], reverse=True)[:5]

        plt.figure(figsize=(15, 5))

        # 1. Structured vs Unstructured Pie
        plt.subplot(1, 3, 1)
        plt.pie(
            [structured_counts, unstructured_counts],
            labels=["Structured", "Unstructured"],
            autopct="%1.1f%%",
            startangle=140
        )
        plt.title("Structured vs Unstructured Documents")

        # 2. Token Load by Type
        plt.subplot(1, 3, 2)
        plt.bar(total_token_by_type.keys(), total_token_by_type.values())
        plt.title("Total Tokens by Document Type")
        plt.xticks(rotation=45)

        # 3. Top 5 Heavy Docs
        plt.subplot(1, 3, 3)
        names = [doc["name"] for doc in top_heavy_docs]
        tokens = [doc["tokens"] for doc in top_heavy_docs]
        plt.barh(names, tokens)
        plt.title("Top 5 Longest Documents")

        plt.tight_layout()
        output_dir = "statistics_output"
        os.makedirs(output_dir, exist_ok=True)
        filename = "analytics_dashboard.png"
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path)
        print("Image has been saved successfully.")

        # Convert to base64 for visualization in frontend (optional)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return {
            "statistics": doc_stats,
            "visualization": img_str,
            "image_path": file_path,
            "error_message": None
        }

    except Exception as e:
        print(f"[ERROR] Failed to generate statistics: {e}")
        return {"error_message": f"Failed to generate statistics: {str(e)}"}


# -- Functions for conditional edges --
def should_continue(state: GraphState) -> Literal["continue", "end_error"]:
    """Determines whether to continue processing or end due to errors."""
    if state.get("error_message"):
        print(
            f"--- [Edge Logic] --- Error encountered: {state['error_message']}. Ending workflow."
        )
        return "end_error"
    print("--- [Edge Logic] --- No errors detected. Continuing workflow.")
    return "continue"


def decide_after_interpret(
    state: GraphState,
) -> Literal["proceed_to_retrieve", "loop_for_clarification", "handle_error"]:
    """Routes flow after interpretation based on errors or need for clarification."""
    print("--- [Edge: Decide After Interpret] ---")
    if state.get("error_message"):
        print("  Decision: Error occurred during interpretation.")
        return "handle_error"

    if state.get("needs_statistics"):
        print("Decision: Statistics requested. Routing to statistics node")
        return "generate_statistics"

    if state.get("needs_clarification"):
        print(
            "  Decision: Clarification needed (API should pause/resume). Routing back for re-interpretation."
        )
        # The API will pause *before* this edge routes. When it resumes,
        # the graph follows this path back to increment/interpret.
        return "loop_for_clarification"

    if not state.get("interpreted_schema"):
        # Safety check: Should not happen if no error and no clarification needed
        print("  Decision: Interpretation successful, but schema missing unexpectedly.")
        state["error_message"] = (
            "Interpretation node finished without error but schema is missing."
        )
        return "handle_error"

    if state["run_mode"] == "refinement":
        print(
            " Decision: Interpretation successful, refining run detected, proceeding to processing"
        )
        return "proceed_to_processing"

    print("  Decision: Interpretation successful, proceeding to retrieve documents.")
    return "proceed_to_retrieve"

def decide_after_synthesize(
    state: GraphState,
) -> Literal["end_workflow", "generate_statistics"]:
    """Routes flow after synthesis to either end or generate statistics."""
    print("--- [Edge: Decide After Synthesize] ---")
    
    if state.get("needs_statistics", False):
        print("  Decision: Statistics requested. Routing to statistics node.")
        return "generate_statistics"
    
    print("  Decision: Normal workflow complete, ending.")
    return "end_workflow"

# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("root", root_node)
workflow.add_node("interpret_query", interpret_query_node)
workflow.add_node("retrieve_documents", retrieve_documents_node)
workflow.add_node("extract_data", extract_data_node)
workflow.add_node("process_data", process_data_node)
workflow.add_node(
    "error_node", lambda state: print("Workflow terminated due to error.")
)

# Define edges
workflow.set_entry_point("root")

workflow.add_edge("root", "interpret_query")

workflow.add_conditional_edges(
    "interpret_query",
    decide_after_interpret,
    {
        "loop_for_clarification": "root",
        "proceed_to_retrieve": "retrieve_documents",
        "proceed_to_processing": "process_data",
        "end_error": "error_node",
    },
)
workflow.add_conditional_edges(
    "synthesize_dataset",
    decide_after_synthesize,
    {
        "end_workflow": END,
        "generate_statistics": "generate_statistics",
    },
)
workflow.add_conditional_edges(
    "retrieve_documents",
    should_continue,
    {
        "continue": "extract_data",
        "end_error": "error_node",
        # Add path for no docs found if needed: "end_no_docs": END
    },
)

workflow.add_conditional_edges(
    "extract_data",
    should_continue,
    {"continue": "process_data", "end_error": "error_node"},
)


# Final step
workflow.add_edge("process_data", END)  # Successful completion ends here
workflow.add_edge("error_node", END)  # Error path ends here
workflow.add_edge("generate_statistics",END)
# Compile the graph
graph = workflow.compile(checkpointer=saver)

if __name__ == "__main__":
    print("\n--- Running Sample Workflow ---")

    # 1. User Query Input:
    initial_state = {"original_query": "give data"}
    print(f"Initial Query: {initial_state['original_query']}")

    # 2. Run the Graph:
    final_state = graph.invoke(
        initial_state, {"recursion_limit": 10}
    )  # Add recursion limit

    # 4. Final Output:
    #    The `final_state` dictionary contains the results of the workflow.
    print("\n--- Workflow Complete ---")
    if final_state.get("error_message"):
        print(f"Workflow failed with error: {final_state['error_message']}")
    elif final_state.get("process_data") is not None:
        print("Final Dataset:")
        # Pretty print the final dataset
        print(json.dumps(final_state["processed_data"], indent=2))
        print(f"\nTotal records generated: {len(final_state['processed_dataset'])}")
    else:
        print(
            "Workflow finished, but no dataset was generated (e.g., no relevant documents found or data extracted)."
        )
