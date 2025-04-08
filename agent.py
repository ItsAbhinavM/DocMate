import json
import os
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
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
    retrieval_query: str = Field(
        description="Optimized query string for vector similarity search based on the user request."
    )
    ambiguity: Optional[str] = Field(
        description="If the user query is too vague or could be interpreted in multiple ways request explain briefly why clarification is needed"
    )
    needs_statistics: Optional[bool] = Field(
        description="If the user needs to see the statistics of the overall database"
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
    original_query: str
    current_query: Optional[str]  # Query from a clarification event
    interpreted_schema: Optional[InterpretedSchema]
    retrieved_docs: Optional[List[Document]]  # Docs from vectorstore
    extracted_data_points: List[ExtractedItem]  # List of raw extractions
    synthesized_dataset: Optional[List[Dict]]  # Final structured dataset
    error_message: Optional[str]
    needs_clarification: Optional[bool]
    current_iteration: int
    statistics_path: Optional[str]


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
saver = MemorySaver()


# --- Define Graph Nodes ---
def root_node(state: GraphState):
    """Agent start point, increments the iteration count on each loop"""
    iteration = state.get("current_iteration", 0) + 1
    print(f"--- [Node: Increment Iteration] --- (Starting Iteration {iteration})")
    return {"current_iteration": iteration}


def interpret_query_node(state: GraphState):
    """Interprets the user query to define schema and retrieval query."""
    print("--- [Node: Interpret Query] ---")
    user_query = state.get("current_query") or state["original_query"]
    parser = PydanticOutputParser(pydantic_object=InterpretedSchema)

    prompt = PromptTemplate(
        template="""Analyze the user's query to define a structured dataset schema and a concise search query.
User Query: {query}

Based on the query, define:
1.  `schema_description`: A dictionary describing the columns/fields the user wants in their dataset. Keys should be concise field names, values should be clear descriptions.
2.  `retrieval_query`: A query string optimized for finding relevant documents/chunks in a vector database related to the user's request. Focus on key entities, actions, and concepts.
3.  `ambiguity`: briefly explain *why* (e.g., "Query is too general", "Specific entities not mentioned"). Otherwise, leave as null.
4. `needs_statistics`: Set to true if the user is requesting statistics or analysis about the document collection itself, rather than data extraction. For example, requests like "show me statistics of the documents", "how many PDFs do we have?", or "what's the average document length?" would set this to true.

{format_instructions}
""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    try:
        interpreted_output = chain.invoke({"query": user_query})
        print(f"Ambiguity: {interpreted_output.ambiguity}")
        print(f"Interpreted Schema: {interpreted_output.schema_description}")
        print(f"Retrieval Query: {interpreted_output.retrieval_query}")
        print(f"Needs Statistics: {interpreted_output.needs_statistics}")

        if interpreted_output.ambiguity:
            return {
                "interpreted_schema": interpreted_output,
                "needs_clarification": True,
                "error_message": None,
            }

        return {
            "interpreted_schema": interpreted_output,
            "needs_clarification": False,
            "error_message": None,
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

    extraction_chain = extraction_prompt | llm | StrOutputParser()

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
            extracted_json_str = extraction_chain.invoke(input_data)

            # Attempt to parse the JSON output
            try:
                extracted_data = json.loads(extracted_json_str)
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


def synthesize_dataset_node(state: GraphState):
    """Synthesizes the final dataset from extracted data points."""
    print("--- [Node: Synthesize Dataset] ---")
    if state.get("error_message"):
        return {}

    extracted_items = state.get("extracted_data_points", [])
    if not extracted_items:
        print("No data points were extracted, resulting dataset is empty.")
        return {"synthesized_dataset": []}

    # Basic synthesis: Just collect the data dictionaries.
    # More advanced synthesis could involve:
    # 1. Deduplication: Identify and merge records referring to the same real-world entity.
    # 2. Conflict Resolution: If multiple documents provide different values for the same field (e.g., different total amounts for the same invoice), apply a rule (e.g., majority vote, latest source, flag for review).
    # 3. Aggregation: Combine information from different documents for the same entity.

    final_dataset = [item.data for item in extracted_items]  # Simple collection for now

    print(f"Synthesized dataset with {len(final_dataset)} records.")
    return {"synthesized_dataset": final_dataset, "error_message": None}


import os

import matplotlib.pyplot as plt
import numpy as np

from filehandler import get_document_stats


def generate_statistics_node(state: GraphState):
    try:
        # Create output directory if it doesn't exist
        output_dir = "statistics_output"
        os.makedirs(output_dir, exist_ok=True)

        # Get statistics from your documents
        doc_stats = get_document_stats()
        print("document statistics are: ", doc_stats)

        # Create some visualizations
        plt.figure(figsize=(10, 6))

        # Document count by type
        plt.subplot(1, 2, 1)
        doc_types = [key for key in doc_stats["doc_types"].keys()]
        counts = [value for value in doc_stats["doc_types"].values()]
        plt.bar(doc_types, counts)
        plt.title("Document Count by Type")
        plt.xticks(rotation=45)

        # Token distribution
        plt.subplot(1, 2, 2)
        plt.hist(doc_stats["token_counts"], bins=10)
        plt.title("Token Count Distribution")

        # Save plot to file in the output directory
        output_path = os.path.join(output_dir, "document_stats.png")
        plt.tight_layout()
        plt.savefig(output_path, format="png")
        plt.close()

        return {
            "statistics_path": output_path,
            "error_message": None,
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

    print("  Decision: Interpretation successful, proceeding to retrieve documents.")
    return "proceed_to_retrieve"


def decide_after_synthesize(
    state: GraphState,
) -> Literal["end_workflow", "proceed_to_statistics"]:
    """Routes flow after synthesis to either end or generate statistics."""
    print("--- [Edge: Decide After Synthesize] ---")

    if state.get("needs_statistics", False):
        print("  Decision: Statistics requested. Routing to statistics node.")
        return "proceed_to_statistics"

    print("  Decision: Normal workflow complete, ending.")
    return "end_workflow"


# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("root", root_node)
workflow.add_node("interpret_query", interpret_query_node)
workflow.add_node("retrieve_documents", retrieve_documents_node)
workflow.add_node("extract_data", extract_data_node)
workflow.add_node("synthesize_dataset", synthesize_dataset_node)
workflow.add_node("generate_statistics", generate_statistics_node)
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
        "end_error": "error_node",
    },
)
workflow.add_conditional_edges(
    "synthesize_dataset",
    decide_after_synthesize,
    {
        "end_workflow": END,
        "proceed_to_statistics": "generate_statistics_node",
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
    {"continue": "synthesize_dataset", "end_error": "error_node"},
)


# Final step
workflow.add_edge("synthesize_dataset", END)  # Successful completion ends here
workflow.add_edge("error_node", END)  # Error path ends here
workflow.add_edge("generate_statistics", END)
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
    elif final_state.get("synthesized_dataset") is not None:
        print("Synthesized Dataset:")
        # Pretty print the final dataset
        print(json.dumps(final_state["synthesized_dataset"], indent=2))
        print(f"\nTotal records generated: {len(final_state['synthesized_dataset'])}")
    else:
        print(
            "Workflow finished, but no dataset was generated (e.g., no relevant documents found or data extracted)."
        )
