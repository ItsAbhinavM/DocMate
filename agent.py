import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
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
    interpreted_schema: Optional[InterpretedSchema]
    retrieved_docs: Optional[List[Document]]  # Docs from vectorstore
    extracted_data_points: List[ExtractedItem]  # List of raw extractions
    synthesized_dataset: Optional[List[Dict]]  # Final structured dataset
    error_message: Optional[str]
    retry_count: int


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")


# --- Define Graph Nodes ---
def interpret_query_node(state: GraphState):
    """Interprets the user query to define schema and retrieval query."""
    print("--- [Node: Interpret Query] ---")
    user_query = state["original_query"]

    parser = PydanticOutputParser(pydantic_object=InterpretedSchema)

    prompt = PromptTemplate(
        template="""Analyze the user's query to define a structured dataset schema and a concise search query.
User Query: {query}

Based on the query, define:
1.  `schema_description`: A dictionary describing the columns/fields the user wants in their dataset. Keys should be concise field names, values should be clear descriptions.
2.  `retrieval_query`: A query string optimized for finding relevant documents/chunks in a vector database related to the user's request. Focus on key entities, actions, and concepts.

{format_instructions}
""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    try:
        interpreted_output = chain.invoke({"query": user_query})
        print(f"Interpreted Schema: {interpreted_output.schema_description}")
        print(f"Retrieval Query: {interpreted_output.retrieval_query}")
        return {"interpreted_schema": interpreted_output, "error_message": None}
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

# from langchain_core.runnables import Runnable
from typing import List
from langchain_core.output_parsers import StrOutputParser

# LLM chain that we’ll define below
# llm_schema_validator: Runnable

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def check_synthesized_dataset_node(state: GraphState):
    retry_count = state.get("retry_count", 0)
    if retry_count < 3:    
        print("--- [Node: Check Synthesized Dataset Quality] ---")
        synthesized = state.get("synthesized_dataset", [])
        original_query = state.get("original_query", "")

        if not synthesized:
            print("[WARN] Synthesized dataset is empty.")
            return "retry_schema"

        prompt = PromptTemplate(
            template="""
    You are a data quality evaluator.

    The user asked: "{original_query}"

    The following dataset was synthesized from document extraction:

    {synthesized_dataset}

    Evaluate this dataset on:
    - Relevance to the user query
    - Presence of duplicate or redundant entries
    - Completeness and clarity

    Is this a good final dataset for the user's intent?
    Respond with a single word: "yes" or "no".
    """,
            input_variables=["original_query", "synthesized_dataset"]
        )

        chain = prompt | llm | StrOutputParser()

        try:
            decision = chain.invoke({
                "original_query": original_query,
                "synthesized_dataset": json.dumps(synthesized, indent=2)
            }).strip().lower()

            if "yes" in decision:
                print("[GOOD FINAL DATASET]")
                return "good_schema"
            else:
                print("[BAD FINAL DATASET]")
                return "retry_schema"
        except Exception as e:
            print(f"[ERROR] Failed to check synthesized dataset: {e}")
            return "retry_schema"

    else:
        print("Exceeded 3...TERMINATING")
        return "end"


# Add a retry node
def re_interpret_query_node(state: GraphState):
    print("--- [Node: Re-Interpret Query] ---")
    failed_schema = state.get("interpreted_schema")
    retrieved_docs = state.get("retrieved_docs")
    bad_results = state.get("extracted_data_points")
    retry_count = state.get("retry_count", 0)

    feedback = f"""
    Previous Schema: {json.dumps(failed_schema.schema_description)}
    Retrieved Docs Count: {len(retrieved_docs or [])}
    Extracted Samples: {[item.data for item in bad_results[:2]]}
    Problem: The previous attempt to synthesize a dataset failed to meet quality standards. Please refine the schema better.
    """

    print("here is the feedback")
    print(feedback)

    parser = PydanticOutputParser(pydantic_object=InterpretedSchema)

    prompt = PromptTemplate(
        template="""Revise the structured schema and retrieval query based on feedback.
User Query: {query}

Feedback from prior run:
{feedback}

Generate:
1.  `schema_description`: A refined dictionary of the data fields the user likely wants.
2.  `retrieval_query`: A better-optimized vector search query for this need.

{format_instructions}
""",
        input_variables=["query", "feedback"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        new_interpretation = chain.invoke(
            {"query": state["original_query"], "feedback": feedback}
        )
        print(f"[Retry] New Schema: {new_interpretation.schema_description}")
        return {"interpreted_schema": new_interpretation, "retry_count": retry_count + 1, "error_message": None}
    except Exception as e:
        print(f"[Retry ERROR] Failed to reinterpret query: {e}")
        return {"error_message": f"Retry failed: {str(e)}"}




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


def should_continue(state: GraphState):
    """Determines whether to continue processing or end due to errors."""
    if state.get("error_message"):
        print(
            f"--- [Edge Logic] --- Error encountered: {state['error_message']}. Ending workflow."
        )
        return "end_error"
    print("--- [Edge Logic] --- No errors detected. Continuing workflow.")
    return "continue"


# --- Build the Graph ---
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("interpret_query", interpret_query_node)
workflow.add_node("retrieve_documents", retrieve_documents_node)
workflow.add_node("extract_data", extract_data_node)
workflow.add_node("synthesize_dataset", synthesize_dataset_node)
workflow.add_node(
    "error_node", lambda state: print("Workflow terminated due to error.")
)
# workflow.add_node("check_schema_success", check_schema_success_node)
workflow.add_node("re_interpret_query", re_interpret_query_node)

# Define edges
workflow.set_entry_point("interpret_query")

workflow.add_conditional_edges(
    "interpret_query",
    should_continue,
    {
        "continue": "retrieve_documents",
        "end_error": "error_node",
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

# workflow.add_conditional_edges(
#     "extract_data",
#     should_continue,
#     {"continue": "check_schema_success", "end_error": "error_node"},
# )

workflow.add_conditional_edges(
    "synthesize_dataset",
    check_synthesized_dataset_node,
    {
        "retry_schema": "re_interpret_query",
        "end": END,
        "good_schema": END,
    },
)

workflow.add_conditional_edges(
    "re_interpret_query",
    check_synthesized_dataset_node,
    {
        "retry_schema": "re_interpret_query",
        "end": END,
        "good_schema": END,
    },
)


# Final step
# workflow.add_edge("synthesize_dataset", END)  # Successful completion ends here
workflow.add_edge("error_node", END)  # Error path ends here

# Compile the graph
graph = workflow.compile()

if __name__ == "__main__":
    print("\n--- Running Sample Workflow ---")

    # 1. User Query Input:
    initial_state = {"original_query": "What is my name"}
    print(f"Initial Query: {initial_state['original_query']}")

    # 2. Run the Graph:
    final_state = graph.invoke(
        initial_state, {"recursion_limit": 10}
    )  # Add recursion limit

    # 4. Final Output:
    #    The `final_state` dictionary contains the results of the workflow.
    print("\n--- Workflow Complete ---")
    print("here is the final state\n")
    for key in final_state.keys():
        print(final_state[key])
    # print(final_state)
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
