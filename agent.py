import json
import os
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict

import fitz
from colorama import Fore, Style, init
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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from filehandler import vectorstore

# Initialize colorama for cross-platform color support
init(autoreset=True)
# Initialize Rich console
console = Console()

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
    page_contents: Optional[Dict]

    # Output
    processed_dataset: Optional[List[Dict]]  # Final structured dataset
    retry_count: int


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21")
saver = MemorySaver()


# --- Helper Functions for Print Formatting ---
def print_header(text):
    """Print a beautifully formatted header."""
    console.print(f"\n{Fore.CYAN}{'=' * 80}")
    console.print(Panel(f"[bold cyan]{text}[/bold cyan]", expand=False))
    console.print(f"{Fore.CYAN}{'=' * 80}\n")


def print_subheader(text):
    """Print a beautifully formatted subheader."""
    console.print(f"\n{Fore.MAGENTA}{'-' * 60}")
    console.print(f"[bold magenta]✨ {text} ✨[/bold magenta]")
    console.print(f"{Fore.MAGENTA}{'-' * 60}\n")


def print_success(text):
    """Print a success message."""
    console.print(f"[bold green]✅ {text}[/bold green]")


def print_info(text):
    """Print an informational message."""
    console.print(f"[blue]ℹ️ {text}[/blue]")


def print_warning(text):
    """Print a warning message."""
    console.print(f"[bold yellow]⚠️ {text}[/bold yellow]")


def print_error(text):
    """Print an error message."""
    console.print(f"[bold red]❌ {text}[/bold red]")


def print_data_summary(data, title="Data Summary"):
    """Print a summary of data in a pretty table."""
    if not data:
        print_info("No data to display")
        return

    table = Table(title=title)

    # Add columns based on the first item keys
    if isinstance(data, list) and data:
        sample = data[0]
        for key in sample.keys():
            table.add_column(key, style="cyan")

        # Add up to 5 rows as preview
        preview_limit = min(5, len(data))
        for i in range(preview_limit):
            row = [str(data[i].get(key, ""))[:50] for key in sample.keys()]
            table.add_row(*row)

        if len(data) > preview_limit:
            table.add_row(*["..." for _ in sample.keys()])

        console.print(table)
        console.print(f"[bold blue]Total records: {len(data)}[/bold blue]")
    elif isinstance(data, dict):
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        for key, value in data.items():
            table.add_row(key, str(value)[:100])

        console.print(table)


def print_progress_bar(completed, total, description="Progress"):
    """Print a progress bar."""
    with tqdm(total=total, desc=description, ncols=80) as pbar:
        pbar.update(completed)


# --- Define Graph Nodes ---
def root_node(state: GraphState):
    """Agent start point, increments the iteration count on each loop"""
    iteration = state.get("current_iteration", 0) + 1
    print_header(f"🚀 STARTING ITERATION {iteration}")
    print_info(f"User query: {state.get('current_query') or state['original_query']}")

    if iteration > 1:
        print_info("Reprocessing after clarification...")

    return {
        "current_iteration": iteration,
        "error_message": None,  # Clear previous errors on loop/restart
        "needs_clarification": False,  # Default to no clarification needed
    }


def interpret_query_node(state: GraphState):
    """Interprets the user query to define schema and retrieval query."""
    print_subheader("🧠 QUERY INTERPRETATION")
    user_query = state.get("current_query") or state["original_query"]
    run_mode = state["run_mode"]

    print_info(f"Mode: {run_mode.upper()}")
    print_info(f'Query: "{user_query}"')

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
4. `needs_statistics`: Set to true if the user is requesting statistics or analysis about the document collection itself, rather than data extraction. For example, requests like "show me statistics of the documents", "how many PDFs do we have?", or "what's the average document length?" would set this to true.
5.  `ambiguity`: Briefly explain *why* clarification is needed if the query is vague (e.g., "Refinement target unclear", "Filter criteria ambiguous"). Otherwise, null.
"""

    else:  # initial_generation
        template += """The user wants to generate a dataset from scratch.
Analyze the query to define:
1.  `schema_description`: A dictionary describing the columns/fields for the new dataset. Keys = concise field names, values = clear descriptions.
2.  `retrieval_query`: A query string optimized for finding relevant documents in a vector database. Focus on key entities, actions, and concepts. This should *not* be null for initial generation unless the query is completely nonsensical.
3.  `refinement_instructions`: Should be null for initial generation.
4. `needs_statistics`: Set to true if the user is requesting statistics or analysis about the document collection itself, rather than data extraction. For example, requests like "show me statistics of the documents", "how many PDFs do we have?", or "what's the average document length?" would set this to true.
5.  `ambiguity`: Briefly explain *why* clarification is needed if the query is vague (e.g., "Query is too general", "Specific entities not mentioned"). Otherwise, null.
"""

    template += "\n{format_instructions}\n"

    prompt = PromptTemplate(
        template=template,
        input_variables=["query", "run_mode"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    print_info("🔄 Interpreting query with AI...")
    chain = prompt | llm | parser
    try:
        interpreted_output = chain.invoke({"query": user_query, "run_mode": run_mode})

        # Print schema as table
        schema_table = Table(title="📊 Interpreted Schema")
        schema_table.add_column("Field", style="cyan")
        schema_table.add_column("Description", style="green")

        for field, description in interpreted_output.schema_description.items():
            schema_table.add_row(field, description)

        console.print(schema_table)

        # Print other interpretation details
        if interpreted_output.ambiguity:
            print_warning(f" 🤔 Ambiguity Detected: {interpreted_output.ambiguity}")
        else:
            print_info(f"🤔 Ambiguity: None")

        if interpreted_output.retrieval_query:
            print_info(f'🔍 Retrieval Query: "{interpreted_output.retrieval_query}"')
        else:
            print_info("🔍 No search query generated")

        if interpreted_output.refinement_instructions:
            print_info(
                f"🔧 Refinement Plan: {interpreted_output.refinement_instructions}"
            )

        # Basic validation
        if (
            run_mode == "initial_generation"
            and not interpreted_output.retrieval_query
            and not interpreted_output.ambiguity
        ):
            print_warning(
                "Initial generation mode but no retrieval query was generated. Setting ambiguity."
            )
            interpreted_output.ambiguity = (
                "Initial generation requires a search query, but none could be derived."
            )

        needs_clarification = bool(interpreted_output.ambiguity)
        if needs_clarification:
            print_warning("👋 Clarification needed before proceeding")
        else:
            print_success("Interpretation completed successfully")

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
        print_error(f"Failed to interpret query: {e}")
        return {"error_message": f"Failed to interpret query: {str(e)}"}


def retrieve_documents_node(state: GraphState):
    """Retrieves documents from the vector store based on the interpreted query."""
    print_subheader("📚 VECTOR SEARCH")
    if state.get("error_message"):  # Skip if previous step failed
        print_error("Skipping retrieval due to previous error")
        return {}
    if not state.get("interpreted_schema"):
        print_error("Interpretation step did not produce schema/query")
        return {"error_message": "Interpretation step did not produce schema/query."}

    retrieval_query = state["interpreted_schema"].retrieval_query
    if not retrieval_query:
        print_info("No retrieval query specified - skipping document retrieval")
        return {"retrieved_docs": [], "extracted_data_points": []}

    try:
        print_info(f'🔍 Searching for documents with query: "{retrieval_query}"')

        documents = vectorstore.max_marginal_relevance_search(
            retrieval_query, k=1, fetch_k=50
        )
        if not documents:
            print_warning("No documents found matching the query")
            return {
                "retrieved_docs": [],
                "extracted_data_points": [],
            }

        print_success(f"Found {len(documents)} relevant documents")

        # Show brief preview of document sources
        source_table = Table(title="📑 Retrieved Documents")
        source_table.add_column("Index", style="cyan", justify="right")
        source_table.add_column("Source", style="green")
        source_table.add_column("Content Preview", style="blue")

        scanned_pages = set()
        page_contents = {}
        for i, doc in enumerate(documents):  # Show first 5 docs
            source = doc.metadata.get("source")
            pg_no = doc.metadata.get("page_number")
            if source and pg_no and pg_no not in scanned_pages:
                with fitz.open(f"uploads/{source}") as doc:
                    if pg_no < 1 or pg_no > len(doc):
                        raise ValueError("Page number out of range")
                    page_text = doc[pg_no - 1].get_text()
                    page_contents[i] = page_text
                scanned_pages.add(pg_no)

            content_preview = documents[i].page_content[:500] + "..."
            source_table.add_row(str(i + 1), source, content_preview)

        console.print(source_table)

        return {
            "retrieved_docs": documents,
            "page_contents": page_contents,
            "error_message": None,
        }
    except Exception as e:
        print_error(f"Failed during vector store query: {e}")
        return {"error_message": f"Failed during vector store query: {str(e)}"}


# from langchain_core.runnables import Runnable
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# LLM chain that we’ll define below
# llm_schema_validator: Runnable



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
            input_variables=["original_query", "synthesized_dataset"],
        )

        chain = prompt | llm | StrOutputParser()

        try:
            decision = (
                chain.invoke(
                    {
                        "original_query": original_query,
                        "synthesized_dataset": json.dumps(synthesized, indent=2),
                    }
                )
                .strip()
                .lower()
            )

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
        return {
            "interpreted_schema": new_interpretation,
            "retry_count": retry_count + 1,
            "error_message": None,
        }
    except Exception as e:
        print(f"[Retry ERROR] Failed to reinterpret query: {e}")
        return {"error_message": f"Retry failed: {str(e)}"}


def extract_data_node(state: GraphState):
    """Extracts structured data from retrieved documents based on the schema."""
    print_subheader("🔍 DATA EXTRACTION")
    if state.get("error_message"):  # Skip if error or no docs
        print_error("Skipping extraction due to previous error")
        # If no docs, ensure extracted_data_points is initialized
        if "extracted_data_points" not in state:
            return {"extracted_data_points": []}
        return {}  # Pass existing state if error occurred

    if not state.get("retrieved_docs"):
        print_info("No documents to extract from")
        return {"extracted_data_points": []}

    original_query = state["original_query"]
    schema_dict = state["interpreted_schema"].schema_description
    documents = state["retrieved_docs"]
    page_contents = state.get("page_contents")

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

    print_info(f"📊 Processing {len(page_contents)} documents for data extraction...")

    with tqdm(total=len(page_contents), desc="Extracting data", ncols=80) as pbar:
        for i in page_contents.keys():
            doc = documents[i]
            source = doc.metadata.get("source", "Unknown")[:30]
            pbar.set_description(f"Processing: {source}...")

            try:
                # Prepare input for the extraction chain
                schema_desc_string = json.dumps(schema_dict, indent=2)
                input_data = {
                    "original_query": original_query,
                    "schema_description": schema_desc_string,
                    "document_content": page_contents[i],
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
                    elif isinstance(extracted_data, list):
                        for entry in extracted_data:
                            if isinstance(entry, dict) and entry:
                                item = ExtractedItem(
                                    data=entry, source_document_info=doc.metadata
                                )
                                extracted_items.append(item)
                except json.JSONDecodeError:
                    print_warning(
                        f"  LLM output was not valid JSON from document {i+1}"
                    )
                except Exception as parse_err:
                    print_warning(
                        f"  Could not parse extracted data from document {i+1}: {parse_err}"
                    )

            except Exception as e:
                print_warning(f"  Failed to extract from document {i+1}: {e}")
                # Continue to next document

            pbar.update(1)

    # Show extraction summary
    if extracted_items:
        print_success(f"Successfully extracted {len(extracted_items)} data points")

        # Show a sample of extracted data
        if len(extracted_items) > 0:
            sample_table = Table(title="📝 Sample Extracted Data")

            # Get keys from first item
            first_item = extracted_items[0].data
            for key in first_item.keys():
                sample_table.add_column(key, style="cyan")

            # Add up to 3 rows for preview
            for item in extracted_items[:3]:
                row_data = [
                    str(item.data.get(key, ""))[:30] for key in first_item.keys()
                ]
                sample_table.add_row(*row_data)

            if len(extracted_items) > 3:
                sample_table.add_row(*["..." for _ in first_item.keys()])

            console.print(sample_table)
    else:
        print_warning("No data points were extracted from the documents")

    return {"extracted_data_points": extracted_items, "error_message": None}


def process_data_node(state: GraphState):
    print_subheader("⚙️ DATA PROCESSING")
    if state.get("error_message"):
        print_error("Skipping processing due to previous error")
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
        print_info(
            "🔄 Mode: Initial Generation - Creating dataset from extracted points"
        )
        # Basic synthesis: collect data dictionaries
        processed_dataset = [item.data for item in newly_extracted_points]
        print_success(f"Synthesized dataset with {len(processed_dataset)} records")

    elif run_mode == "refinement":
        print_info("🔄 Mode: Refinement - Applying changes to previous dataset")
        if not previous_dataset:
            print_warning(
                "Refinement mode, but no previous dataset was provided. Using only newly extracted points."
            )
            processed_dataset = [item.data for item in newly_extracted_points]
        else:
            print_info(f"Newly extracted points: {len(newly_extracted_points)}")
            print_info(f"Refinement Instructions: {refinement_instructions}")

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

**Output:** Only return the final refined dataset as a valid JSON list. Do not use triple backtick return the raw json.
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
                print_info("🔄 Calling AI for refinement processing...")

                processed_dataset = refinement_chain.invoke(refinement_input)
                print_success(
                    f"AI refinement resulted in {len(processed_dataset)} records"
                )

            except Exception as e:
                print_error(f"Failed to process refinement: {e}")
                # Log raw output for debugging
                try:
                    raw_output = (refinement_prompt | llm | StrOutputParser()).invoke(
                        refinement_input
                    )
                    print_warning(
                        f"Raw LLM Output (first 200 chars): {raw_output[:200]}..."
                    )
                except:
                    pass

                # Handle error: Maybe return original + new, or fail.
                processed_dataset = previous_dataset + [
                    item.data for item in newly_extracted_points
                ]  # Naive fallback
                state["error_message"] = (
                    f"LLM refinement output parsing failed: {e}"  # Propagate soft error
                )

    # Display final dataset summary
    if processed_dataset:
        print_data_summary(processed_dataset, title="FINAL DATASET PREVIEW")
    else:
        print_warning("No data was processed in the final step")

    return {"processed_dataset": processed_dataset, "error_message": None}

def generate_statistics_node(state: GraphState):
    """Generates statistics about the available documents."""
    print_subheader("📈 DOCUMENT STATISTICS")

    try:
        # Import necessary libraries
        import numpy as np
        import matplotlib.pyplot as plt
        import io
        import base64
        import os
        from filehandler import vectorstore  # Import vectorstore directly
        import glob
        
        # Create output directory
        output_dir = "statistics_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Manual document stats collection if get_document_stats() isn't returning the right format
        print_info("Collecting document statistics directly...")
        
        # Get documents directly from the uploads directory
        upload_dir = "uploads"
        pdf_files = glob.glob(f"{upload_dir}/*.pdf")
        txt_files = glob.glob(f"{upload_dir}/*.txt")
        docx_files = glob.glob(f"{upload_dir}/*.docx")
        other_files = glob.glob(f"{upload_dir}/*.*")
        
        # Filter out already counted files from other_files
        other_extensions = set(os.path.splitext(f)[1] for f in other_files) - {'.pdf', '.txt', '.docx'}
        other_files = [f for f in other_files if os.path.splitext(f)[1] in other_extensions]
        
        documents = []
        
        # Process PDF files
        for pdf_path in pdf_files:
            try:
                filename = os.path.basename(pdf_path)
                file_size = os.path.getsize(pdf_path)
                page_count = 0
                token_estimate = 0
                
                # Try to get page count
                try:
                    import fitz  # PyMuPDF
                    with fitz.open(pdf_path) as doc:
                        page_count = len(doc)
                        # Estimate tokens (very rough approximation)
                        text_length = sum(len(page.get_text()) for page in doc)
                        token_estimate = text_length // 4  # Rough estimate: 4 chars per token
                except Exception as e:
                    print_warning(f"Could not analyze PDF {filename}: {e}")
                
                doc_info = {
                    "name": filename,
                    "path": pdf_path,
                    "type": "PDF",
                    "size_bytes": file_size,
                    "pages": page_count,
                    "tokens": token_estimate,
                    "structured": False  # Default assumption
                }
                documents.append(doc_info)
            except Exception as e:
                print_warning(f"Error processing PDF {pdf_path}: {e}")
        
        # Process TXT files
        for txt_path in txt_files:
            try:
                filename = os.path.basename(txt_path)
                file_size = os.path.getsize(txt_path)
                
                # Count lines and estimate tokens
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    try:
                        content = f.read()
                        line_count = content.count('\n') + 1
                        token_estimate = len(content.split()) # Word count as token estimate
                    except:
                        line_count = 0
                        token_estimate = file_size // 4  # Fallback estimate
                
                doc_info = {
                    "name": filename,
                    "path": txt_path,
                    "type": "TXT",
                    "size_bytes": file_size,
                    "lines": line_count,
                    "tokens": token_estimate,
                    "structured": False
                }
                documents.append(doc_info)
            except Exception as e:
                print_warning(f"Error processing TXT {txt_path}: {e}")
        
        # Process DOCX files (basic stats)
        for docx_path in docx_files:
            try:
                filename = os.path.basename(docx_path)
                file_size = os.path.getsize(docx_path)
                
                # Try to get word count
                word_count = 0
                try:
                    import docx
                    doc = docx.Document(docx_path)
                    word_count = sum(len(p.text.split()) for p in doc.paragraphs)
                except:
                    word_count = file_size // 20  # Very rough estimate
                
                doc_info = {
                    "name": filename,
                    "path": docx_path,
                    "type": "DOCX",
                    "size_bytes": file_size,
                    "tokens": word_count,
                    "structured": True  # Docx often has structure
                }
                documents.append(doc_info)
            except Exception as e:
                print_warning(f"Error processing DOCX {docx_path}: {e}")
        
        # Process other files (just basic info)
        for other_path in other_files:
            try:
                filename = os.path.basename(other_path)
                file_type = os.path.splitext(filename)[1].upper().replace('.', '')
                file_size = os.path.getsize(other_path)
                
                doc_info = {
                    "name": filename,
                    "path": other_path,
                    "type": file_type if file_type else "Unknown",
                    "size_bytes": file_size,
                    "tokens": file_size // 10,  # Very rough estimate
                    "structured": False
                }
                documents.append(doc_info)
            except Exception as e:
                print_warning(f"Error processing file {other_path}: {e}")
        
        # Try to get vector store statistics
        try:
            if hasattr(vectorstore, 'get_document_count'):
                vector_count = vectorstore.get_document_count()
                print_info(f"Vector store contains {vector_count} document chunks")
            elif hasattr(vectorstore, '__len__'):
                vector_count = len(vectorstore)
                print_info(f"Vector store contains {vector_count} document chunks")
            else:
                vector_count = "Unknown"
                print_warning("Could not determine vector store document count")
        except Exception as e:
            vector_count = "Error"
            print_warning(f"Error accessing vector store: {e}")
        
        # Check if we found documents
        total_docs = len(documents)
        if total_docs == 0:
            print_warning("No documents found in uploads directory")
            return {
                "statistics": {"total_docs": 0, "documents": []},
                "error_message": "No documents available for statistics generation"
            }
        
        print_success(f"Found {total_docs} documents")
        
        # Continue with statistics generation similar to before
        # Calculate structured vs unstructured
        structured_counts = sum(1 for doc in documents if doc.get("structured", False))
        unstructured_counts = total_docs - structured_counts
        
        # Calculate file types distribution
        file_types = {}
        for doc in documents:
            doc_type = doc.get("type", "Unknown")
            file_types.setdefault(doc_type, 0)
            file_types[doc_type] += 1
        
        # Calculate token counts by type
        token_counts_by_type = {}
        for doc in documents:
            doc_type = doc.get("type", "Unknown")
            tokens = doc.get("tokens", 0)
            token_counts_by_type.setdefault(doc_type, []).append(tokens)
        
        total_token_by_type = {k: sum(v) for k, v in token_counts_by_type.items()}
        
        # Get top heavy docs
        top_heavy_docs = sorted(
            documents,
            key=lambda d: d.get("tokens", 0),
            reverse=True
        )[:5]
        
        # Create visualizations (now with 4 plots in a 2x2 grid)
        plt.figure(figsize=(14, 10))
        
        # 1. Structured vs Unstructured Pie
        plt.subplot(2, 2, 1)
        plt.pie(
            [structured_counts, unstructured_counts],
            labels=["Structured", "Unstructured"],
            autopct="%1.1f%%",
            startangle=140
        )
        plt.title("Document Structure Types")
        
        # 2. File Types Distribution
        plt.subplot(2, 2, 2)
        plt.bar(file_types.keys(), file_types.values())
        plt.title("Document Format Types")
        plt.xticks(rotation=45)
        
        # 3. Token Load by Type
        plt.subplot(2, 2, 3)
        plt.bar(total_token_by_type.keys(), total_token_by_type.values())
        plt.title("Total Tokens by Document Type")
        plt.xticks(rotation=45)
        
        # 4. Top Heavy Docs
        plt.subplot(2, 2, 4)
        names = [doc.get("name", f"Doc {i}")[:20] for i, doc in enumerate(top_heavy_docs)]
        tokens = [doc.get("tokens", 0) for doc in top_heavy_docs]
        plt.barh(names, tokens)
        plt.title("Top 5 Largest Documents (by token count)")
        
        plt.tight_layout()
        
        # Save visualization
        filename = "document_analytics.png"
        file_path = os.path.join(output_dir, filename)
        plt.savefig(file_path)
        print_success(f"Statistics visualization saved to {file_path}")
        
        # Convert to base64 for visualization in frontend
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # Generate text summary
        summary = {
            "total_documents": total_docs,
            "structured_documents": structured_counts,
            "unstructured_documents": unstructured_counts,
            "document_types": file_types,
            "total_tokens_by_type": total_token_by_type,
            "vector_store_documents": vector_count,
            "largest_documents": [
                {
                    "name": doc.get("name", "Unknown"), 
                    "type": doc.get("type", "Unknown"),
                    "tokens": doc.get("tokens", 0)
                } 
                for doc in top_heavy_docs
            ]
        }
        
        print_data_summary(summary, title="DOCUMENT STATISTICS SUMMARY")
        
        # Create combined statistics object
        doc_stats = {
            "total_docs": total_docs,
            "documents": documents,
            "summary": summary
        }
        
        return {
            "statistics": doc_stats,
            "summary": summary,
            "visualization": img_str,
            "image_path": file_path,
            "error_message": None
        }

    except Exception as e:
        print_error(f"Failed to generate statistics: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full traceback for debugging
        return {"error_message": f"Failed to generate statistics: {str(e)}"} 


# -- Functions for conditional edges --
def should_continue(state: GraphState) -> Literal["continue", "end_error"]:
    """Determines whether to continue processing or end due to errors."""
    if state.get("error_message"):
        print_error(f"Error encountered: {state['error_message']}. Ending workflow.")
        return "end_error"
    print_info("No errors detected. Continuing workflow.")
    return "continue"


def decide_after_interpret(
    state: GraphState,
) -> Literal["proceed_to_retrieve", "loop_for_clarification", "handle_error"]:
    """Routes flow after interpretation based on errors or need for clarification."""
    print_subheader("🧭 WORKFLOW DECISION POINT")
    if state.get("error_message"):
        print_error("Error occurred during interpretation")
        return "handle_error"

    if state.get("needs_statistics"):
        print("Decision: Statistics requested. Routing to statistics node")
        return "generate_statistics"

    if state.get("needs_clarification"):
        print_warning("Clarification needed. Routing back to root")
        # The API will pause *before* this edge routes. When it resumes,
        # the graph follows this path back to increment/interpret.
        return "loop_for_clarification"

    if not state.get("interpreted_schema"):
        # Safety check: Should not happen if no error and no clarification needed
        print_error("Interpretation successful, but schema missing unexpectedly")
        state["error_message"] = (
            "Interpretation node finished without error but schema is missing."
        )
        return "handle_error"

    if state["run_mode"] == "refinement":
        print_success(
            "Interpretation successful (refinement mode). Proceeding to processing."
        )
        return "proceed_to_processing"

    print_success("Interpretation successful. Proceeding to retrieve documents.")
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
workflow.add_node("generate_statistics",generate_statistics_node)
workflow.add_node(
    "error_node", lambda state: print_error("⛔ Workflow terminated due to error.")
)
# workflow.add_node("check_schema_success", check_schema_success_node)
workflow.add_node("re_interpret_query", re_interpret_query_node)

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
        "handle_error": "error_node",
        "generate_statistics": "generate_statistics",
        "end_error": "error_node",
    },
)
workflow.add_edge("process_data",'generate_statistics')

workflow.add_conditional_edges(
    "process_data",
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
    },
)

workflow.add_conditional_edges(
    "extract_data",
    should_continue,
    {"continue": "process_data", "end_error": "error_node"},
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
