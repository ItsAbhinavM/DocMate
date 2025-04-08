import json
import os
import uuid
from typing import Optional

import aiofiles
from fastapi import FastAPI, HTTPException, Request, UploadFile
from pydantic import BaseModel

from agent import graph
from filehandler import (csv_driver, doc_driver, docx_driver, image_driver,
                         pdf_driver, xlsx_driver)

app = FastAPI()
active_runs = {}


class Prompt(BaseModel):
    content: str


# --- Request/Response Models ---
class InvokeRequest(BaseModel):
    original_query: str


class InvokeResponse(BaseModel):
    run_id: str
    status: str  # e.g., "running", "waiting_clarification", "complete", "error"
    message: Optional[str] = None  # e.g., the clarification question
    final_dataset: Optional[list] = None  # Only present on completion


class RespondRequest(BaseModel):
    run_id: str
    user_feedback: str


class RespondResponse(BaseModel):
    run_id: str
    status: str  # e.g., "running", "complete", "error"
    message: Optional[str] = None
    final_dataset: Optional[list] = None


@app.post("/send_prompt")
async def send_prompt(request: InvokeRequest):
    run_id = str(uuid.uuid4())
    print(f"Starting run_id: {run_id} for query: '{request.original_query}'")

    # Configuration for LangGraph state checkpointing
    config = {"configurable": {"thread_id": run_id}}

    # Initial state for this run
    initial_state = {"original_query": request.original_query, "current_iteration": 0}

    try:
        # Use stream to process step-by-step and handle interrupts
        async for chunk in graph.astream(
            initial_state, config=config, stream_mode="updates"
        ):
            # chunk is a dictionary where keys are node names and values are the output state *after* that node ran
            node_name = list(chunk.keys())[0]
            current_state: GraphState = list(chunk.values())[
                0
            ]  # Get the state dictionary

            print(f"Run {run_id}: Just executed node -> {node_name}")

            # --- Check if clarification is needed AFTER interpretation ---
            # We check the state *after* the node that *might* trigger the interrupt condition
            if node_name == "interpret_query":
                if current_state.get("needs_clarification"):
                    print(f"Run {run_id}: Pausing for clarification.")
                    interpreted = current_state.get("interpreted_schema", None)
                    if interpreted:
                        question = interpreted.ambiguity
                        active_runs[run_id] = {
                            "status": "waiting_clarification",
                            "question": question or "Could you be more specific",
                        }
                        return InvokeResponse(
                            run_id=run_id,
                            status="waiting_clarification",
                            message=question,
                        )

        # If the stream finishes without interruption or after resuming
        print(f"Run {run_id}: Workflow finished.")
        final_state = graph.get_state(config)
        dataset = final_state.values.get("synthesized_dataset")
        error = final_state.values.get("error_message")

        if error:
            active_runs[run_id] = {"status": "error", "message": error}
            # Raise HTTPException for API error, or return structured error response
            # raise HTTPException(status_code=500, detail=error)
            return InvokeResponse(run_id=run_id, status="error", message=error)
        else:
            active_runs[run_id] = {"status": "complete", "dataset": dataset}
            return InvokeResponse(
                run_id=run_id, status="complete", final_dataset=dataset
            )

    except Exception as e:
        print(f"Run {run_id}: Error during workflow execution: {e}")
        active_runs[run_id] = {"status": "error", "message": str(e)}
        # Depending on the error, you might want different status codes
        raise HTTPException(
            status_code=500, detail=f"Workflow execution error: {str(e)}"
        )
@app.get("/document_statistics")
async def get_statistics():
    """Endpoint to get statistics about the documents in the system."""
    try:
        # Configuration for LangGraph state checkpointing
        run_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": run_id}}
        
        # Initial minimal state just for statistics
        initial_state = {"original_query": "generate_statistics", "current_iteration": 0}
         
        stats_result = await graph.acall(
            inputs=initial_state,
            config=config,
            return_only="generate_statistics"
        )
        
        if stats_result.get("error_message"):
            raise HTTPException(status_code=500, detail=stats_result["error_message"])
            
        return {
            "statistics": stats_result["statistics"],
            "visualization": stats_result["visualization"]
        }
    except Exception as e:
        print(f"Error generating statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")

@app.post("/respond", response_model=RespondResponse)
async def respond_to_workflow(request: RespondRequest):
    """Provides user feedback to a paused workflow instance."""
    run_id = request.run_id
    user_feedback = request.user_feedback
    print(f"Received response for run_id: {run_id}")

    if (
        run_id not in active_runs
        or active_runs[run_id]["status"] != "waiting_clarification"
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Run ID '{run_id}' not found or not waiting for clarification.",
        )

    config = {"configurable": {"thread_id": run_id}}

    # --- Prepare the state update ---
    # We ONLY provide the fields that need updating to resume.
    # The checkpointer loads the rest of the state automatically.
    update_state = {
        "current_query": user_feedback,  # Update the query to be interpreted next
    }

    active_runs[run_id] = {"status": "running"}  # Update status

    try:
        # Resume the stream, passing ONLY the update dictionary
        async for chunk in graph.astream(
            update_state, config=config, stream_mode="updates"
        ):
            node_name = list(chunk.keys())[0]
            current_state: GraphState = list(chunk.values())[0]

            # --- Check if clarification is needed AFTER interpretation ---
            # We check the state *after* the node that *might* trigger the interrupt condition
            if node_name == "interpret_query":
                if current_state.get("needs_clarification"):
                    print(f"Run {run_id}: Pausing for clarification.")
                    interpreted = current_state.get("interpreted_schema", None)
                    if interpreted:
                        question = interpreted.ambiguity
                        active_runs[run_id] = {
                            "status": "waiting_clarification",
                            "question": question or "Could you be more specific",
                        }
                        return InvokeResponse(
                            run_id=run_id,
                            status="waiting_clarification",
                            message=question,
                        )

            print(f"Run {run_id} (Resumed): Just executed node -> {node_name}")

        # If the stream finishes after resuming
        print(f"Run {run_id}: Workflow finished after resuming.")
        final_state = graph.get_state(config)
        dataset = final_state.values.get("synthesized_dataset")
        error = final_state.values.get("error_message")

        if error:
            active_runs[run_id] = {"status": "error", "message": error}
            # raise HTTPException(status_code=500, detail=error)
            return RespondResponse(run_id=run_id, status="error", message=error)
        else:
            active_runs[run_id] = {"status": "complete", "dataset": dataset}
            return RespondResponse(
                run_id=run_id, status="complete", final_dataset=dataset
            )

    except Exception as e:
        print(f"Run {run_id}: Error during workflow resumption: {e}")
        active_runs[run_id] = {"status": "error", "message": str(e)}
        raise HTTPException(
            status_code=500, detail=f"Workflow resumption error: {str(e)}"
        )


@app.post("/file_upload")
async def upload_image(file: UploadFile, request: Request):
    file_path = f"uploads/{file.filename}"
    async with aiofiles.open(file_path, "wb") as local_file:
        content = await file.read()
        await local_file.write(content)

    ext = os.path.splitext(file.filename)[1].lower()

    match ext:
        case ".pdf":
            pdf_driver(file_path)
        case ".csv":
            csv_driver(file_path)
        case ".xlsx":
            xlsx_driver(file_path)
        case ".doc":
            doc_driver(file_path)
        case ".docx":
            docx_driver(file_path)
        case _:
            return {"error": f"Unsupported file type: {ext}"}

    return f"{request.base_url}uploads/{file.filename}"

