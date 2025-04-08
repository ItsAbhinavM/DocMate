import json
import os

import aiofiles
from fastapi import FastAPI, Request, UploadFile
from pydantic import BaseModel

from agent import graph
from filehandler import pdf_driver, csv_driver, xlsx_driver, image_driver, doc_driver, docx_driver

app = FastAPI()


class Prompt(BaseModel):
    content: str


@app.post("/send_prompt")
def send_prompt(prompt: Prompt):
    initial_state = {"original_query": prompt.content}
    final_state = graph.invoke(initial_state, {"recursion_limit": 10})
    if final_state.get("error_message"):
        return f"Workflow failed with error: {final_state['error_message']}"
    elif final_state.get("synthesized_dataset") is not None:
        return json.dumps(final_state["synthesized_dataset"], indent=2)
    else:
        return "Workflow finished, but no dataset was generated (e.g., no relevant documents found or data extracted)."


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