import json

import aiofiles
from fastapi import FastAPI, Request, UploadFile
from pydantic import BaseModel

from agent import graph

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
    async with aiofiles.open(f"uploads/{file.filename}", "wb") as local_file:
        content = await file.read()
        await local_file.write(content)
    return f"{request.base_url}uploads/{file.filename}"
