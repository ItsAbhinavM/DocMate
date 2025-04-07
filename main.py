import json

import aiofiles
from fastapi import FastAPI, Request, UploadFile
from pydantic import BaseModel

from agent import agent_executor

app = FastAPI()


class Prompt(BaseModel):
    content: str


@app.post("/send_prompt")
def send_prompt(prompt: Prompt):
    output = agent_executor.invoke({"input": prompt.content})["output"]
    if "direct_response" in output:
        return json.loads(output)
    else:
        response = {}
        response["message"] = output
        return response


@app.post("/file_upload")
async def upload_image(file: UploadFile, request: Request):
    async with aiofiles.open(f"uploads/{file.filename}", "wb") as local_file:
        content = await file.read()
        await local_file.write(content)
    return f"{request.base_url}uploads/{file.filename}"
