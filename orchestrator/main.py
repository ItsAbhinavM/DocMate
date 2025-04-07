from fastapi import FastAPI, UploadFile, Request
import json
import aiofiles
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import agent_executor

app = FastAPI()

# Serve pictures from the Media directory
app.mount("/media", StaticFiles(directory="Media"), name="media")

class Prompt(BaseModel):
    content: str

@app.post("/send_prompt")
def send_prompt(prompt: Prompt):
    output = agent_executor.invoke({"input":prompt.content})["output"]
    if "direct_response" in output:
        return json.loads(output)
    else:
        response = {}
        response["message"] = output
        return response

@app.post("/images/upload")
async def upload_image(file: UploadFile, request: Request):
    async with aiofiles.open(f"Media/{file.filename}", "wb") as local_file:
        content  = await file.read()
        await local_file.write(content)
    return f"{request.base_url}media/{file.filename}"
