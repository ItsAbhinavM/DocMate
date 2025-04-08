from fastapi import FastAPI, UploadFile, Request
import json
import aiofiles
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import agent_executor
import os
from filehandler import pdf_driver, csv_driver

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

@app.post("/upload_file")
async def upload_file(file: UploadFile, request: Request):
    file_path = f"Media/{file.filename}"
    async with aiofiles.open(file_path, "wb") as local_file:
        content  = await file.read()
        await local_file.write(content)

    ext = os.path.splitext(file.filename)[1].lower()

    match ext:
        case ".pdf":
            pdf_driver(file_path)
        case ".csv":
            csv_driver(file_path)
        case _:
            return {"error": f"Unsupported file type: {ext}"}
    return f"{request.base_url}media/{file.filename}"

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# from typing import Optional

# class ItemOut(BaseModel):
#     name: str
#     price: float

# @app.get("/item/{item_id}", response_model=ItemOut)
# def get_item(item_id: int):
#     return {"name": "Apple", "price": 0.99, "extra_field": "ignored"}

