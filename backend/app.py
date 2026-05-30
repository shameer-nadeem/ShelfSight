from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from inference import run_inference

app = FastAPI(title="ShelfSight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ShelfSight backend running"}

@app.post("/analyse")
async def analyse(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = run_inference(temp_path)
        return JSONResponse(content=result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
