import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from app import pitch, match
from datetime import datetime

STARTUP_TIME = datetime.now()

app = FastAPI(
    title = "Scarlatti555",
    description = "QbH for classical music",
    version = "0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def root():
    return {"status":"ok", "message":"Scarlatti555 is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "corpus_size": len(match.TEST_CORPUS),
        "corpus_titles": [theme["title"] for theme in match.TEST_CORPUS],
        "last_started": STARTUP_TIME.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/identify")
async def identify(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail = f"file must be audio, got {file.content_type}" 
        )
    
    with tempfile.NamedTemporaryFile(suffix = ".wav", delete = False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        intervals = pitch.process(tmp_path)
        if len(intervals) == 0:
            raise HTTPException(
                status_code = 422,
                detail = "no pitched audio detected -- hum again"
            )
        results = match.match(intervals)
        return {
            "results": results,
            "intervals_detected": len(intervals),
            "interval_sequence": intervals.tolist()
        }
    
    finally:
        os.unlink(tmp_path)