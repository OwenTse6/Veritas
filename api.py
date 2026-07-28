"""
api.py

Local API service for slopcheck.

Run with:
    uvicorn api:app --reload

Endpoints:
    POST /check/text   {"text": "..."}
    POST /check/image  multipart file upload
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from core import score_text, score_image

app = FastAPI(title="slopcheck", description="Heuristic, explainable AI-content detector.")


class TextRequest(BaseModel):
    text: str


@app.post("/check/text")
def check_text(request: TextRequest):
    return score_text(request.text)


@app.post("/check/image")
async def check_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = score_image(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/check/text", "/check/image"]}
