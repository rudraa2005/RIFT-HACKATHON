"""
API Routes — upload, health, and metrics endpoints.

Extracted from app/main.py for cleaner separation.
"""

import io
import time

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.processing_pipeline import ProcessingService
from utils.metrics import MetricsTracker
from utils.validators import validate_csv

router = APIRouter()
metrics_tracker = MetricsTracker()


@router.get("/health")
async def health():
    """Return system health status."""
    return {"status": "healthy", "version": "1.0.0"}


@router.get("/metrics")
async def metrics():
    """Return processing statistics from the most recent run."""
    return metrics_tracker.get_metrics()


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept CSV upload, perform graph-based money muling detection,
    and return a structured JSON response.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    validation_error = validate_csv(df)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)

    start_time = time.time()
    service = ProcessingService()
    result = service.process(df)
    processing_time = round(time.time() - start_time, 2)

    result["summary"]["processing_time_seconds"] = processing_time
    metrics_tracker.record(result["summary"])

    return JSONResponse(content=result)
