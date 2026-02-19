"""
FastAPI application for the Graph-Based Money Muling Detection Engine.

Endpoints:
    POST /upload  — Accept CSV, return detection results as JSON
    GET  /health  — System health check
    GET  /metrics — Processing statistics

Time Complexity: Dominated by processing pipeline (see services/processing_pipeline.py)
Memory: O(V + E) during processing, released after response
"""

import io
import time

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from services.processing_pipeline import ProcessingService
from utils.metrics import MetricsTracker
from utils.validators import validate_csv

app = FastAPI(
    title="Graph-Based Money Muling Detection Engine",
    description="Detects money muling rings using graph-based analysis of transaction data.",
    version="1.0.0",
)

metrics_tracker = MetricsTracker()


@app.get("/health")
async def health():
    """Return system health status."""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metrics")
async def metrics():
    """Return processing statistics from the most recent run."""
    return metrics_tracker.get_metrics()


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accept CSV upload, perform graph-based money muling detection,
    and return a structured JSON response with suspicious accounts,
    fraud rings, and a processing summary.
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
