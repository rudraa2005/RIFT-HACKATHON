"""
API Routes — upload, health, and metrics endpoints.

Extracted from app/main.py for cleaner separation.
"""

import io
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from utils.history_store import HistoryStore
from utils.metrics import MetricsTracker

router = APIRouter()
metrics_tracker = MetricsTracker()
history_store = HistoryStore()


@router.get("/health")
async def health():
    """Return system health status."""
    return {"status": "healthy", "version": "1.0.0"}


@router.get("/metrics")
async def metrics():
    """Return processing statistics from the most recent run."""
    return metrics_tracker.get_metrics()


@router.get("/history")
async def history():
    """Return historical upload runs from persistent storage."""
    return {"status": "ready", "items": history_store.list_runs(limit=200)}


@router.get("/history/{run_id}")
async def history_report(run_id: int):
    """Return the stored JSON report for a historical run."""
    report = history_store.get_run_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="History report not found.")
    return {"status": "ready", "report": report}


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
        import pandas as pd
        df = pd.read_csv(
            io.BytesIO(contents),
            parse_dates=["timestamp"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    from utils.validators import validate_csv
    validation_error = validate_csv(df)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)

    start_time = time.time()
    from services.processing_pipeline import ProcessingService
    service = ProcessingService()
    result = service.process(df)
    processing_time = round(time.time() - start_time, 2)

    result["summary"]["processing_time_seconds"] = processing_time
    metrics_tracker.record(result["summary"])
    history_store.record_run(
        filename=file.filename,
        file_size_bytes=len(contents),
        report=result,
    )

    return JSONResponse(content=result)
