"""
FastAPI application for the Graph-Based Money Muling Detection Engine.

Endpoints:
    POST /upload  — Accept CSV, return detection results as JSON
    GET  /health  — System health check
    GET  /metrics — Processing statistics
"""

import logging
import sys
import os

# Add the parent directory (backend) to sys.path to resolve imports like 'api', 'services', etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import threading
from api.routes import router
from services.processing_pipeline import warmup_ml_model

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Graph-Based Money Muling Detection Engine",
    description="Detects money muling rings using graph-based analysis of transaction data.",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compress large JSON responses (graph payloads/history) to reduce network latency.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(router)


@app.on_event("startup")
async def _startup_warmup():
    logger.info("App starting up. Launching background library warmup...")
    threading.Thread(target=warmup_ml_model, daemon=True).start()
