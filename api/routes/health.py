import time
from datetime import datetime
from fastapi import APIRouter, Response

from api.schemas.responses import HealthResponse

router = APIRouter()

# Track service start time
SERVICE_START_TIME = time.time()

# Global state
_model_loaded = False
_model_version = None


def set_model_status(loaded: bool, version: str = None):
    global _model_loaded, _model_version
    _model_loaded = loaded
    _model_version = version


@router.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - SERVICE_START_TIME

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=_model_loaded,
        model_version=_model_version,
        uptime_seconds=uptime,
        timestamp=datetime.utcnow()
    )


@router.get("/health/live")
async def liveness():
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness():
    if not _model_loaded:
        return Response(
            content='{"status": "not ready", "reason": "model not loaded"}',
            status_code=503,
            media_type="application/json"
        )
    return {"status": "ready"}
