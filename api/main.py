import os
import time
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.routes import predict, health, monitoring
from api.schemas.responses import ErrorResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Global model cache
model_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting VNPT Credit Scoring API...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")

    # Load model on startup (optional)
    try:
        from api.routes.predict import load_model
        load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Model not loaded on startup: {e}")

    yield

    logger.info("Shutting down VNPT Credit Scoring API...")


# Create FastAPI application
app = FastAPI(
    title="VNPT Credit Scoring API",
    description="""
    API for Vietnamese Credit Scoring System.

    ## Features
    - Real-time credit scoring predictions
    - Batch prediction support
    - Reason code generation for adverse actions
    - Model performance monitoring
    - PSI/CSI drift detection

    ## Endpoints
    - `/predict` - Single prediction
    - `/batch_predict` - Batch predictions
    - `/reason_codes` - Get adverse action reasons
    - `/health` - Health check
    - `/metrics` - Prometheus metrics
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            timestamp=datetime.utcnow().isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            timestamp=datetime.utcnow().isoformat()
        ).model_dump()
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(monitoring.router, prefix="/api/v1", tags=["Monitoring"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "VNPT Credit Scoring API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENVIRONMENT", "development") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
