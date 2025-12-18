from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ReasonCode(BaseModel):
    code: str = Field(..., description="Reason code identifier")
    feature: str = Field(..., description="Feature name")
    description: str = Field(..., description="Human-readable description")
    impact: float = Field(..., description="Impact on score (negative = worse)")


class PredictionResult(BaseModel):
    customer_id: str = Field(..., description="Customer identifier")
    probability_of_default: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of default (0-1)"
    )
    credit_score: int = Field(
        ...,
        ge=300,
        le=850,
        description="Credit score (300-850)"
    )
    risk_rating: str = Field(..., description="Risk rating category")
    risk_band: str = Field(..., description="Risk band (A-E)")
    reason_codes: Optional[List[ReasonCode]] = Field(
        default=None,
        description="Top adverse action reasons"
    )


class PredictResponse(BaseModel):
    success: bool = Field(default=True)
    prediction: PredictionResult
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = Field(None, description="Processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "prediction": {
                    "customer_id": "CUST001",
                    "probability_of_default": 0.05,
                    "credit_score": 720,
                    "risk_rating": "Low Risk",
                    "risk_band": "A",
                    "reason_codes": [
                        {
                            "code": "RC01",
                            "feature": "credit_utilization",
                            "description": "High credit utilization ratio",
                            "impact": -15.5
                        }
                    ]
                },
                "model_version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "latency_ms": 45.2
            }
        }


class BatchPredictResponse(BaseModel):
    success: bool = Field(default=True)
    predictions: List[PredictionResult]
    total_count: int = Field(..., description="Total predictions made")
    failed_count: int = Field(default=0, description="Number of failed predictions")
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: Optional[float] = Field(None, description="Total processing time")


class ReasonCodeResponse(BaseModel):
    success: bool = Field(default=True)
    customer_id: str
    reason_codes: List[ReasonCode]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    uptime_seconds: float = Field(..., description="Service uptime")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "model_version": "1.0.0",
                "uptime_seconds": 3600.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid request",
                "status_code": 400,
                "timestamp": "2024-01-15T10:30:00Z",
                "details": {"field": "age", "message": "Must be between 18 and 100"}
            }
        }


class DriftMetrics(BaseModel):
    psi: float = Field(..., description="Population Stability Index")
    psi_status: str = Field(..., description="PSI interpretation")
    is_stable: bool = Field(..., description="Whether distribution is stable")
    last_checked: datetime = Field(..., description="Last drift check timestamp")


class PerformanceMetrics(BaseModel):
    total_predictions: int = Field(..., description="Total predictions made")
    avg_latency_ms: float = Field(..., description="Average latency in ms")
    p95_latency_ms: float = Field(..., description="95th percentile latency")
    p99_latency_ms: float = Field(..., description="99th percentile latency")
    error_rate: float = Field(..., description="Error rate (0-1)")


class MetricsResponse(BaseModel):
    performance: PerformanceMetrics
    drift: Optional[DriftMetrics] = None
    model_metrics: Optional[Dict[str, float]] = Field(
        None,
        description="Model performance metrics (AUC, KS, etc.)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
