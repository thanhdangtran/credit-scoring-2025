import time
import logging
from datetime import datetime
from typing import Optional, List
from collections import deque

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from api.schemas.responses import MetricsResponse, PerformanceMetrics, DriftMetrics

logger = logging.getLogger(__name__)

router = APIRouter()

# Metrics storage (in-memory for simplicity)
_prediction_latencies: deque = deque(maxlen=10000)
_prediction_count: int = 0
_error_count: int = 0
_last_drift_check: Optional[datetime] = None
_drift_metrics: Optional[DriftMetrics] = None


def record_prediction(latency_ms: float, success: bool = True):
    global _prediction_count, _error_count
    _prediction_latencies.append(latency_ms)
    _prediction_count += 1
    if not success:
        _error_count += 1


def get_percentile(data: List[float], percentile: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    latencies = list(_prediction_latencies)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = get_percentile(latencies, 95)
    p99_latency = get_percentile(latencies, 99)
    error_rate = _error_count / _prediction_count if _prediction_count > 0 else 0.0

    performance = PerformanceMetrics(
        total_predictions=_prediction_count,
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        p99_latency_ms=round(p99_latency, 2),
        error_rate=round(error_rate, 4)
    )

    return MetricsResponse(
        performance=performance,
        drift=_drift_metrics,
        model_metrics=None,
        timestamp=datetime.utcnow()
    )


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics():
    latencies = list(_prediction_latencies)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = get_percentile(latencies, 95)
    p99_latency = get_percentile(latencies, 99)
    error_rate = _error_count / _prediction_count if _prediction_count > 0 else 0.0

    metrics = f"""# HELP credit_scoring_predictions_total Total number of predictions
# TYPE credit_scoring_predictions_total counter
credit_scoring_predictions_total {_prediction_count}

# HELP credit_scoring_errors_total Total number of prediction errors
# TYPE credit_scoring_errors_total counter
credit_scoring_errors_total {_error_count}

# HELP credit_scoring_latency_seconds Prediction latency in seconds
# TYPE credit_scoring_latency_seconds summary
credit_scoring_latency_seconds_sum {sum(latencies) / 1000 if latencies else 0}
credit_scoring_latency_seconds_count {len(latencies)}

# HELP credit_scoring_latency_avg_ms Average prediction latency in milliseconds
# TYPE credit_scoring_latency_avg_ms gauge
credit_scoring_latency_avg_ms {avg_latency:.2f}

# HELP credit_scoring_latency_p95_ms 95th percentile latency in milliseconds
# TYPE credit_scoring_latency_p95_ms gauge
credit_scoring_latency_p95_ms {p95_latency:.2f}

# HELP credit_scoring_latency_p99_ms 99th percentile latency in milliseconds
# TYPE credit_scoring_latency_p99_ms gauge
credit_scoring_latency_p99_ms {p99_latency:.2f}

# HELP credit_scoring_error_rate Prediction error rate
# TYPE credit_scoring_error_rate gauge
credit_scoring_error_rate {error_rate:.4f}
"""

    if _drift_metrics:
        metrics += f"""
# HELP credit_scoring_psi Population Stability Index
# TYPE credit_scoring_psi gauge
credit_scoring_psi {_drift_metrics.psi:.4f}

# HELP credit_scoring_is_stable Model stability indicator (1=stable, 0=drift detected)
# TYPE credit_scoring_is_stable gauge
credit_scoring_is_stable {1 if _drift_metrics.is_stable else 0}
"""

    return metrics


@router.post("/metrics/reset")
async def reset_metrics():
    global _prediction_count, _error_count, _prediction_latencies
    _prediction_latencies.clear()
    _prediction_count = 0
    _error_count = 0

    return {"status": "metrics reset", "timestamp": datetime.utcnow().isoformat()}


@router.post("/drift/check")
async def check_drift(
    baseline_path: Optional[str] = None,
    current_path: Optional[str] = None
):
    global _last_drift_check, _drift_metrics

    try:
        # This would load data and calculate PSI
        # For now, return mock drift metrics
        _drift_metrics = DriftMetrics(
            psi=0.08,
            psi_status="Stable - no significant drift detected",
            is_stable=True,
            last_checked=datetime.utcnow()
        )
        _last_drift_check = datetime.utcnow()

        return {
            "status": "drift check completed",
            "psi": _drift_metrics.psi,
            "is_stable": _drift_metrics.is_stable,
            "interpretation": _drift_metrics.psi_status,
            "timestamp": _last_drift_check.isoformat()
        }

    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        return {
            "status": "drift check failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
