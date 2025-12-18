import os
import time
import logging
from typing import Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.schemas.requests import PredictRequest, BatchPredictRequest, CustomerFeatures
from api.schemas.responses import (
    PredictResponse,
    BatchPredictResponse,
    PredictionResult,
    ReasonCode,
    ReasonCodeResponse,
)
from api.routes.health import set_model_status

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model cache
_model = None
_model_version = "1.0.0"
_scorecard = None


def load_model(model_path: Optional[str] = None):
    global _model, _model_version, _scorecard

    try:
        # Try loading from MLflow registry first
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        model_name = os.getenv("MODEL_NAME", "vnpt-credit-scoring")

        if mlflow_uri:
            try:
                from mlops.registry import ModelRegistry
                registry = ModelRegistry(mlflow_uri)
                _model = registry.load_production_model(model_name)
                prod_version = registry.get_production_model(model_name)
                _model_version = prod_version.version if prod_version else "unknown"
                logger.info(f"Loaded model from MLflow: {model_name} v{_model_version}")
            except Exception as e:
                logger.warning(f"Could not load from MLflow: {e}")

        # Fallback to local model file
        if _model is None and model_path:
            import joblib
            _model = joblib.load(model_path)
            logger.info(f"Loaded model from local file: {model_path}")

        if _model is not None:
            set_model_status(True, _model_version)
            return True

        logger.warning("No model loaded - predictions will use mock scoring")
        return False

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def get_risk_rating(pd_value: float) -> tuple:
    if pd_value < 0.02:
        return "Very Low Risk", "A"
    elif pd_value < 0.05:
        return "Low Risk", "B"
    elif pd_value < 0.10:
        return "Medium Risk", "C"
    elif pd_value < 0.20:
        return "High Risk", "D"
    else:
        return "Very High Risk", "E"


def pd_to_score(pd_value: float, base_score: int = 600, pdo: int = 50) -> int:
    odds = (1 - pd_value) / max(pd_value, 0.0001)
    score = base_score + pdo * np.log(odds) / np.log(2)
    return int(np.clip(score, 300, 850))


def generate_reason_codes(
    features: dict,
    n_reasons: int = 4
) -> list:
    reasons = []

    # Rule-based reason code generation
    if features.get('credit_utilization', 0) > 0.5:
        reasons.append(ReasonCode(
            code="RC01",
            feature="credit_utilization",
            description="Tỷ lệ sử dụng tín dụng cao",
            impact=-15.0
        ))

    if features.get('max_dpd_12m', 0) > 30:
        reasons.append(ReasonCode(
            code="RC02",
            feature="max_dpd_12m",
            description="Có lịch sử chậm thanh toán",
            impact=-20.0
        ))

    if features.get('cic_score', 700) < 600:
        reasons.append(ReasonCode(
            code="RC03",
            feature="cic_score",
            description="Điểm tín dụng CIC thấp",
            impact=-25.0
        ))

    if features.get('nhnn_loan_group', 1) >= 3:
        reasons.append(ReasonCode(
            code="RC04",
            feature="nhnn_loan_group",
            description="Nhóm nợ NHNN cao",
            impact=-30.0
        ))

    if features.get('dti_ratio', 0) > 0.5:
        reasons.append(ReasonCode(
            code="RC05",
            feature="dti_ratio",
            description="Tỷ lệ nợ trên thu nhập cao",
            impact=-12.0
        ))

    if features.get('employment_years', 5) < 1:
        reasons.append(ReasonCode(
            code="RC06",
            feature="employment_years",
            description="Thời gian làm việc ngắn",
            impact=-8.0
        ))

    if features.get('age', 35) < 25:
        reasons.append(ReasonCode(
            code="RC07",
            feature="age",
            description="Độ tuổi trẻ, ít lịch sử tín dụng",
            impact=-5.0
        ))

    # Sort by impact and return top N
    reasons.sort(key=lambda x: x.impact)
    return reasons[:n_reasons]


def predict_single(
    features: dict,
    return_reason_codes: bool = False
) -> PredictionResult:
    customer_id = features.pop('customer_id', 'unknown')

    # Use model if available, otherwise mock scoring
    if _model is not None:
        try:
            df = pd.DataFrame([features])
            if hasattr(_model, 'predict_proba'):
                pd_value = _model.predict_proba(df)[0, 1]
            else:
                pd_value = _model.predict(df)[0]
        except Exception as e:
            logger.warning(f"Model prediction failed: {e}, using mock scoring")
            pd_value = mock_score(features)
    else:
        pd_value = mock_score(features)

    # Convert to score and rating
    credit_score = pd_to_score(pd_value)
    risk_rating, risk_band = get_risk_rating(pd_value)

    # Generate reason codes if requested
    reason_codes = None
    if return_reason_codes:
        reason_codes = generate_reason_codes(features)

    return PredictionResult(
        customer_id=customer_id,
        probability_of_default=round(pd_value, 6),
        credit_score=credit_score,
        risk_rating=risk_rating,
        risk_band=risk_band,
        reason_codes=reason_codes
    )


def mock_score(features: dict) -> float:
    base_pd = 0.10

    # Adjust based on CIC score
    cic = features.get('cic_score', 650)
    if cic >= 700:
        base_pd *= 0.5
    elif cic < 550:
        base_pd *= 2.0

    # Adjust based on loan group
    loan_group = features.get('nhnn_loan_group', 1)
    base_pd *= (1 + (loan_group - 1) * 0.3)

    # Adjust based on DPD
    dpd = features.get('max_dpd_12m', 0)
    if dpd > 90:
        base_pd *= 3.0
    elif dpd > 30:
        base_pd *= 1.5

    # Adjust based on income
    income = features.get('monthly_income', 10000000)
    if income > 30000000:
        base_pd *= 0.7
    elif income < 5000000:
        base_pd *= 1.3

    return min(max(base_pd, 0.001), 0.999)


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start_time = time.time()

    try:
        features = request.customer.to_feature_dict()
        features['customer_id'] = request.customer.customer_id

        prediction = predict_single(
            features,
            return_reason_codes=request.return_reason_codes
        )

        latency_ms = (time.time() - start_time) * 1000

        return PredictResponse(
            success=True,
            prediction=prediction,
            model_version=_model_version,
            timestamp=datetime.utcnow(),
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch_predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    start_time = time.time()

    predictions = []
    failed_count = 0

    for customer in request.customers:
        try:
            features = customer.to_feature_dict()
            features['customer_id'] = customer.customer_id

            prediction = predict_single(
                features,
                return_reason_codes=request.return_reason_codes
            )
            predictions.append(prediction)

        except Exception as e:
            logger.warning(f"Failed prediction for {customer.customer_id}: {e}")
            failed_count += 1

    latency_ms = (time.time() - start_time) * 1000

    return BatchPredictResponse(
        success=True,
        predictions=predictions,
        total_count=len(predictions),
        failed_count=failed_count,
        model_version=_model_version,
        timestamp=datetime.utcnow(),
        latency_ms=round(latency_ms, 2)
    )


@router.get("/reason_codes/{customer_id}", response_model=ReasonCodeResponse)
async def get_reason_codes(
    customer_id: str,
    cic_score: int = 650,
    credit_utilization: float = 0.3,
    max_dpd_12m: int = 0,
    nhnn_loan_group: int = 1
):
    features = {
        'cic_score': cic_score,
        'credit_utilization': credit_utilization,
        'max_dpd_12m': max_dpd_12m,
        'nhnn_loan_group': nhnn_loan_group
    }

    reason_codes = generate_reason_codes(features, n_reasons=4)

    return ReasonCodeResponse(
        success=True,
        customer_id=customer_id,
        reason_codes=reason_codes,
        timestamp=datetime.utcnow()
    )
