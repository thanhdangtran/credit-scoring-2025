import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "VNPT Credit Scoring API"
        assert "version" in data

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data

    def test_liveness(self):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestPredictEndpoints:
    def test_predict_single(self):
        payload = {
            "customer": {
                "customer_id": "TEST001",
                "age": 35,
                "monthly_income": 15000000,
                "cic_score": 680,
                "credit_utilization": 0.3
            },
            "return_reason_codes": False
        }

        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "prediction" in data

        prediction = data["prediction"]
        assert prediction["customer_id"] == "TEST001"
        assert 0 <= prediction["probability_of_default"] <= 1
        assert 300 <= prediction["credit_score"] <= 850
        assert prediction["risk_band"] in ["A", "B", "C", "D", "E"]

    def test_predict_with_reason_codes(self):
        payload = {
            "customer": {
                "customer_id": "TEST002",
                "age": 25,
                "cic_score": 550,
                "credit_utilization": 0.7,
                "max_dpd_12m": 45
            },
            "return_reason_codes": True
        }

        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        prediction = data["prediction"]
        assert prediction["reason_codes"] is not None
        assert len(prediction["reason_codes"]) > 0

    def test_batch_predict(self):
        payload = {
            "customers": [
                {
                    "customer_id": "BATCH001",
                    "age": 30,
                    "cic_score": 700
                },
                {
                    "customer_id": "BATCH002",
                    "age": 45,
                    "cic_score": 600
                }
            ],
            "return_reason_codes": False
        }

        response = client.post("/api/v1/batch_predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["total_count"] == 2
        assert len(data["predictions"]) == 2

    def test_predict_invalid_age(self):
        payload = {
            "customer": {
                "customer_id": "INVALID",
                "age": 150  # Invalid age
            }
        }

        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestReasonCodeEndpoints:
    def test_get_reason_codes(self):
        response = client.get(
            "/api/v1/reason_codes/TEST001",
            params={
                "cic_score": 550,
                "credit_utilization": 0.8,
                "max_dpd_12m": 60
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["customer_id"] == "TEST001"
        assert len(data["reason_codes"]) > 0


class TestMetricsEndpoints:
    def test_get_metrics(self):
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "performance" in data
        assert "total_predictions" in data["performance"]

    def test_prometheus_metrics(self):
        response = client.get("/api/v1/metrics/prometheus")
        assert response.status_code == 200
        assert "credit_scoring_predictions_total" in response.text

    def test_reset_metrics(self):
        response = client.post("/api/v1/metrics/reset")
        assert response.status_code == 200
        assert response.json()["status"] == "metrics reset"
