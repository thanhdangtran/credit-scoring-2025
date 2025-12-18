from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")

    # Demographic features
    age: Optional[int] = Field(None, ge=18, le=100, description="Customer age")
    gender: Optional[str] = Field(None, description="Gender (M/F)")
    education: Optional[str] = Field(None, description="Education level")
    marital_status: Optional[str] = Field(None, description="Marital status")
    province: Optional[str] = Field(None, description="Province code")

    # Financial features
    monthly_income: Optional[float] = Field(None, ge=0, description="Monthly income (VND)")
    employment_type: Optional[str] = Field(None, description="Employment type")
    employment_years: Optional[float] = Field(None, ge=0, description="Years of employment")
    dti_ratio: Optional[float] = Field(None, ge=0, le=1, description="Debt-to-income ratio")
    savings_amount: Optional[float] = Field(None, ge=0, description="Total savings (VND)")

    # Credit features
    cic_score: Optional[int] = Field(None, ge=300, le=850, description="CIC credit score")
    nhnn_loan_group: Optional[int] = Field(None, ge=1, le=5, description="NHNN loan group (1-5)")
    total_loans: Optional[int] = Field(None, ge=0, description="Total number of loans")
    credit_utilization: Optional[float] = Field(None, ge=0, le=1, description="Credit utilization ratio")
    max_dpd_12m: Optional[int] = Field(None, ge=0, description="Max days past due in 12 months")

    # Telecom features
    telecom_tenure_months: Optional[int] = Field(None, ge=0, description="Telecom tenure in months")
    arpu: Optional[float] = Field(None, ge=0, description="Average revenue per user")
    payment_rate: Optional[float] = Field(None, ge=0, le=1, description="Payment rate")

    # Additional features as dict
    additional_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional custom features"
    )

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v is not None and v.upper() not in ['M', 'F', 'MALE', 'FEMALE']:
            raise ValueError('Gender must be M, F, MALE, or FEMALE')
        return v.upper() if v else v

    def to_feature_dict(self) -> Dict[str, Any]:
        features = self.model_dump(exclude={'customer_id', 'additional_features'})
        features = {k: v for k, v in features.items() if v is not None}

        if self.additional_features:
            features.update(self.additional_features)

        return features

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST001",
                "age": 35,
                "gender": "M",
                "monthly_income": 15000000,
                "cic_score": 680,
                "nhnn_loan_group": 1,
                "credit_utilization": 0.35,
                "max_dpd_12m": 0
            }
        }


class PredictRequest(BaseModel):
    customer: CustomerFeatures = Field(..., description="Customer features for prediction")
    return_reason_codes: bool = Field(
        default=False,
        description="Include reason codes in response"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Specific model version to use (default: production)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "customer": {
                    "customer_id": "CUST001",
                    "age": 35,
                    "monthly_income": 15000000,
                    "cic_score": 680
                },
                "return_reason_codes": True
            }
        }


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(
        ...,
        description="List of customers for batch prediction",
        min_length=1,
        max_length=1000
    )
    return_reason_codes: bool = Field(
        default=False,
        description="Include reason codes in response"
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Specific model version to use"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    {
                        "customer_id": "CUST001",
                        "age": 35,
                        "monthly_income": 15000000,
                        "cic_score": 680
                    },
                    {
                        "customer_id": "CUST002",
                        "age": 28,
                        "monthly_income": 10000000,
                        "cic_score": 620
                    }
                ],
                "return_reason_codes": False
            }
        }
