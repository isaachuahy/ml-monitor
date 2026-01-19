from pydantic import BaseModel, Field
from typing import Dict, Optional
from uuid import UUID

class PredictionRequest(BaseModel):
    # Features for our Credit Risk model
    income: float = Field(..., gt=0, description="Annual income of the applicant")
    debt: float = Field(..., ge=0, description="Total current debt")
    credit_score: int = Field(..., ge=300, le=850, description="FICO Credit Score")

class PredictionResponse(BaseModel):
    request_id: UUID
    prediction_prob: float # between 0 and 1
    prediction_class: int # 0 or 1
    model_version: str