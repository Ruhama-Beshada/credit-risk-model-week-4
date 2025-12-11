from pydantic import BaseModel
from typing import List, Optional

class CustomerData(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    # add all numeric features used by the model
    # feature names must match exactly what the model expects

class PredictionResponse(BaseModel):
    is_high_risk: int
    probability: float
