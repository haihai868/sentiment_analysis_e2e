from pydantic import BaseModel, Field
from typing import List, Optional

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=280)
    
class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100)
    
class PredictionResponse(BaseModel):
    clean_text: str
    sentiment: str
    probabilities: dict
    
class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str]