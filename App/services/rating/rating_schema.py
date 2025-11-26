# rating_schema.py
from pydantic import BaseModel
from typing import List, Optional, Union, Any

class FormField(BaseModel):
    name: str
    value: Optional[Union[str, float]] = None
    confidence: Optional[float] = None

class LogoText(BaseModel):
    text: str
    confidence: Optional[float] = None
    
class DealInput(BaseModel):
    # Accept both the extractor output and previous shapes
    text: Optional[str] = None
    form_fields: Optional[List[FormField]] = None
    logo_text: Optional[Union[str, List[LogoText]]] = None
    detected_apr: Optional[Union[str, float]] = None
    raw: Optional[Any] = None

    class Config:
        extra = "allow"

