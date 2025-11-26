from pydantic import BaseModel
from typing import List, Optional, Union

class FormField(BaseModel):
    name: str
    value: str
    confidence: float

class LogoText(BaseModel):
    text: str
    confidence: Optional[float] = None

class ExtractResponse(BaseModel):
    text: str
    form_fields: Optional[List[FormField]] = []
    logo_text: Optional[List[LogoText]] = []
    detected_apr: Optional[Union[str, float]] = None # <-- new