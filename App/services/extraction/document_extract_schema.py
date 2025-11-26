from pydantic import BaseModel
from typing import Optional, List

class LogoTextExtraction(BaseModel):
    logo_text: List[str]
    confidence: float
    raw_response: Optional[dict] = None

class DocumentExtractInput(BaseModel):
    file_path: str
    extract_logos: bool = True
    extract_text: bool = False