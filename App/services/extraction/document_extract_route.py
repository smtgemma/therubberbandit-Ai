from fastapi import APIRouter, UploadFile, File
from .document_extract_schema import DocumentExtractInput
from .document_extract import extract_logo_text_gemini
import json

router = APIRouter(prefix="/document-extract", tags=["Document Extraction"])

@router.post("/extract-logo")
async def extract_logo_from_document(
    files: UploadFile = File(..., description="Upload an image or document file to extract logo text")
):
    """
    Extract text from logos in an uploaded document using Gemini.
    Accepts image or document files and returns extracted logo text.
    """
    try:
        # Read the uploaded file
        file_content = await files.read()
        file_name = files.filename
        content_type = files.content_type
        
        # Call Gemini to extract logo text
        extracted_data = await extract_logo_text_gemini(
            file_content=file_content,
            file_name=file_name,
            content_type=content_type
        )
        
        return {
            "status": "success",
            "file_name": file_name,
            "data": extracted_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
