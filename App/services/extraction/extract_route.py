from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
from .gemini_extractor import GeminiExtractor

router = APIRouter(prefix="/extraction", tags=["Extraction"])

# Use Gemini extractor exclusively for this route
gemini_extractor = GeminiExtractor()


@router.post("/upload")
async def upload_and_extract(
    files: List[UploadFile] = File(..., description="Upload document files (PDF, PNG, JPEG, TIFF)")
):
    mime_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }

    # Validate file types
    for file in files:
        ext = "." + file.filename.lower().rsplit(".", 1)[-1]
        if ext not in mime_map:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Please upload PDF, PNG, JPEG, or TIFF."
            )

    # Process files using the Gemini extractor
    try:
        parsed = await gemini_extractor.extract_quote_data(files)

        # Attempt simple APR detection on returned raw text
        detected_apr = None
        try:
            import re
            raw_text = ""
            if isinstance(parsed, dict):
                raw_text = parsed.get("extracted_text", {}).get("raw_text", "") or ""
            apr_pattern = re.compile(r"(\d{1,2}\.\d{1,2})\s*%")
            matches = apr_pattern.findall(raw_text)
            apr_candidates = [float(a) for a in matches if float(a) < 20]
            if apr_candidates:
                detected_apr = min(apr_candidates)
        except Exception:
            detected_apr = None

        # Return the Gemini-parsed structure with APR included
        if isinstance(parsed, dict):
            parsed["detected_apr"] = detected_apr
            return parsed
        else:
            return {"result": parsed, "detected_apr": detected_apr}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
