from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
from .extract import extract_text_sync
from .extract_schema import ExtractResponse
from App.core.config import settings
from google.cloud import documentai
from google.oauth2 import service_account
import img2pdf
from typing import List

router = APIRouter(prefix="/extraction", tags=["Extraction"])

credentials = service_account.Credentials.from_service_account_file(
    settings.gcp_key_path
)
client = documentai.DocumentProcessorServiceClient(credentials=credentials)

PROCESSOR_NAME = f"projects/{settings.gcp_project_id}/locations/{settings.gcp_location}/processors/{settings.gcp_processor_id}"

def get_text_from_text_anchor(document_text, text_anchor):
    if not text_anchor or not text_anchor.text_segments:
        return ""
    segment = text_anchor.text_segments[0]
    start = segment.start_index or 0
    end = segment.end_index or 0
    return document_text[start:end]

@router.post("/upload", response_model=ExtractResponse)
async def upload_and_extract(files: List[UploadFile] = File(...)):
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

    # Process files
    try:
        if len(files) == 1 and files[0].filename.lower().endswith('.pdf'):
            # Single PDF file
            contents = await files[0].read()
            mime_type = "application/pdf"
        else:
            # Multiple images - convert to PDF
            image_contents = []
            for file in files:
                content = await file.read()
                image_contents.append(content)
            # Convert images to PDF
            contents = img2pdf.convert(image_contents)
            mime_type = "application/pdf"

        raw_doc = documentai.RawDocument(content=contents, mime_type=mime_type)
        request = documentai.ProcessRequest(
            name=PROCESSOR_NAME, raw_document=raw_doc
        )
        result = client.process_document(request=request)
        document = result.document

        form_fields = []
        logo_text = []
        apr_candidates = [] 


        for page in document.pages:
            # Extract logo text from blocks in header region
            page_height = page.dimension.height if page.dimension else 1.0
            header_threshold = 0.15  # Top 15% of page
            
            for block in page.blocks:
                text = get_text_from_text_anchor(document.text, block.layout.text_anchor)
                if text and block.layout.bounding_poly:
                    vertices = block.layout.bounding_poly.normalized_vertices
                    if vertices and len(vertices) > 0:
                        # Calculate average Y position
                        avg_y = sum(v.y for v in vertices) / len(vertices)
                        if avg_y < header_threshold:
                            logo_text.append({
                                "text": text,
                                "confidence": block.layout.confidence if hasattr(block.layout, 'confidence') else None
                            })

            # Extract form fields
            for field in page.form_fields:
                field_name = get_text_from_text_anchor(document.text, field.field_name.text_anchor).strip()
                field_value = get_text_from_text_anchor(document.text, field.field_value.text_anchor).strip()
                form_fields.append({
                    "name": field_name,
                    "value": field_value,
                    "confidence": field.field_value.confidence,
                })
    

    # -----------------------------
            # 🧠 NEW SECTION: APR detection
            # -----------------------------
            import re
            apr_pattern = re.compile(r"(\d{1,2}\.\d{1,2})\s*%")

            # Combine all page text (including tables)
            page_text = " ".join([
                get_text_from_text_anchor(document.text, b.layout.text_anchor)
                for b in page.blocks
                if b.layout and b.layout.text_anchor
            ])

            # Search in the entire text
            matches = apr_pattern.findall(page_text)
            if matches:
                apr_candidates.extend(matches)
        # -----------------------------
        # END APR DETECTION SECTION
        # -----------------------------

        # Select the most likely APR
        detected_apr = None
        if apr_candidates:
            try:
                apr_values = [float(a) for a in apr_candidates if float(a) < 20]
                if apr_values:
                    detected_apr = min(apr_values)
            except ValueError:
                pass

        # ✅ Return full response
        return ExtractResponse(
            text=document.text,
            form_fields=form_fields,
            logo_text=logo_text,
            detected_apr=detected_apr  # <-- Add this field in your ExtractResponse schema
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
