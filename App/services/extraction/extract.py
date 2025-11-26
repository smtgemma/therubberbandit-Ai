from pathlib import Path
from typing import Union
from google.cloud import documentai
from google.oauth2 import service_account
from App.core.config import settings

credentials = service_account.Credentials.from_service_account_file(
    settings.gcp_key_path
)

client = documentai.DocumentProcessorServiceClient(credentials=credentials)

PROCESSOR_NAME = (
    f"projects/{settings.gcp_project_id}/locations/{settings.gcp_location}"
    f"/processors/{settings.gcp_processor_id}"
)



def extract_text_sync(file_path: Union[str, Path]):
    """Extract form fields and tables as fully structured objects."""

    def get_text(layout):
        """Extracts text from the document using text anchor indices."""
        if not layout or not layout.text_anchor or not layout.text_anchor.text_segments:
            return ""
        text_fragments = []
        for segment in layout.text_anchor.text_segments:
            start = segment.start_index or 0
            end = segment.end_index
            text_fragments.append(document.text[start:end])
        return "".join(text_fragments).strip()

    file_path = Path(file_path)
    mime_type = "application/pdf"

    with file_path.open("rb") as f:
        raw_doc = documentai.RawDocument(content=f.read(), mime_type=mime_type)

    request = documentai.ProcessRequest(name=PROCESSOR_NAME, raw_document=raw_doc)

    result = client.process_document(request=request)
    document = result.document

    extracted = {
        "pages": []
    }

    for page in document.pages:
        page_data = {
            "page_number": page.page_number,
            "logo_text": [],
            "form_fields": [],
            "tables": []
        }

        # Extract logo/header text from blocks
        page_height = page.dimension.height if page.dimension else 1.0
        header_threshold = 0.10  # Top 15% of page

        for block in page.blocks:
            text = get_text(block.layout)
            if text and block.layout.bounding_poly:
                # Get Y-coordinate of block
                vertices = block.layout.bounding_poly.normalized_vertices
                if vertices and len(vertices) > 0:
                    # Check if block is in header region
                    avg_y = sum(v.y for v in vertices) / len(vertices)
                    if avg_y < header_threshold:
                        page_data["logo_text"].append({
                            "text": text,
                            "confidence": block.layout.confidence if hasattr(block.layout, 'confidence') else None
                        })


        # Extract form fields
        for field in page.form_fields:
            field_name = get_text(field.field_name)
            field_value = get_text(field.field_value)
            confidence = field.field_value.confidence if field.field_value else None

            page_data["form_fields"].append({
                "field_name": {
                    "text": field_name,
                    "confidence": field.field_name.confidence if field.field_name else None
                },
                "field_value": {
                    "text": field_value,
                    "confidence": confidence
                }
            })

        # Extract tables
        for table in page.tables:
            table_obj = {
                "detected_columns": table.detected_columns,
                "header_rows": [],
                "body_rows": []
            }

            def extract_cells(row_cells):
                cells_list = []
                for cell in row_cells:
                    text = get_text(cell.layout)
                    confidence = cell.layout.confidence if hasattr(cell.layout, 'confidence') else None
                    cells_list.append({
                        "text": text,
                        "confidence": confidence,
                        "row_span": cell.row_span,
                        "col_span": cell.col_span
                    })
                return cells_list

            # Header rows
            for header_row in table.header_rows:
                row_cells = extract_cells(header_row.cells)
                table_obj["header_rows"].append(row_cells)

            # Body rows
            for body_row in table.body_rows:
                row_cells = extract_cells(body_row.cells)
                table_obj["body_rows"].append(row_cells)

            page_data["tables"].append(table_obj)

        extracted["pages"].append(page_data)

    return extracted
