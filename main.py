from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from App.services.extraction.extract_route import router as extraction_router
from App.services.extraction.document_extract_route import router as document_extract_router
from App.services.extraction.ocr_extract_route import router as ocr_extract_router
from App.services.rating.rating_route import router as rating_router
from App.services.chatbot.chatbot_routes import router as chatbot_router
from App.services.quiz.quiz_routes import router as quiz_router
from App.services.contract.multi_image_analysis_route import router as contract_router
from App.services.lease.lease_analysis_route import router as lease_router
from App.services.rate_helper.discount_schema import DiscountLineItem, DiscountTotals
from typing import List, Optional, Dict
from fastapi import UploadFile
import os
import json
import base64
import requests
import re
import copy

from App.services.contract.multi_image_analysis_schema import (
    MultiImageAnalysisResponse, Flag, NormalizedPricing, 
    APRData, TermData, TradeData, Narrative
)
from App.services.rate_helper.audit_classifier import AuditClassifier, AuditClassification

app = FastAPI(
    title="Document-AI FastAPI", 
    version="1.0.0",
)


def _fix_openapi_file_uploads(schema: dict) -> dict:
    """
    Walk the OpenAPI schema and convert 3.1.0-style file upload fields
    (contentMediaType) to 3.0.2-style (format: binary) so Swagger UI
    renders proper file-picker widgets instead of garbled text inputs.
    Also convert anyOf-null patterns to nullable for 3.0.x compat.
    """
    if isinstance(schema, dict):
        # Fix file upload: {type: string, contentMediaType: ...} -> {type: string, format: binary}
        if schema.get("type") == "string" and "contentMediaType" in schema:
            schema.pop("contentMediaType", None)
            schema["format"] = "binary"
            return schema

        # Fix anyOf with null (3.1.0 nullable) -> 3.0.x nullable
        if "anyOf" in schema:
            non_null = [s for s in schema["anyOf"] if s != {"type": "null"}]
            has_null = len(non_null) < len(schema["anyOf"])
            if has_null and len(non_null) == 1:
                merged = {k: v for k, v in schema.items() if k != "anyOf"}
                merged.update(non_null[0])
                merged["nullable"] = True
                # Recurse into the merged result
                return _fix_openapi_file_uploads(merged)

        # Recurse into all dict values
        for key, value in schema.items():
            schema[key] = _fix_openapi_file_uploads(value)

    elif isinstance(schema, list):
        schema = [_fix_openapi_file_uploads(item) for item in schema]

    return schema


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    # Force OpenAPI 3.0.2 and fix file schemas
    openapi_schema["openapi"] = "3.0.2"
    openapi_schema = _fix_openapi_file_uploads(openapi_schema)
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Include all routers (each only ONCE)
app.include_router(extraction_router)
app.include_router(document_extract_router)
app.include_router(ocr_extract_router)
app.include_router(rating_router)
app.include_router(chatbot_router)
app.include_router(quiz_router)
app.include_router(contract_router)
app.include_router(lease_router)

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


