import base64
import os
from openai import OpenAI
from typing import Optional
import json
import requests

# Initialize client lazily (only when needed)
client = None

def get_openai_client():
    """Get or create OpenAI client"""
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it before using the logo extraction feature."
            )
        client = OpenAI(api_key=api_key)
    return client

async def extract_logo_text_openai(
    file_content: bytes,
    file_name: str,
    content_type: str
) -> dict:
    """
    Extract text from logos in a document using OpenAI's Vision API.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OPENAI_API_KEY not set"}
        
        # Encode file to base64
        base64_image = base64.standard_b64encode(file_content).decode("utf-8")
        
        # Determine media type
        if content_type.startswith("image"):
            media_type = content_type
        elif "pdf" in content_type.lower():
            media_type = "application/pdf"
        else:
            media_type = "image/jpeg"
        
        # Use direct HTTP request to OpenAI Vision API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4.1-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {
                        "type": "text",
                        "text": """Extract text from logos in this document. 
                        IMPORTANT: Consider all text within a single logo/brand element as ONE logo.
                        If multiple words appear together as part of the same brand/logo (like "Nissan Shottenkirk Katy"), 
                        group them together as a single logo entry.
                        
                        Return the result as JSON with this structure:
                        {
                            "logos": [
                                {
                                    "logo_name": "combined brand name",
                                    "extracted_text": "all text from the single logo",
                                    "location": "where on the document"
                                }
                            ],
                            "total_logos_found": 1,
                            "confidence_score": 0.85
                        }
                        If no logos found, return empty logos array."""
                    }
                    ]
                }
            ],
            "max_tokens": 1024
        }
        
        # Make request
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            return {
                "error": f"OpenAI API error: {response.status_code}",
                "details": response.text
            }
        
        response_data = response.json()
        response_text = response_data["choices"][0]["message"]["content"]
        
        # Extract JSON from response
        try:
            extracted_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON if wrapped in markdown
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                extracted_data = {"logos": [], "total_logos_found": 0, "confidence_score": 0}
        
        return {
            "logos": extracted_data.get("logos", []),
            "total_logos_found": extracted_data.get("total_logos_found", 0),
            "confidence_score": extracted_data.get("confidence_score", 0),
            "raw_response": extracted_data
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Request error: {str(e)}"
        }
    except Exception as e:
        return {
            "error": f"Error: {str(e)}"
        }
