import base64
import os
import json
import requests

async def extract_logo_text_anthropic(
    file_content: bytes,
    file_name: str,
    content_type: str
) -> dict:
    """
    Extract text from logos in a document using Anthropic Claude.
    """
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return {"error": "ANTHROPIC_API_KEY not set"}
        
        # Encode file to base64
        base64_image = base64.standard_b64encode(file_content).decode("utf-8")
        
        # Determine media type - Claude accepts image/jpeg, image/png, image/gif, image/webp
        # Default to image/jpeg if not matched
        media_type = content_type
        if media_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
            media_type = "image/jpeg"
        
        # Use direct HTTP request to Anthropic API
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 1024,
            "system": """Extract text from logos in this document. 
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
                        If no logos found, return empty logos array.""",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        }
        
        # Make request
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            return {
                "error": f"Anthropic API error: {response.status_code}",
                "details": response.text
            }
        
        response_data = response.json()
        response_text = response_data["content"][0]["text"]
        
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
