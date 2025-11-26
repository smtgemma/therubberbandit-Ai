from fastapi import APIRouter, Body
from .rating_schema import DealInput
from .rating import call_groq_audit
import json
import re

router = APIRouter(prefix="/rating", tags=["Rating"])


# Regional mapping dictionaries
WEST_STATES = {'AK', 'AZ', 'CA', 'CO', 'HI', 'ID', 'MT', 'NV', 'NM', 'OR', 'UT', 'WA', 'WY'}
SOUTH_STATES = {'AL', 'AR', 'FL', 'GA', 'KY', 'LA', 'MS', 'NC', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV', 'DC'}
NORTH_STATES = {'CT', 'DE', 'IL', 'IN', 'IA', 'KS', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NE', 'NH', 'NJ', 'NY', 'ND', 'OH', 'PA', 'RI', 'SD', 'VT', 'WI'}

def categorize_region(state_code):
    """Categorize US state into regional groups"""
    if not state_code:
        return "Outside US"
    
    state_upper = state_code.upper().strip()
    
    if state_upper in WEST_STATES:
        return "West"
    elif state_upper in SOUTH_STATES:
        return "South"
    elif state_upper in NORTH_STATES:
        return "North"
    else:
        # All other US states fall into East region
        return "East"

def extract_state_from_address(address):
    """Extract state code from address string"""
    if not address:
        return None
    
    # Common US state patterns (2-letter codes)
    state_pattern = r'\b(A[KLRZ]|C[AOT]|D[CE]|FL|GA|HI|I[ADLN]|K[SY]|LA|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AIT]|W[AIVY])\b'
    
    match = re.search(state_pattern, address.upper())
    if match:
        return match.group()
    
    return None


def format_narrative(narrative_data, normalized_pricing=None):




    def get_field(key, fallback):
        value = narrative_data.get(key)
        return value if value and str(value).strip().lower() != "none" else fallback

    return {
        "vehicle_overview": get_field(
            'vehicle_overview',
            'Vehicle overview information is missing. Please include details about the make, model, year, mileage, condition, and key features of the vehicle.'
        ),
        "trust_score_summary": get_field(
            'trust_score_summary',
            'No trust score summary provided. Please include insights on fairness, transparency, and APR context. '
        ),
        "market_comparison": get_field(
            'market_comparison',
            'No market comparison found. Include pricing comparisons for GAP, VSC, and total deal structure. '
        ),

        "gap_logic": get_field(
            'gap_logic',
            'GAP pricing information is missing. However, GAP can be beneficial for buyers with high loan-to-value ratios, low down payments, or long-term loans. Assess buyer risk and discuss coverage value.'
        ),
        "vsc_logic": get_field(
            'vsc_logic',
            'VSC price data is unavailable. Still, extended warranties may be useful for buyers planning to keep the car long-term or purchasing a vehicle with uncertain reliability.'
        ),

        "apr_bonus_rule": get_field(
            'apr_bonus_rule',
            'APR data not found. Ensure APR is competitive (6.5–9.5% typical). If too high, negotiate a rate reduction or explore outside financing. '),

        "lease_audit": get_field(
            'lease_audit',
            ' For the lease audit section, if the deal is a lease, do check the details of the lease terms including residual value, money factor, lease duration, monthly payments, and implications for the buyers financial risk and benefits.'
        ),
        "negotiation_insight": get_field(
            'negotiation_insight',
            '''- **Ask to waive unnecessary fees**  
Use third-party deal comparisons to justify fair pricing.

- **Negotiate APR and monthly payments**  
Reduce long-term costs and improve affordability.

- **Request added perks**  
Push for free service, accessories, or better warranty coverage.'''
        ),
        "final_recommendation": get_field(
            'final_recommendation',
            'Final recommendation is missing. Please provide a concise summary of key findings and next steps for the buyer. Minimum 200 words.'
        )
    }


@router.post("/")
def audit_deal(input_data: DealInput = Body(...)):
    try:

        detected_apr = getattr(input_data, "detected_apr", None)
        # Call AI with just the input data
        result_json_str = call_groq_audit({
            "text": input_data.text,
            "form_fields": [f.dict() for f in input_data.form_fields],
            "logo_text": [l.dict() for l in input_data.logo_text] if input_data.logo_text else [],
            "detected_apr": detected_apr,
        })

        # Parse AI response
        result = json.loads(result_json_str)
        
        # Validate required fields exist (updated for new flag structure)
        required_fields = ['score', 'buyer_name', 'logo_text', 'dealer_name','email', 'phone_number', 'address',  'state', 'region', 'selling_price', 'vin_number', 'date',  'badge', 'buyer_message', 'red_flags', 'green_flags', 
                          'blue_flags', 'normalized_pricing', 'apr', 'term', 
                          'quote_type', 'bundle_abuse', 'narrative']
        
        for field in required_fields:
            if field not in result:
                if field in ['buyer_name', 'dealer_name','address', 'state', 'region']:
                    result[field] = None
                else:
                    raise ValueError(f"Missing required field: {field}")
                
        # If AI didn't categorize region, do it here as fallback
        if not result.get('region') and result.get('state'):
            result['region'] = categorize_region(result['state'])
        elif not result.get('region'):
            result['region'] = "Outside US"

        # Format narrative
        formatted_narrative_text = format_narrative(
            narrative_data=result.get("narrative", {}),
            normalized_pricing=result.get("normalized_pricing", {})
        )

        # Return structured response with fallbacks
        return {
            "score": result.get("score", 0),
            "buyer_name": result.get("buyer_name"),  # From AI analysis
            "dealer_name": result.get("dealer_name"),
            "logo_text": result.get("logo_text"),
 "email": result.get("email"),  # ✅ NEW FIELD
            "phone_number": result.get("phone_number"),  # ✅ NEW FIELD
            "address": result.get("address"),  # ✅ NEW FIELD
            "state": result.get("state"),  # ✅ NEW FIELD
            "region": result.get("region"),  # ✅ NEW FIELD
            "date": result.get("date"),
            "selling_price": result.get("selling_price"),      # ❌ ADD THIS
            "vin_number": result.get("vin_number"), 
            "badge": result.get("badge", "Unknown"),
            "buyer_message": result.get("buyer_message", "No message generated"),
            "red_flags": result.get("red_flags", []),
            "green_flags": result.get("green_flags", []),
            "blue_flags": result.get("blue_flags", []),
            "normalized_pricing": result.get("normalized_pricing", {}),
            "apr": result.get("apr", {}),
            "term": result.get("term", {}),
            "quote_type": result.get("quote_type", "Unknown"),
            "bundle_abuse": result.get("bundle_abuse", {}),
            "narrative": {
                "formatted": formatted_narrative_text
            }
        }

    except json.JSONDecodeError:
        return {
            "error": "❌ Failed to parse AI response as JSON",
            "raw_response": result_json_str
        }
    except ValueError as e:
        return {
            "error": f"❌ Invalid response format: {str(e)}",
            "raw_response": result_json_str
        }
    except Exception as e:
        return {
            "error": f"❌ Unexpected error: {str(e)}",
            "raw_response": result_json_str if 'result_json_str' in locals() else None
        }
