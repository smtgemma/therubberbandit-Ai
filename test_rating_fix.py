import asyncio
import json

from App.services.rating.rating import MultiImageAnalyzer

# Mock parsed data that previously would get clobbered by LLM flags
mock_parsed_data = {
    "buyer_name": "Test User",
    "dealer_name": "Test Dealer",
    "vin_number": "1234567890",
    "selling_price": 20000,
    "quote_type": "QUOTE",
    "red_flags": [
        {"type": "FAKE_AI_FLAG", "message": "This should be preserved but have 0 deduction", "item": "AI", "deduction": 10}
    ],
    "flags": [
        # Deterministic scoring flag from engine (that triggers overpriced doc fee)
        {"id": "DOC_FEE_ABOVE_STATE_CAP", "confidence": 1.0}
    ],
    "normalized_pricing": {
        "msrp": 20000,
        "doc_fee": 1200,  # Should trigger DOC_FEE_ABOVE_STATE_CAP
        "selling_price": 20000
    },
    "audit_status": "COMPLETE",
    "has_precomputed_flags": False
}

async def test_rating():
    analyzer = MultiImageAnalyzer()
    
    original_call = analyzer._call_json_analysis_api
    
    def fake_call(data, lang):
        return {
            "choices": [{"message": {"content": json.dumps(data)}}]
        }
    
    analyzer._call_json_analysis_api = fake_call
    analyzer._call_narrative_api = lambda *args, **kwargs: {}

    try:
        res = await analyzer.analyze_images(parsed_data=mock_parsed_data)
        
        print("Final Score:", res.score)
        
        red_flag_types = [f.type for f in res.red_flags]
        print("Red Flags present:", red_flag_types)
        
        if "DOC_FEE_ABOVE_STATE_CAP" in red_flag_types:
            print("SUCCESS: Deterministic flag DOC_FEE_ABOVE_STATE_CAP preserved.")
        else:
            print("FAILURE: Deterministic flag DOC_FEE_ABOVE_STATE_CAP missing.")
            
        print("Red flag objects:")
        for r in res.red_flags:
            print("  -", r.type, "- deduction:", r.deduction)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Test failed with exception:", str(e))

if __name__ == "__main__":
    asyncio.run(test_rating())
