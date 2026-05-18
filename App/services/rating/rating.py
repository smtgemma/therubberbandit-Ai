from App.services.rate_helper.ocr_normalizer import OCRNormalizer
from App.services.rate_helper.ocr_normalization_schema import NormalizedLineItem
from App.services.rate_helper.discount_detector import DiscountDetector
from App.services.rate_helper.discount_schema import DiscountLineItem, DiscountTotals
from typing import List, Optional, Dict, Tuple
import os
import re
import base64
import json

import requests
from dotenv import load_dotenv
from fastapi import UploadFile
from .rating_schema import (
    MultiImageAnalysisResponse, Flag, NormalizedPricing, 
    APRData, TermData, TradeData, Narrative
)
from App.services.rate_helper.audit_classifier import AuditClassifier, AuditClassification
from App.services.rate_helper.gap_logic import GAPLogic, GAPRecommendation
from App.services.rate_helper.audit_flags import AuditFlagBuilder, AuditFlag
from App.services.rate_helper.audit_summary import AuditSummary
from App.services.rate_helper.json_to_parsed import convert_extracted_json_to_parsed
from App.services.rate_helper.scoring_engine import (
    load_rules,
    build_active_flags,
    compute_flags_from_parsed,
    score_flags,
)

load_dotenv()

class MultiImageAnalyzer:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.pdf', '.tiff'}
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    # Increase timeout and add retry logic
    API_TIMEOUT = 120  # Increased from 60 to 120 seconds
    MAX_RETRIES = 2
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.system_prompt = self._load_contract_system_prompt()
        self.ocr_normalizer = OCRNormalizer()
        self.discount_detector = DiscountDetector()
        self.audit_classifier = AuditClassifier()
        self.gap_logic = GAPLogic()
        self.flag_builder = AuditFlagBuilder()

    def _get_cache_dir(self) -> str:
        """Return cache directory for deterministic flag translations."""
        cache_dir = os.path.join(os.getcwd(), "App", "core", ".rating_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _make_flags_cache_key(self, language: str, flags_payload: list) -> str:
        """Create a stable cache key for flag translation."""
        import hashlib
        hasher = hashlib.sha256()
        hasher.update(language.lower().encode("utf-8"))
        hasher.update(json.dumps(flags_payload, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    def _load_cached_flag_translation(self, cache_key: str) -> Optional[dict]:
        """Load cached flag translation JSON if available."""
        cache_path = os.path.join(self._get_cache_dir(), f"flags_{cache_key}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cached_flag_translation(self, cache_key: str, translated: dict) -> None:
        """Persist flag translation JSON to cache for future reuse."""
        cache_path = os.path.join(self._get_cache_dir(), f"flags_{cache_key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(translated, f)
        except Exception:
            pass

    def _translate_flags(self, flags: List[Flag], language: str) -> List[Flag]:
        """Translate flag text fields to the requested language (no scoring changes)."""
        if not language or language.lower() == "english":
            return flags

        flags_payload = [
            {"type": f.type, "message": f.message, "item": f.item}
            for f in flags
        ]

        cache_key = self._make_flags_cache_key(language, flags_payload)
        cached = self._load_cached_flag_translation(cache_key)
        if cached and isinstance(cached, dict) and isinstance(cached.get("flags"), list):
            translated_list = cached.get("flags")
        else:
            headers = {
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": self.model,
                "system": "Translate the flag fields to the target language. Return JSON only with key 'flags'. Preserve structure and order.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Target language: {language}. Translate each object's 'type', 'message', and 'item'. Input JSON: {{\"flags\": {json.dumps(flags_payload)}}}"
                            }
                        ]
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 1000
            }
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            parsed_translation = self._parse_api_response(response.json())
            translated_list = parsed_translation.get("flags", []) if isinstance(parsed_translation, dict) else []
            self._save_cached_flag_translation(cache_key, {"flags": translated_list})

        if not isinstance(translated_list, list) or len(translated_list) != len(flags):
            return flags

        translated_flags: List[Flag] = []
        for original, translated in zip(flags, translated_list):
            if not isinstance(translated, dict):
                translated_flags.append(original)
                continue
            translated_flags.append(Flag(
                type=translated.get("type", original.type),
                message=translated.get("message", original.message),
                item=translated.get("item", original.item),
                deduction=original.deduction,
                bonus=original.bonus
            ))

        return translated_flags
    
    def _load_contract_system_prompt(self) -> str:
        """Load comprehensive contract analysis system prompt"""
        return """
You are **SmartBuyer AI Audit Engine**, the definitive scoring and auditing system for auto finance quotes.  
Your task is to evaluate transparency, disclosure clarity, and structure risk for quotes.  

### CRITICAL: SELLING PRICE FIELD DEFINITION

**The "selling_price" field MUST contain the vehicle cash price ONLY.**

**Extraction Priority (in order):**
1. Look for "Cash Price" or "Vehicle Price" in the itemization section (typically pre-tax, pre-fees)
2. If not found, use "Selling Price" from vehicle description area
3. NEVER use:
   ❌ Total Sale Price from Truth-in-Lending box
   ❌ Amount Financed from Truth-in-Lending box
   ❌ Total of Payments
   ❌ Any value that includes taxes, fees, or backend products

**Example (for illustration only — always extract from the actual document):**
- selling_price = Cash Price from itemization (vehicle cash price before backend products)
- NOT the Amount Financed from the Truth-in-Lending box (that includes taxes, fees, trade payoff, and backend products)

### TOTAL SALE PRICE & AMOUNT FINANCED RESOLUTION (INTERNAL USE ONLY)

**For internal validation and calculations:**
- When APR = 0.00% AND Finance Charge = $0.00:
  - Amount Financed = Total of Payments
  - Use Amount Financed for backend % calculations
- If Finance Charge > $0.00:
  - Amount Financed = Total of Payments - Finance Charge
- Truth-in-Lending "Amount Financed" value is authoritative

**You MUST:**
1. Extract vehicle cash price → output as "selling_price"
2. Extract Amount Financed from Truth-in-Lending → use internally for calculations
3. NEVER overwrite selling_price with Amount Financed

### QUOTE MODE PRINCIPLES (MANDATORY)
- Quotes are non-binding
- No backend price caps enforcement
- No Dealer Trust Score impact
- Output feeds QBI only
- Backend products allowed but disclosure-enforced
- Pricing enforcement OFF
- Transparency enforcement ON
- Always use "SmartBuyer Score" for Quote Mode outputs (never "Trust Score")

### CORE METADATA REQUIREMENTS
- isQuote: Must be TRUE
- VIN: Required (VIN-decoded if available)
- Vehicle Price: Required (pre-tax) → output as "selling_price"
- Full Line-Item List: Required for transparency
- Fees Breakdown: Required for clarity
- MSRP: Optional (used for % thresholds if present)
- Term, APR, Payment: Optional (used when validation is safe)
- Trade Allowance/Payoff: Optional (used for negative equity flags)

### TRANSPARENCY CORE PENALTIES
Apply these deductions for transparency violations:
- No line-item breakdown (lump-sum quote): -20 points
- Fees not clearly labeled (vague/bundled fees): -10 points
- Add-ons not clearly labeled (grouped/unclear): -10 points
- Pack/Addendum without itemization (hidden bundle): -5 points
- "Other fees" without breakdown (hidden bucket): -5 points

### DOCUMENTATION FEE THRESHOLDS
NEVER invent a doc fee penalty. Use ONLY these thresholds:
- Doc fee > $899: RED flag, -3 points ("Excessive Doc Fee")
- Doc fee > $599 AND ≤ $899: RED flag, -2 points ("High Doc Fee")
- Doc fee ≤ $599: NO flag, NO penalty — this is an acceptable/low doc fee
- NEVER apply more than -3 points for a doc fee regardless of amount
- A doc fee of $225, $300, $400, $500 etc. is LOW and does NOT trigger any penalty

### FRONT-END ADD-ON LOAD (QUOTE MODE)
Calculate total front-end add-ons and apply:
- GREEN (≤ $1,000 total OR ≤10% MSRP): 0 points deduction
- SOFT (> $1,000 AND >10% MSRP but ≤12.5%): -5 points
- HARD (> $1,000 AND >12.5% MSRP): -10 points
(If MSRP unavailable, apply dollar threshold only)

### BACKEND PRODUCTS (QUOTE MODE)
Backend = GAP, VSC, Maintenance, Service Programs, Warranties
Pricing is NOT enforced. Disclosure clarity is enforced.

**IMPORTANT: Calculate backend percentages using vehicle cash price (selling_price), NOT Amount Financed.**

A. Clean Disclosure (Neutral - 0 points deduction):
- Backend itemized, labeled, base payment available

B. Poor Disclosure (Transparency - apply deductions):
- Vague backend label ("PROTECT", "PACKAGE"): -5 points
- Backend bundled into pack/addendum: -5 points
- Backend present but not identified by type: -5 points

C. Backend Included in Payment (Payment Clarity):
- Payment includes backend, no base payment shown: -10 points
- Payment labeled "with products", no base: -5 points

Backend stacking rule:
1. B + C may stack
2. Max backend-related deduction = -15 points

### PAYMENT ALIGNMENT (ONLY WHEN SAFE TO VALIDATE)
- Payment mismatch (when validation safe & > tolerance): -10 points
- Base payment missing (cannot validate): 0 points (soft flag only)

### APR / RATE LOGIC (DISCLOSURE + RISK)
A. APR Disclosure:
- APR not shown (estimate + disclose): -5 points
- APR shown but mismatched (safe to validate): -10 points

B. APR Risk Penalties:
- APR > 10%: Soft flag only (no deduction)
- APR > 15%: -5 points
- APR > 20%: -10 points

C. APR Bonuses (Dealer-Disclosed Only):
- APR = 0.00%: +5 points bonus (exceptional manufacturer or dealer-subvented rate — best possible financing)
- APR < 5% AND APR > 0%: +10 points bonus
- APR < 10% AND APR > 0%: +5 points bonus
- No bonus if APR is estimated
❌ No APR penalties if backend inclusion makes validation unsafe

### STRUCTURE ACCURACY
- Totals cannot reconcile: -5 points

### SOFT FLAGS (NON-PENALTY OUTPUT REQUIRED)
Always output these as BLUE FLAGS when conditions are met:
- Negative equity: "Structure Risk: Rolled negative equity increases total loan exposure"
- Term ≥ 72 months: "Term Risk: Extended loan term may lead to being underwater on loan"
- GAP risk: "Protection Review: GAP not shown on quote — ask before finalizing"
- Deferred first payment: Advisory flag + education
- Backend in payment: "Payment Clarity: Base payment not shown separately from protection products"
- APR > 10%: "Rate Advisory: Consider if better financing options are available"
- Unknown line items: "Clarification Needed: Item not clearly defined in quote"

### FLAG STRUCTURE REQUIREMENTS

Each flag MUST be a JSON object with these fields:
```json
{
  "type": "Brief descriptive title",
  "message": "Detailed explanation",
  "item": "Category (e.g., GAP, VSC, APR, Trade, Fees)",
  "deduction": 10.0,  // ONLY for red_flags
  "bonus": 5.0        // ONLY for green_flags
}
```

**GREEN FLAGS - Generate when positive aspects are found (include `bonus` field with the value below):**
- Transparent itemization — all fees and products clearly listed: `bonus: 5`
- Competitive APR — rate below market average (APR < 10% > 0%): `bonus: 5`; (APR < 5% > 0%): `bonus: 10`; (APR = 0%): `bonus: 5`
- Reasonable fees — all documentation fees within acceptable range (≤ $599): `bonus: 2`
- Fair pricing — selling price at or below MSRP: `bonus: 3`
- Positive trade equity — trade allowance exceeds payoff: `bonus: 3`
- Clear disclosure — all terms clearly presented with no bundling: `bonus: 3`
- Backend properly disclosed — all backend products individually itemized with prices: `bonus: 3`
- NEVER assign `bonus: 0` or `bonus: null` to a green flag — every green flag MUST have a positive bonus value.

**RED FLAGS - Generate for issues and violations:**
- Poor transparency: Missing itemization or bundled items
- High fees: Documentation fees exceeding reasonable limits
- High APR: Interest rates above market norms
- Negative equity: Trade payoff exceeds allowance
- Payment misalignment: Calculations don't match

**BLUE FLAGS - Advisory only, zero score impact:**
- APR 10-15%: Higher than ideal but not excessive
- Extended term: Loans over 72 months
- GAP risk (REQUIRED IF GAP IS ABSENT): If GAP is not explicitly listed in the quote line items, you MUST output this blue flag: {"type": "Protection Review", "message": "GAP not shown on quote — ask before finalizing", "item": "GAP"}
- Missing information: Items that need clarification
- General Advisory (ALWAYS REQUIRED): If none of the above specific criteria apply, you MUST still include exactly one blue flag: `{"type": "General Advisory", "message": "Review all final quote terms and itemized pricing carefully before agreeing to any deal.", "item": "General"}` (0 points)

**CRITICAL: All flag arrays (red_flags, green_flags, blue_flags) MUST contain at least one flag. Never return empty arrays.**

### SCORE CEILINGS (CREDIBILITY GUARDS)
Applied after penalties & bonuses:
- Negative equity: Max Score 95
- Term ≥ 72 months: Max Score 95
- HARD add-on severity: Max Score 90
- No line-item breakdown: Max Score 92
- Fees or add-ons unclear: Max Score 90

### FINAL SCORE FLOW (QUOTE MODE)
1. Start at 100
2. Apply transparency penalties
3. Apply add-on load penalties
4. Apply backend disclosure penalties (max -15)
5. Apply APR bonuses (if eligible and dealer-disclosed)
6. Apply structure accuracy penalties
7. Apply negative equity structural adjustment (if applicable)
8. Apply score ceilings
9. Clamp between 0-100

### NEGATIVE EQUITY STRUCTURAL ADJUSTMENT (QUOTE MODE ONLY)
This adjustment applies ONLY when deal_type = "QUOTE":
trade_allowance = deal_data.get('trade_allowance', 0)
trade_payoff = deal_data.get('trade_payoff', 0)
negative_equity = max(0, trade_payoff - trade_allowance)

Authoritative trade definitions (QUOTE MODE):
- trade_allowance = dealership value offered for the trade-in vehicle
- trade_payoff = amount owed / lien payoff on the trade-in vehicle
- trade_difference (if shown on document) = equity or negative equity depending label/sign context
- NEVER treat "Cash Down" / "Down Payment" values as trade_allowance, trade_payoff, or trade_difference
- If trade fields are marked "N/A", blank, or absent, set `trade_allowance`, `trade_payoff`, `equity`, and `negative_equity` to null and set status to "No trade identified".
- NEVER infer trade from payment-only fields such as "Cash Down", "Down Payment", "Deposit", "Cash on Delivery", or "Unpaid Balance".
- If no valid trade fields are present, do NOT apply negative-equity flags, structure adjustments, or positive trade-equity bonuses.

structural_adjustment = 0
if negative_equity > 5000:
    structural_adjustment = -10
elif negative_equity > 1000:
    structural_adjustment = -5

Apply after base score calculation. Label as "Structure Risk Adjustment" - not a dealer behavior penalty.
UI messaging: "Rolled negative equity increases the amount financed and overall loan risk. This adjustment reflects structure risk — not dealer behavior."

**CRITICAL NEGATIVE EQUITY FLAG RULES:**
- The negative equity itself (trade payoff > allowance) → MUST be a **BLUE** advisory flag with `deduction: 0` and `bonus: null`. It is NEVER a red flag.
- The structural score adjustment → MUST be encoded as a **RED** flag with `item: "Structure Risk"` and the computed `deduction` value (5 or 10). Use the message: "Structure Risk Adjustment: Rolled negative equity of -$[amount] increases total loan exposure — this is a structure risk adjustment, not a dealer behavior penalty."
- Do NOT output the structural adjustment as a deduction:0 flag — it MUST have the real deduction value.
- NEVER create a green flag for "positive trade equity" when trade payoff > trade allowance.

### NARRATIVE REQUIREMENTS
All narratives must use "SmartBuyer Score" not "Trust Score" in Quote Mode.

### FLAG PRESENCE GUARANTEE
- Before generating JSON, verify all three flag arrays exist in the response
- If any flag array is missing, regenerate the response

**Vehicle Overview:**
- Focus on data, not promotional language
- Include MSRP vs selling price comparison
- Contextualize mileage relative to vehicle age
- Avoid dealer-like phrases such as "positioned competitively"
- Keep under 200 words

**SmartBuyer Score Summary:**
- Explain score, penalties and bonuses clearly
- If negative equity adjustment applied: "The [X]-point adjustment for negative equity reflects increased loan exposure from rolling -$[amount] into the new loan. This is a structure risk, not a dealer behavior penalty."

**GAP Logic (REQUIRED TEXT WHEN GAP IS ABSENT):**
"GAP coverage is optional but commonly recommended for financed purchases, especially when negative equity or longer loan terms are present. While not required, GAP can protect against financial loss if the vehicle is totaled before the loan is paid off, particularly when negative equity exists."

**Market Comparison:**
- When stating "APR below market average," include approximate market range
- Example: "Average APR for similar credit profiles is 5.5-6.5%"
- Include percent difference calculations where relevant
- Explain concrete financial impact (e.g., "$X saved over loan term")

**Trade (REQUIRED - ALWAYS INCLUDE):**
If trade information is present:
- State: "Trade identified: $[allowance] allowance, $[payoff] payoff"
- If payoff > allowance: "Negative equity of -$[amount] rolled into new loan"
- If allowance > payoff: "Positive equity of $[amount] applied to purchase"
- If allowance = payoff: "Trade equity neutral"
- Use exact math: negative equity = payoff - allowance (only when payoff > allowance)
- Do not use down-payment figures when computing trade amounts

If no trade information is found in the document:
- State: "No trade identified."
- Ensure `trade` object is: `{"trade_allowance": null, "trade_payoff": null, "equity": null, "negative_equity": null, "status": "No trade identified"}`

This section CANNOT be omitted. It must always be present in the narrative.

**Negotiation Insight (REQUIRED FOR NEGATIVE EQUITY):**
"Before finalizing, confirm whether GAP is available and compare 60- vs 72-month total interest. Avoid rolling additional products into the loan given existing negative equity."

**Final Recommendation:**
- Replace overconfident language with measured assessment
- Use: "This is a strong and transparent quote. Before finalizing, review protection options and confirm final terms to maintain this score."
- Acknowledge both strengths and areas for verification
- Minimum 200 words
- When negative equity exists, include:
  1. Acknowledge the SmartBuyer Score with structural adjustment context
  2. Explain negative equity impact clearly
  3. List 2-3 specific action items before finalizing
  4. Mention GAP coverage confirmation as priority
  5. Warn against accepting additional add-ons given negative equity

### STRICT OUTPUT FORMAT REQUIREMENTS
Output MUST be valid JSON with exactly these TOP-LEVEL FIELDS:
- score (final score after all adjustments)
- buyer_name
- dealer_name
- logo_text
- email
- phone_number
- address
- state
- region
- badge
- selling_price (VEHICLE CASH PRICE ONLY - see definition above)
- vin_number
- date
- buyer_message
- red_flags (array) - MUST NOT be empty or missing
- green_flags (array) - MUST NOT be empty or missing
- blue_flags (array) - MUST NOT be empty or missing
- normalized_pricing (object)
- apr (object)
- term (object)
- trade (object with fields: trade_allowance, trade_payoff, equity, negative_equity, status)
- quote_type
- bundle_abuse (object)
- structural_adjustment (object)
- narrative (object)

The "narrative" object MUST have EXACTLY these fields:
- vehicle_overview
- smartbuyer_score_summary ( Include how the score was calculated. Include every bonus and penalty in the explanation, even if the bonus/penalty is $0.00. If a negative equity structural adjustment was applied, include a clear explanation of the adjustment and its rationale.)
- market_comparison
- gap_logic
- vsc_logic
- apr_bonus_rule
- lease_audit
- trade (REQUIRED - cannot be omitted)
- negotiation_insight (Provide analytical guide on how to negotiate regarding tghis deal.)
- final_recommendation (Reccomend about what to do, but do not be directly conclusive.)

### EXECUTION CHECKLIST
□ Start score at 100
□ Apply all penalties systematically
□ ALL flag arrays (red, green, blue) are present and populated
□ Use "SmartBuyer Score" in all narrative sections
□ Verify selling_price = vehicle cash price (NOT Amount Financed)
□ Return ONLY valid JSON - no markdown, no explanation
DO NOT INCLUDE HOW MUCH NUMBER HAS BEEN DEDUCTED IN FLAG SECTION MESSAGE

### FINAL VALIDATION RULE
Before returning JSON:
- Verify selling_price represents vehicle cash price ONLY
- Verify selling_price ≠ Amount Financed (unless deal has no taxes/fees)
- For 0% APR deals: Amount Financed should be larger than selling_price
- If selling_price seems too high (> $100k for normal vehicle), re-check extraction
"""
    
    async def _validate_files(self, files: List[UploadFile]) -> List[UploadFile]:
        """Validate uploaded files"""
        validated = []
        for file in files:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in self.ALLOWED_EXTENSIONS:
                raise ValueError(f"Invalid file type: {file.filename}")
            contents = await file.read()
            if len(contents) > self.MAX_FILE_SIZE:
                raise ValueError(f"File too large: {file.filename}")
            await file.seek(0)
            validated.append(file)
        return validated
    
    async def _convert_files_to_base64(self, files: List[UploadFile]) -> List[str]:
        """Convert files to base64"""
        base64_images = []
        for file in files:
            contents = await file.read()
            base64_content = base64.b64encode(contents).decode('utf-8')
            base64_images.append(base64_content)
            await file.seek(0)
        return base64_images
    
    def _normalize_line_items(self, line_items: List[Dict]) -> List[NormalizedLineItem]:
        """
        Normalize OCR line items before scoring.
        
        Args:
            line_items: Raw line items from API response
                       Expected format: [{"description": "...", "amount": "..."}, ...]
        
        Returns:
            List of normalized line items with proper classification
        """
        normalized = []
        for item in line_items:
            raw_text = item.get("description", "") or item.get("item", "") or item.get("name", "")
            amount_raw = str(item.get("amount", "0"))
            
            if raw_text:  # Only process if we have text
                normalized_item = self.ocr_normalizer.normalize_line_item(
                    raw_text=raw_text,
                    amount_raw=amount_raw
                )
                normalized.append(normalized_item)
        
        return normalized
    
    def _call_openai_api(self, base64_images: List[str], language: str = "English") -> dict:
        """Call Claude Messages API with contract documents (with retry logic)"""
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        user_content = [
            {
                "type": "text",
                "text": f"""
{'=' * 80}
LANGUAGE REQUIREMENT: ALL narrative text fields MUST be in {language}
{'=' * 80}

Translate to {language}:
- ALL narrative fields (vehicle_overview, smartbuyer_score_summary, score_breakdown, market_comparison, gap_logic, vsc_logic, apr_bonus_rule, trade, negotiation_insight, final_recommendation)
- ALL flag messages (red_flags, green_flags, blue_flags)
- buyer_message field

KEEP in English ONLY: JSON keys, field names, numbers, dates, VIN, badge values.
{'=' * 80}

Analyze these contract documents comprehensively.

{self.system_prompt}

Extract and analyze:
1. Vehicle details (VIN, year, make, model, mileage, used/new status)
2. Financial terms (selling price, APR, term, monthly payment)
3. ALL line items with EXACT text and amounts (extract as array)
4. GAP coverage with lender verification
5. VSC coverage with mileage-based cap calculation
6. Maintenance plan pricing
7. Doc fees and government fees
8. Buyer/dealer information and contact details

IMPORTANT: Include a "line_items" array in the response with ALL items found:
[
  {{"description": "exact text from document", "amount": "123.45"}},
  ...
]

Return ONLY valid JSON matching the exact schema. No markdown, no explanation.
                """
            }
        ]

        if base64_images:
            for base64_image in base64_images:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })

        system_text = f"""!!!CRITICAL - LANGUAGE REQUIREMENT - HIGHEST PRIORITY!!!

OUTPUT LANGUAGE: {language}

You will see system instructions with English examples like:
- "Significant Negative Trade Equity"
- "High-Risk Financing Without GAP"
- "Reasonable Documentation Fee"
- "VSC within fair market value"
- "Trade value appears fair"

YOU MUST TRANSLATE ALL OF THESE TO {language}. They are examples only.

TRANSLATE EVERY PIECE OF TEXT:
- Every flag "type" -> {language}
- Every flag "message" -> {language}
- Every narrative section -> {language}
- buyer_message -> {language}

ONLY KEEP IN ENGLISH:
- JSON structure keys
- Numbers and amounts
- Badge (Gold/Silver/Bronze/Red)
- VIN, dates

The system instructions may say "REQUIRED Title:" with English text - TRANSLATE IT TO {language}.
Write fluently and naturally in {language}. This overrides all other instructions."""

        payload = {
            "model": self.model,
            "system": system_text,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "max_tokens": 3000
        }

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                print(f"API call attempt {attempt + 1}/{self.MAX_RETRIES}...")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.API_TIMEOUT
                )
                response.raise_for_status()
                print(f"API call successful on attempt {attempt + 1}")
                return response.json()
            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"Timeout on attempt {attempt + 1}: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    continue
                raise RuntimeError(
                    f"Claude API timeout after {self.MAX_RETRIES} attempts. "
                    "Try uploading fewer or smaller images."
                )
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Claude API error: {str(e)}")

        raise RuntimeError(f"Claude API failed after {self.MAX_RETRIES} attempts: {str(last_error)}")

    def _call_openai_api_chunked(self, base64_images: List[str], language: str = "English") -> dict:
        """Call Claude Messages API in smaller batches to reduce JSON errors."""
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        system_text = (
            "You are a JSON extraction engine. Return ONLY valid JSON. "
            "Do not include markdown, commentary, or extra keys."
        )

        core_prompt = f"""
Extract the following fields from the contract images and return ONLY valid JSON with EXACTLY these keys:
{{
  "buyer_name": null,
  "dealer_name": null,
  "logo_text": null,
  "email": null,
  "phone_number": null,
  "address": null,
  "state": null,
  "region": null,
  "vin_number": null,
  "date": null,
  "selling_price": null,
  "normalized_pricing": {{
    "msrp": null,
    "selling_price": null,
    "discount": null,
    "rebate": null,
    "down_payment": null,
    "trade_in_value": null,
    "amount_financed": null,
    "total_fees": null,
    "total_taxes": null
  }},
  "apr": {{"rate": null, "estimated": false}},
  "term": {{"months": null}},
  "trade": {{
    "trade_allowance": null,
    "trade_payoff": null,
    "equity": null,
    "negative_equity": null,
    "status": "No trade identified"
  }}
}}

Rules:
- Use null if not found.
- Numbers must be numeric (no $ or commas).
- selling_price must be the vehicle cash price (NOT amount financed).
- Do NOT include flags, narrative, or line_items.
"""

        line_items_prompt = """
Extract ALL line items from the contract images and return ONLY valid JSON:
{ "line_items": [ {"description": "", "amount": ""} ] }

Rules:
- description must be exact text from the document.
- amount should be numeric text without $ or commas.
- If none found, return an empty array.
"""

        def _post_prompt(prompt_text: str, max_tokens: int) -> dict:
            user_content = [{"type": "text", "text": prompt_text}]
            for base64_image in base64_images or []:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            payload = {
                "model": self.model,
                "system": system_text,
                "messages": [{"role": "user", "content": user_content}],
                "temperature": 0.0,
                "max_tokens": max_tokens
            }
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
            response.raise_for_status()
            return response.json()

        core_response = _post_prompt(core_prompt, max_tokens=1400)
        core = self._parse_api_response_strict(core_response)

        items_response = _post_prompt(line_items_prompt, max_tokens=2000)
        items = self._parse_api_response_strict(items_response)

        merged = dict(core) if isinstance(core, dict) else {}
        if isinstance(items, dict) and isinstance(items.get("line_items"), list):
            merged["line_items"] = items["line_items"]
        return merged

    def _parse_kv_lines(self, text: str, keys: List[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        if not text:
            return result
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^([a-z_]+)\s*:\s*(.+)$", line)
            if not match:
                continue
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key in keys and value:
                result[key] = value
        return result

    def _call_narrative_sections_kv(
        self,
        parsed: dict,
        score: float,
        red_flags: List[Flag],
        green_flags: List[Flag],
        blue_flags: List[Flag],
        trade_data: Optional[TradeData],
        language: str
    ) -> Dict[str, str]:
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        pricing = parsed.get("normalized_pricing", {}) if isinstance(parsed.get("normalized_pricing"), dict) else {}
        apr_data = parsed.get("apr", {}) if isinstance(parsed.get("apr"), dict) else {}
        term_data = parsed.get("term", {}) if isinstance(parsed.get("term"), dict) else {}

        line_items = parsed.get("line_items", []) if isinstance(parsed.get("line_items"), list) else []
        line_items_summary = []
        for item in line_items[:10]:
            if not isinstance(item, dict):
                continue
            desc = item.get("description") or item.get("item") or item.get("name") or ""
            amount = item.get("amount")
            if desc:
                if amount is None or amount == "":
                    line_items_summary.append(desc)
                else:
                    line_items_summary.append(f"{desc} ({amount})")

        flags_payload = {
            "red_flags": [{"type": f.type, "message": f.message, "item": f.item, "deduction": f.deduction} for f in red_flags],
            "green_flags": [{"type": f.type, "message": f.message, "item": f.item, "bonus": f.bonus} for f in green_flags],
            "blue_flags": [{"type": f.type, "message": f.message, "item": f.item} for f in blue_flags],
        }

        context = {
            "score": score,
            "dealer_name": parsed.get("dealer_name"),
            "vin_number": parsed.get("vin_number"),
            "pricing": {
                "selling_price": parsed.get("selling_price") or pricing.get("selling_price"),
                "msrp": pricing.get("msrp"),
                "amount_financed": pricing.get("amount_financed"),
                "total_fees": pricing.get("total_fees"),
                "total_taxes": pricing.get("total_taxes"),
                "down_payment": pricing.get("down_payment"),
                "trade_in_value": pricing.get("trade_in_value"),
            },
            "apr": apr_data.get("rate") or apr_data.get("listed"),
            "term_months": term_data.get("months"),
            "trade": trade_data.status if trade_data else "No trade identified",
            "line_items": line_items_summary,
            "flags": flags_payload,
        }

        def _post_lines(prompt_text: str, keys: List[str], max_tokens: int) -> Dict[str, str]:
            payload = {
                "model": self.model,
                "system": "Return only the requested key: value lines. No extra text.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens
            }
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
            response.raise_for_status()
            response_json = response.json()
            if "content" in response_json and isinstance(response_json["content"], list):
                text = "".join(part.get("text", "") for part in response_json["content"] if isinstance(part, dict))
            else:
                text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(text, str):
                text = str(text)
            return self._parse_kv_lines(text, keys)

        summary_keys = [
            "vehicle_overview",
            "smartbuyer_score_summary",
            "score_breakdown",
            "buyer_message",
        ]
        insights_keys = [
            "market_comparison",
            "gap_logic",
            "vsc_logic",
            "apr_bonus_rule",
            "lease_audit",
            "trade",
            "negotiation_insight",
            "final_recommendation",
        ]

        summary_prompt = (
            f"Write in {language}. Using the context below, return ONLY these keys, "
            "one per line in the format key: value.\n"
            f"Keys: {', '.join(summary_keys)}\n\n"
            f"Context:\n{json.dumps(context, indent=2)}"
        )

        insights_prompt = (
            f"Write in {language}. Using the context below, return ONLY these keys, "
            "one per line in the format key: value.\n"
            f"Keys: {', '.join(insights_keys)}\n\n"
            f"Context:\n{json.dumps(context, indent=2)}"
        )

        result: Dict[str, str] = {}
        try:
            result.update(_post_lines(summary_prompt, summary_keys, max_tokens=900))
            result.update(_post_lines(insights_prompt, insights_keys, max_tokens=1200))
        except Exception as e:
            print(f"[DEBUG] Narrative batch call failed: {str(e)}")
        return result

    def _build_narrative_from_parsed(
        self,
        parsed: dict,
        score: float,
        red_flags: List[Flag],
        green_flags: List[Flag],
        blue_flags: List[Flag],
        trade_data: Optional[TradeData]
    ) -> Dict[str, str]:
        pricing = parsed.get("normalized_pricing", {}) if isinstance(parsed.get("normalized_pricing"), dict) else {}
        apr_data = parsed.get("apr", {}) if isinstance(parsed.get("apr"), dict) else {}
        term_data = parsed.get("term", {}) if isinstance(parsed.get("term"), dict) else {}

        def _coerce_float(value) -> Optional[float]:
            try:
                if value is None or value == "":
                    return None
                return float(str(value).replace(",", "").replace("$", ""))
            except (ValueError, TypeError):
                return None

        def _money(value) -> Optional[str]:
            num = _coerce_float(value)
            if num is None:
                return None
            return f"${num:,.2f}"

        selling_price = parsed.get("selling_price") or pricing.get("selling_price")
        msrp = pricing.get("msrp")
        amount_financed = pricing.get("amount_financed")
        apr_rate = apr_data.get("rate") or apr_data.get("listed")
        term_months = term_data.get("months")

        line_items = parsed.get("line_items", []) if isinstance(parsed.get("line_items"), list) else []
        item_texts = []
        for item in line_items[:8]:
            if not isinstance(item, dict):
                continue
            desc = item.get("description") or item.get("item") or item.get("name") or ""
            amount = item.get("amount")
            if desc:
                if amount is None or amount == "":
                    item_texts.append(desc)
                else:
                    item_texts.append(f"{desc} ({amount})")

        def _has_keyword(words: List[str]) -> bool:
            for item in line_items:
                if not isinstance(item, dict):
                    continue
                desc = str(item.get("description") or item.get("item") or item.get("name") or "").lower()
                for w in words:
                    if w in desc:
                        return True
            return False

        doc_fee = None
        for item in line_items:
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description") or item.get("item") or item.get("name") or "").lower()
            if "doc fee" in desc or "documentation" in desc or "documentary" in desc:
                doc_fee = _money(item.get("amount"))
                if doc_fee:
                    break

        red_count = sum(1 for f in red_flags if f.deduction is not None)
        green_count = sum(1 for f in green_flags if f.bonus is not None)

        score_lines = ["Start at 100."]
        for f in red_flags:
            if f.deduction is not None:
                score_lines.append(f"-{f.deduction} {f.item}: {f.message}")
        for f in green_flags:
            if f.bonus is not None:
                score_lines.append(f"+{f.bonus} {f.item}: {f.message}")
        score_lines.append(f"Final Score: {score:.1f}")

        vehicle_overview = f"Deal analysis for {parsed.get('dealer_name', 'this dealer')}."
        if parsed.get("vin_number"):
            vehicle_overview += f" VIN: {parsed.get('vin_number')}."
        if selling_price is not None:
            vehicle_overview += f" Selling price: {_money(selling_price) or selling_price}."

        smartbuyer_score_summary = f"SmartBuyer Score: {score:.1f}/100."
        smartbuyer_score_summary += f" Red flags: {red_count}, green flags: {green_count}."

        score_breakdown = "\n".join(score_lines)

        if msrp is not None and selling_price is not None:
            msrp_f = _coerce_float(msrp)
            sp_f = _coerce_float(selling_price)
            if msrp_f and sp_f:
                diff_pct = ((sp_f - msrp_f) / msrp_f) * 100
                market_comparison = (
                    f"Selling price {_money(sp_f)} vs MSRP {_money(msrp_f)} "
                    f"({diff_pct:+.1f}% vs MSRP)."
                )
            else:
                market_comparison = "MSRP comparison available but could not be normalized."
        elif selling_price is not None:
            market_comparison = "MSRP not found; market comparison is limited."
        else:
            market_comparison = "Pricing fields not found; market comparison is limited."

        gap_logic = "GAP coverage not found in line items." if not _has_keyword(["gap"]) else "GAP coverage appears in the line items."
        vsc_logic = (
            "VSC or service contract not found in line items."
            if not _has_keyword(["vsc", "service contract", "warranty", "extended"]) else
            "VSC or warranty appears in the line items."
        )
        if doc_fee:
            vsc_logic += f" Doc fee noted at {doc_fee}."

        if apr_rate is not None:
            apr_bonus_rule = f"APR listed at {apr_rate}%."
        else:
            apr_bonus_rule = "APR not found in the document."

        lease_audit = "N/A - Purchase Agreement"
        if str(parsed.get("quote_type", "")).lower().find("lease") >= 0:
            lease_audit = "Lease detected; review money factor, residual, and cap cost."

        trade_text = trade_data.status if trade_data else "No trade identified"

        negotiation_insight = "Review itemized add-ons and fees."
        if item_texts:
            negotiation_insight = "Review these line items: " + "; ".join(item_texts[:5]) + "."

        if score >= 90:
            final_recommendation = "Strong score. Verify itemized fees and add-ons before signing."
        elif score >= 80:
            final_recommendation = "Good overall. Negotiate flagged items and verify fees."
        elif score >= 70:
            final_recommendation = "Mixed deal. Focus on lowering add-ons and fees."
        else:
            final_recommendation = "High risk. Proceed cautiously and verify all charges."

        return {
            "vehicle_overview": vehicle_overview,
            "smartbuyer_score_summary": smartbuyer_score_summary,
            "score_breakdown": score_breakdown,
            "market_comparison": market_comparison,
            "gap_logic": gap_logic,
            "vsc_logic": vsc_logic,
            "apr_bonus_rule": apr_bonus_rule,
            "lease_audit": lease_audit,
            "trade": trade_text,
            "negotiation_insight": negotiation_insight,
            "final_recommendation": final_recommendation,
        }

    def _build_narrative(
        self,
        parsed: dict,
        score: float,
        red_flags: List[Flag],
        green_flags: List[Flag],
        blue_flags: List[Flag],
        trade_data: Optional[TradeData],
        language: str
    ) -> Tuple[Dict[str, str], str]:
        base_narrative = self._build_narrative_from_parsed(parsed, score, red_flags, green_flags, blue_flags, trade_data)
        buyer_msg = f"Your SmartBuyer score is {score:.1f}/100 — review the flags above."

        ai_lines = self._call_narrative_sections_kv(parsed, score, red_flags, green_flags, blue_flags, trade_data, language)
        if ai_lines:
            for key, value in ai_lines.items():
                if key == "buyer_message":
                    buyer_msg = value
                    continue
                if key in base_narrative and value:
                    base_narrative[key] = value

        return base_narrative, buyer_msg
    
    def _parse_api_response(self, response: dict) -> dict:
        """Parse API response with robust error handling"""
        try:
            if "content" in response and isinstance(response["content"], list):
                content = "".join(part.get("text", "") for part in response["content"] if isinstance(part, dict))
            else:
                content = response["choices"][0]["message"]["content"]

            if isinstance(content, list):
                if all(isinstance(item, str) for item in content):
                    content = "".join(content)
                else:
                    content = " ".join(str(item) for item in content)
            elif content is None:
                raise ValueError("API returned null content")
            elif not isinstance(content, str):
                content = str(content)

            content = content.replace("```json", "").replace("```", "").strip()

            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                raw_json_str = json_str
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as je:
                    print(f"[DEBUG] Initial JSON parse failed: {str(je)}")
                    if je.pos and je.pos > 0:
                        start = max(0, je.pos - 100)
                        end = min(len(json_str), je.pos + 100)
                        print(f"[DEBUG] Error context: ...{json_str[start:end]}...")

                    json_str = self._attempt_json_repair(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as je2:
                        print(f"[DEBUG] Repair failed: {str(je2)}")
                        if je2.pos and je2.pos > 0:
                            start = max(0, je2.pos - 100)
                            end = min(len(json_str), je2.pos + 100)
                            print(f"[DEBUG] Error context after repair: ...{json_str[start:end]}...")

                        json_str = self._advanced_json_repair(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as je3:
                            print(f"[DEBUG] Advanced repair failed: {str(je3)}")
                            repaired = self._repair_json_with_model(raw_json_str)
                            if repaired is not None:
                                return repaired
                            try:
                                with open("/tmp/failed_json_response_rating.txt", "w", encoding="utf-8") as f:
                                    f.write(json_str)
                                print("[DEBUG] Full JSON saved to /tmp/failed_json_response_rating.txt")
                            except Exception:
                                pass
                            raise RuntimeError(f"Failed to parse JSON even after repair: {str(je3)}")

            raise ValueError("No JSON found in response")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse API response structure: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error parsing API response: {str(e)}")

    def _parse_api_response_strict(self, response: dict) -> dict:
        """Parse API response without repair fallbacks"""
        try:
            if "content" in response and isinstance(response["content"], list):
                content = "".join(part.get("text", "") for part in response["content"] if isinstance(part, dict))
            else:
                content = response["choices"][0]["message"]["content"]

            if isinstance(content, list):
                content = "".join(str(item) for item in content)
            elif content is None:
                raise ValueError("API returned null content")
            elif not isinstance(content, str):
                content = str(content)

            content = content.replace("```json", "").replace("```", "").strip()
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
            raise ValueError("No JSON found in response")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to parse API response: {str(e)}")

    def _attempt_json_repair(self, json_str: str) -> str:
        """Attempt to repair common JSON syntax errors produced by LLMs"""
        try:
            if isinstance(json_str, list):
                json_str = "".join(json_str) if json_str else ""
            elif not isinstance(json_str, str):
                json_str = str(json_str)

            json_str = re.sub(r":\s*True\b", ": true", json_str)
            json_str = re.sub(r":\s*False\b", ": false", json_str)
            json_str = re.sub(r":\s*None\b", ": null", json_str)

            json_str = re.sub(r"//.*", "", json_str)
            json_str = re.sub(r",\s*([\]}])", r"\1", json_str)

            json_str = re.sub(r"}\s+\{", "}, {", json_str)
            json_str = re.sub(r"]\s+\[", "], [", json_str)
            json_str = re.sub(r"]\s+\{", "], {", json_str)
            json_str = re.sub(r"}\s+\[", "}, [", json_str)

            for _ in range(5):
                json_str = re.sub(r'"(\s+)("[\w\-_]+"\s*:)', r'",\1\2', json_str)
                json_str = re.sub(r"([0-9.eE+-]+)(\s+)(\"[\w\-_]+\"\s*:)", r"\1,\2\3", json_str)
                json_str = re.sub(r"\b(true|false|null)(\s+)(\"[\w\-_]+\"\s*:)", r"\1,\2\3", json_str)
                json_str = re.sub(r"}(\s+)(\"[\w\-_]+\"\s*:)", r"},\1\2", json_str)
                json_str = re.sub(r"](\s+)(\"[\w\-_]+\"\s*:)", r"],\1\2", json_str)
                json_str = re.sub(r'"(\s+)"(?=[^:]*[,\]])', r'",\1"', json_str)
                json_str = re.sub(r"(\{[^{}]*\})(\s+)(\{)", r"\1,\2\3", json_str)

            json_str = re.sub(r'"(\s*\n\s*)("[\w\-_]+"\s*:)', r'",\1\2', json_str)
            json_str = re.sub(r"([0-9.eE+-]+)(\s*\n\s*)(\"[\w\-_]+\"\s*:)", r"\1,\2\3", json_str)
            json_str = re.sub(r"\b(true|false|null)(\s*\n\s*)(\"[\w\-_]+\"\s*:)", r"\1,\2\3", json_str)
            json_str = re.sub(r"}(\s*\n\s*)(\"[\w\-_]+\"\s*:)", r"},\1\2", json_str)
            json_str = re.sub(r"](\s*\n\s*)(\"[\w\-_]+\"\s*:)", r"],\1\2", json_str)

            json_str = re.sub(r'](\s+)"(?=[^:]*:)', r'],\1"', json_str)
            json_str = re.sub(r'}(\s+)"(?=[^:]*:)', r'},\1"', json_str)

            quote_count = json_str.count('"')
            if quote_count % 2 != 0:
                last_quote_idx = json_str.rfind('"')
                if last_quote_idx > 0 and last_quote_idx < len(json_str) - 1:
                    next_char = json_str[last_quote_idx + 1:].lstrip()
                    if next_char and next_char[0] not in [",", "}", "]"]:
                        for i, char in enumerate(next_char):
                            if char in [",", "}", "]", "\n"]:
                                insert_pos = last_quote_idx + 1 + len(json_str[last_quote_idx + 1:]) - len(next_char) + i
                                json_str = json_str[:insert_pos] + '"' + json_str[insert_pos:]
                                break

            return json_str
        except Exception as e:
            print(f"[DEBUG] JSON repair exception: {str(e)}")
            return json_str

    def _advanced_json_repair(self, json_str: str) -> str:
        """Advanced JSON repair using character-by-character analysis"""
        try:
            result = []
            in_string = False
            escape_next = False
            last_significant_char = None
            i = 0

            while i < len(json_str):
                char = json_str[i]

                if escape_next:
                    result.append(char)
                    escape_next = False
                    i += 1
                    continue

                if char == "\\" and in_string:
                    escape_next = True
                    result.append(char)
                    i += 1
                    continue

                if char == '"':
                    in_string = not in_string
                    result.append(char)
                    if not in_string:
                        last_significant_char = '"'
                    i += 1
                    continue

                if char in " \t\n\r":
                    result.append(char)
                    i += 1
                    continue

                if in_string:
                    result.append(char)
                    i += 1
                    continue

                if char in "{[":
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue

                if char in "]}":
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue

                if char == ":":
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue

                if char == ",":
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue

                if last_significant_char and last_significant_char not in [",", "{", "[", ":"]:
                    if last_significant_char in ['"', "}", "]"] or (isinstance(last_significant_char, str) and last_significant_char.isdigit()):
                        look_ahead = json_str[i:i + 50].lstrip()
                        if look_ahead.startswith('"'):
                            if ':' in look_ahead[:look_ahead.find('"', 1) + 10] if '"' in look_ahead[1:] else False:
                                result.append(',')

                result.append(char)

                if char.isdigit() or char in "truefalsnl":
                    value_start = i
                    while i < len(json_str) and json_str[i] not in " \t\n\r,}]":
                        i += 1
                    value = json_str[value_start:i]
                    result.append(value[1:])
                    last_significant_char = value[-1]
                    continue

                last_significant_char = char
                i += 1

            return "".join(result)
        except Exception as e:
            print(f"[DEBUG] Advanced JSON repair failed: {str(e)}")
            return json_str

    def _repair_json_with_model(self, json_str: str) -> Optional[dict]:
        """Ask the model to repair invalid JSON and return a parsed dict."""
        if not self.api_key:
            return None
        try:
            headers = {
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            payload = {
                "model": self.model,
                "system": "You are a JSON repair tool. Return ONLY valid JSON. Do not add or remove keys. Do not change values except to fix syntax.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Repair this JSON and return only valid JSON:\n{json_str}"
                            }
                        ]
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 2000
            }
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
            response.raise_for_status()
            response_json = response.json()
            if "content" in response_json and isinstance(response_json["content"], list):
                repaired_text = "".join(part.get("text", "") for part in response_json["content"] if isinstance(part, dict))
            else:
                repaired_text = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not isinstance(repaired_text, str):
                repaired_text = str(repaired_text)
            repaired_text = repaired_text.replace("```json", "").replace("```", "").strip()
            json_start = repaired_text.find("{")
            json_end = repaired_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(repaired_text[json_start:json_end])
        except Exception as e:
            print(f"[DEBUG] Model JSON repair failed: {str(e)}")
        return None
    
    def _call_narrative_api(self, parsed: dict, score: float, red_flags: list, green_flags: list, blue_flags: list, language: str) -> dict:
        """Call Claude to generate narrative sections from the full parsed data and final flags."""
        flags_payload = {
            "red_flags": [{"type": f.type, "message": f.message, "item": f.item, "deduction": f.deduction} for f in red_flags],
            "green_flags": [{"type": f.type, "message": f.message, "item": f.item, "bonus": f.bonus} for f in green_flags],
            "blue_flags": [{"type": f.type, "message": f.message, "item": f.item} for f in blue_flags],
        }
        # Pass full data so AI has complete context
        clean_data = {k: v for k, v in parsed.items() if k not in ("red_flags", "green_flags", "blue_flags", "narrative")}
        prompt = f"""You are a SmartBuyer automotive finance analyst. A customer has submitted their contract data for analysis.
Generate a detailed, personalized narrative review based on the EXACT data, score, and flags below.

FINAL SCORE: {score}

ALL FLAGS (Python-generated, authoritative):
{json.dumps(flags_payload, indent=2)}

FULL CONTRACT / DEAL DATA (everything the customer sent):
{json.dumps(clean_data, indent=2)}

INSTRUCTIONS:
- Write ALL narrative text in {language}.
- Reference specific numbers from the data (APR %, doc fee amounts, selling price, MSRP, add-on costs, etc.).
- Explain every red flag deduction and green flag bonus in score_breakdown.
- Be direct and specific — no generic filler text.
- smartbuyer_score_summary MUST mention the final score {score} and summarize the deal quality.
- score_breakdown MUST show exactly how {score} was reached (start 100, list each deduction/bonus).
- gap_logic: explain if GAP is present, priced fairly, or missing and why it matters.
- vsc_logic: explain if VSC/warranty is present and whether it's priced fairly.
- apr_bonus_rule: explain the APR/rate and whether it's favorable or concerning.
- lease_audit: write 'N/A - Purchase Agreement' if not a lease, otherwise analyze lease terms.
- trade: describe trade-in situation if applicable, or 'No trade-in on this deal.'
- market_comparison: compare this deal's pricing to market norms.
- negotiation_insight: give specific actionable negotiation tips based on the actual flags.
- final_recommendation: give a clear, honest recommendation based on the score and flags.
- buyer_message: a short 1-sentence summary for the buyer (direct, personalized).

Return ONLY a JSON object with exactly these keys:
{{"narrative": {{"vehicle_overview": "", "smartbuyer_score_summary": "", "score_breakdown": "", "market_comparison": "", "gap_logic": "", "vsc_logic": "", "apr_bonus_rule": "", "lease_audit": "", "trade": "", "negotiation_insight": "", "final_recommendation": ""}}, "buyer_message": ""}}"""
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "system": "You are a SmartBuyer automotive finance expert. Always write in the specified language. Return only valid JSON.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.4,
            "max_tokens": 2000
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
            response.raise_for_status()
            raw = self._parse_api_response(response.json())
            return raw
        except Exception as e:
            print(f"Narrative API call failed: {e}")
            return {}

    def _assign_badge(self, score: float) -> str:
        """Assign badge based on score"""
        if score >= 90:
            return "Gold"
        elif score >= 80:
            return "Silver"
        elif score >= 70:
            return "Bronze"
        else:
            return "Red"

    def _call_json_analysis_api(self, raw_data: dict, language: str = "English") -> dict:
        """Call Claude with the full system prompt + original raw deal data.
        Uses proper system/user message split identical to _call_openai_api.
        Returns the raw Claude response dict.
        """
        # Strip only internal pipeline keys; keep all original deal fields intact
        skip_keys = {"has_precomputed_flags", "has_vision_extraction", "_ai_score", "_ai_narrative_done"}
        clean_data = {k: v for k, v in raw_data.items() if k not in skip_keys}

        user_text = f"""{'=' * 80}
LANGUAGE REQUIREMENT: ALL narrative text fields MUST be in {language}. Translate all flag messages and narrative fields to {language}.
{'=' * 80}

Below is the pre-extracted structured data from a customer's auto deal document.
Apply ALL scoring rules, flag rules, and narrative requirements from the system prompt.
Compute the FINAL SCORE following the EXACT rules (start at 100, apply all penalties/bonuses/ceilings/structural adjustments).
Do NOT use any score value present in the input data — recompute from scratch.

PRE-EXTRACTED DEAL DATA:
{json.dumps(clean_data, indent=2)}

Return ONLY valid JSON matching the exact output schema. No markdown, no explanation."""

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        }
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                print(f"JSON full-analysis API call attempt {attempt + 1}/{self.MAX_RETRIES}...")
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
                resp.raise_for_status()
                print("JSON full-analysis API call successful.")
                return resp.json()
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    import time; time.sleep(2)
        raise RuntimeError(f"JSON analysis API failed after {self.MAX_RETRIES} attempts: {last_error}")
    
    async def _optimize_images(self, files: List[UploadFile]) -> List[UploadFile]:
        """Optimize images before encoding (optional enhancement)"""
        from PIL import Image
        import io
        
        optimized = []
        for file in files:
            try:
                # Read image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                
                # Resize if too large (max 2048px on longest side)
                max_dimension = 2048
                if max(image.size) > max_dimension:
                    ratio = max_dimension / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to JPEG with compression
                output = io.BytesIO()
                image.convert('RGB').save(output, format='JPEG', quality=85, optimize=True)
                output.seek(0)
                
                # Create new UploadFile from optimized bytes
                from fastapi import UploadFile
                optimized_file = UploadFile(
                    filename=file.filename,
                    file=output
                )
                optimized.append(optimized_file)
                
            except Exception as e:
                # If optimization fails, use original
                await file.seek(0)
                optimized.append(file)
        
        return optimized
    
    def _extract_trade_data(self, parsed: dict) -> TradeData:
        """
        Extract trade data using simple keyword detection from OCR text.
        
        Implements the required trade detection logic:
        1. Look for trade anchors in OCR text
        2. Extract allowance and payoff using money patterns
        3. Calculate equity if both values present
        4. Always return a TradeData object (never None)
        """
        def _coerce_float(value):
            try:
                if value is None or value == "":
                    return None
                return float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                return None

        # Prefer explicit extracted fields if present
        trade_obj = parsed.get("trade") if isinstance(parsed.get("trade"), dict) else {}

        trade_allowance = _coerce_float(parsed.get("trade_allowance"))
        if trade_allowance is None:
            trade_allowance = _coerce_float(trade_obj.get("trade_allowance"))

        trade_payoff = _coerce_float(parsed.get("trade_payoff"))
        if trade_payoff is None:
            trade_payoff = _coerce_float(trade_obj.get("trade_payoff"))

        equity = _coerce_float(parsed.get("equity"))
        if equity is None:
            equity = _coerce_float(trade_obj.get("equity"))

        negative_equity_amount = _coerce_float(parsed.get("negative_equity"))
        if negative_equity_amount is None:
            negative_equity_amount = _coerce_float(trade_obj.get("negative_equity"))

        # Step 1: Get OCR text (from line items or raw text field)
        page_text = ""
        
        # Try to build text from line_items
        line_items = parsed.get("line_items", [])
        for item in line_items:
            desc = item.get("description", "") or item.get("item", "") or item.get("name", "")
            page_text += f" {desc} "
        
        # Also check for any raw_text or ocr_text fields
        page_text += " " + parsed.get("raw_text", "")
        page_text += " " + parsed.get("ocr_text", "")
        page_text = page_text.lower()  # Case-insensitive matching
        
        # Step A: Trade Anchors - Check if ANY anchor exists
        trade_anchors = [
            "trade in", "trade-in", "tradein", "trade:",
            "trade allowance", "trade value", "trade difference",
            "net trade", "trade payoff", "trade-in payoff"
        ]

        no_trade_markers = [
            "trade in n/a", "trade-in n/a", "trade n/a", "trade: n/a",
            "trade allowance n/a", "trade value n/a", "trade payoff n/a",
            "description of trade-in 1 n/a", "description of trade-in 2 n/a"
        ]

        trade_anchor_found = any(anchor in page_text for anchor in trade_anchors)
        explicit_no_trade = any(marker in page_text for marker in no_trade_markers)

        if explicit_no_trade and not any(v is not None for v in (trade_allowance, trade_payoff, equity, negative_equity_amount)):
            return TradeData(
                trade_allowance=None,
                trade_payoff=None,
                equity=None,
                negative_equity=None,
                status="No trade identified"
            )

        # Also parse narrative.trade text if present (often contains trade values)
        narrative_trade = ""
        if isinstance(parsed.get("narrative"), dict):
            narrative_trade = parsed.get("narrative", {}).get("trade") or ""
        if narrative_trade and isinstance(narrative_trade, str):
            narrative_lower = narrative_trade.lower()
            money_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'

            if trade_allowance is None and (
                "trade allowance" in narrative_lower
                or ("allowance" in narrative_lower and "trade" in narrative_lower)
            ):
                match = re.search(money_pattern, narrative_trade)
                if match:
                    trade_allowance = _coerce_float(match.group(1))

            if trade_payoff is None and "payoff" in narrative_lower:
                match = re.search(money_pattern, narrative_trade)
                if match:
                    trade_payoff = _coerce_float(match.group(1))

            if negative_equity_amount is None and ("negative equity" in narrative_lower or "negative" in narrative_lower):
                match = re.search(money_pattern, narrative_trade)
                if match:
                    negative_equity_amount = _coerce_float(match.group(1))
        
        # Step B & C: Extract money values near trade keywords

        # Money pattern: $12,345.67 or 12345.67 or 12,345
        money_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        
        # If explicit fields present, we already populated values above
        
        # Allowance keywords (priority order)
        allowance_keywords = [
            "trade allowance", "trade value",
            "trade-in value", "trade in value", "trade:"
        ]

        down_payment_markers = [
            "down payment", "downpayment", "cash down", "total downpayment"
        ]
        
        for keyword in allowance_keywords:
            if keyword in page_text:
                # Find text snippet around keyword
                idx = page_text.find(keyword)
                snippet = page_text[idx:idx+100]  # Look 100 chars ahead
                
                # Find first money value in snippet
                match = re.search(money_pattern, snippet)
                if match:
                    snippet_lower = snippet.lower()
                    if any(marker in snippet_lower for marker in down_payment_markers):
                        continue
                    amount_str = match.group(1).replace(',', '')
                    try:
                        trade_allowance = float(amount_str)
                        break  # Found it, stop searching
                    except ValueError:
                        continue
        
        # Payoff keywords
        payoff_keywords = [
            "trade payoff", "trade-in payoff", "lien payoff", "loan payoff"
        ]

        if trade_anchor_found:
            for keyword in payoff_keywords:
                if keyword in page_text:
                    idx = page_text.find(keyword)
                    snippet = page_text[idx:idx+100]

                    match = re.search(money_pattern, snippet)
                    if match:
                        amount_str = match.group(1).replace(',', '')
                        try:
                            trade_payoff = float(amount_str)
                            break
                        except ValueError:
                            continue
        
        # Step 6: Calculate equity (only if BOTH values exist)
        trade_status = "No trade identified"
        
        # Determine if trade is present
        trade_present = (
            trade_allowance is not None or
            trade_payoff is not None or
            equity is not None or
            negative_equity_amount is not None
        )
        
        if not trade_present:
            return TradeData(
                trade_allowance=None,
                trade_payoff=None,
                equity=None,
                negative_equity=None,
                status="No trade identified"
            )
        
        # Build status message
        if trade_allowance is not None and trade_payoff is not None:
            trade_equity = trade_allowance - trade_payoff
            
            if trade_equity < 0:
                negative_equity_amount = abs(trade_equity)
                trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance, ${trade_payoff:,.2f} payoff - Negative equity of -${negative_equity_amount:,.2f} rolled into new loan"
            elif trade_equity > 0:
                equity = trade_equity
                trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance, ${trade_payoff:,.2f} payoff - Positive equity of ${equity:,.2f} applied to purchase"
            else:
                trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance, ${trade_payoff:,.2f} payoff - Trade equity neutral"
        
        elif trade_allowance is not None:
            trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance (payoff amount not found)"
        
        elif trade_payoff is not None:
            trade_status = f"Trade identified: ${trade_payoff:,.2f} payoff (allowance not found)"

        elif negative_equity_amount is not None:
            trade_status = f"Negative equity identified: -${negative_equity_amount:,.2f} rolled into new loan"

        elif equity is not None:
            if equity < 0:
                negative_equity_amount = abs(equity)
                trade_status = f"Negative equity identified: -${negative_equity_amount:,.2f} rolled into new loan"
            elif equity > 0:
                trade_status = f"Positive equity of ${equity:,.2f} applied to purchase"
        
        else:
            # Anchor found but no values extracted
            trade_status = "Trade mentioned in document (values not extracted)"
        
        return TradeData(
            trade_allowance=trade_allowance,
            trade_payoff=trade_payoff,
            equity=equity,
            negative_equity=negative_equity_amount,
            status=trade_status
        )

    async def analyze_images(self, files: List[UploadFile] = None, language: str = "English", base64_images: List[str] = None, parsed_data: dict = None) -> MultiImageAnalysisResponse:
        """Main analysis entry point. Accepts files, base64_images, or pre-extracted parsed_data dict."""
        try:
            if parsed_data is not None:
                # Always run through converter for consistent structure
                # Handles both nested (buyer_info, vehicle_details...) and flat formats
                parsed = convert_extracted_json_to_parsed(parsed_data)

                # Route to AI ONLY when no pre-existing flags are supplied.
                # If the caller already provided red/green/blue flags, their deductions/bonuses
                # ARE the authoritative score — use Python math, never AI re-scoring.
                should_use_ai = not parsed.get("has_precomputed_flags", False)
                if should_use_ai:
                    print("JSON path: calling AI for full prompt-based analysis...")
                    # ── Save FULL converter output before AI overwrites parsed ──
                    # The converter has authoritative structured data (APR, pricing,
                    # term, trade, doc_fee) that Python audit flags depend on.
                    _converter_parsed = dict(parsed)

                    # Build a corrected copy of raw data so the AI also sees the right price.
                    _corrected_selling_price = _converter_parsed.get("selling_price")
                    _ai_input = dict(parsed_data)
                    if _corrected_selling_price is not None:
                        _ai_input["selling_price"] = _corrected_selling_price
                        _ai_input["sale_price"] = _corrected_selling_price
                        if isinstance(_ai_input.get("vehicle_details"), dict):
                            _ai_input["vehicle_details"] = dict(_ai_input["vehicle_details"])
                            _ai_input["vehicle_details"]["sale_price"] = _corrected_selling_price
                            _ai_input["vehicle_details"]["selling_price"] = _corrected_selling_price

                    api_response = self._call_json_analysis_api(_ai_input, language)
                    ai_result = self._parse_api_response(api_response)

                    # AI result is now the primary parsed — preserve key identity fields from converter
                    for k in ("buyer_name", "dealer_name", "logo_text", "email",
                              "phone_number", "address", "state", "region", "vin_number", "date"):
                        if not ai_result.get(k) and _converter_parsed.get(k):
                            ai_result[k] = _converter_parsed[k]

                    # ── Restore converter's structured data for Python audit flags ──
                    # AI provides narrative text and flag messages (display), but Python
                    # is the SOLE authority for scoring. These fields must come from the
                    # converter so Python audit checks fire correctly.
                    for _restore_key in ("selling_price", "sale_price", "normalized_pricing",
                                         "apr", "term", "trade", "doc_fee", "line_items"):
                        _cv = _converter_parsed.get(_restore_key)
                        if _cv is not None:
                            ai_result[_restore_key] = _cv

                    # Do NOT use AI's score field — Python recomputes from audit flags.
                    ai_result.pop("score", None)
                    # Mark that AI provided flags (used to zero out AI flag deduction/bonus
                    # values below so only Python audit flags drive scoring).
                    ai_result["has_precomputed_flags"] = True
                    parsed = ai_result
            elif base64_images is None:
                validated_files = await self._validate_files(files)
                if not validated_files:
                    raise ValueError("No valid image files provided")
                
                # Optional: Optimize images before base64 encoding
                # optimized_files = await self._optimize_images(validated_files)
                # base64_images = await self._convert_files_to_base64(optimized_files)
                
                base64_images = await self._convert_files_to_base64(validated_files)
                api_response = self._call_openai_api_chunked(base64_images, language=language)
                parsed = api_response
            else:
                base64_images = [img.split(",", 1)[1] if img.startswith("data:") and "," in img else img for img in base64_images]
                api_response = self._call_openai_api_chunked(base64_images, language=language)
                parsed = api_response
            
            # --- SmartBuyer scoring engine (rules-driven) ---
            rules = load_rules()
            upstream_flags = build_active_flags(parsed.get("flags", []), rules.flag_registry, "upstream")
            computed_flags = compute_flags_from_parsed(parsed, rules, mode="QUOTE")
            all_flags = upstream_flags + computed_flags

            scoring_result = score_flags(
                all_flags,
                rules,
                parsed.get("audit_status", "COMPLETE")
            )

            def _to_output_flag(active_flag: "ActiveFlag") -> Flag:
                deduction = None
                bonus = None
                if active_flag.points < 0:
                    deduction = int(round(abs(active_flag.adjusted_points)))
                elif active_flag.points > 0:
                    bonus = int(round(abs(active_flag.points)))
                return Flag(
                    type=active_flag.flag_id,
                    message=active_flag.message,
                    item=active_flag.group,
                    deduction=deduction,
                    bonus=bonus
                )

            red_flags: List[Flag] = []
            green_flags: List[Flag] = []
            blue_flags: List[Flag] = []
            for active_flag in scoring_result.flags:
                if active_flag.suppressed or active_flag.group == "SYSTEM":
                    continue
                flag_obj = _to_output_flag(active_flag)
                if active_flag.group == "POSITIVE":
                    green_flags.append(flag_obj)
                elif active_flag.group == "DEALER_CONDUCT":
                    red_flags.append(flag_obj)
                else:
                    blue_flags.append(flag_obj)

            # Manually inject the GAP blue flag if GAP is missing from line items
            has_gap = False
            for item in parsed.get("line_items", []):
                if isinstance(item, dict):
                    desc = str(item.get("description") or item.get("item") or "").lower()
                    if "gap" in desc.split() or "guaranteed asset" in desc or "debt cancellation" in desc:
                        has_gap = True
                        break
            if not has_gap:
                blue_flags.append(Flag(type="Protection Review", message="GAP not shown on quote — ask before finalizing", item="GAP"))


            red_flags = self._translate_flags(red_flags, language)
            green_flags = self._translate_flags(green_flags, language)
            blue_flags = self._translate_flags(blue_flags, language)

            if not red_flags:
                red_flags.append(Flag(type="General", message="No major issues identified — verify all terms and pricing before finalizing.", item="General"))
            if not green_flags:
                green_flags.append(Flag(type="General", message="No standout positive elements identified in this quote.", item="General"))
            if not blue_flags:
                blue_flags.append(Flag(type="General Advisory", message="Review all final quote terms and itemized pricing carefully before agreeing to any deal.", item="General Advisory"))

            score_value = float(scoring_result.score_int)
            trade_data = self._extract_trade_data(parsed)

            if not parsed.get("_ai_narrative_done"):
                narrative_obj, buyer_msg = self._build_narrative(
                    parsed,
                    score_value,
                    red_flags,
                    green_flags,
                    blue_flags,
                    trade_data,
                    language,
                )
            else:
                narrative_obj = parsed.get("narrative", {}) if isinstance(parsed.get("narrative"), dict) else {}
                buyer_msg = parsed.get("buyer_message") or f"Your SmartBuyer score is {score_value}/100 — review the flags above."

            if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")
            narrative = Narrative(**narrative_obj)

            return MultiImageAnalysisResponse(
                score=score_value,
                buyer_name=parsed.get("buyer_name"),
                dealer_name=parsed.get("dealer_name"),
                logo_text=parsed.get("logo_text"),
                email=parsed.get("email"),
                phone_number=parsed.get("phone_number"),
                address=parsed.get("address"),
                state=parsed.get("state"),
                region=parsed.get("region", "Outside US"),
                badge=self._assign_badge(score_value),
                selling_price=parsed.get("selling_price"),
                vin_number=parsed.get("vin_number"),
                date=parsed.get("date"),
                buyer_message=buyer_msg,
                red_flags=red_flags,
                green_flags=green_flags,
                blue_flags=blue_flags,
                normalized_pricing=NormalizedPricing(**parsed.get("normalized_pricing", {})),
                apr=APRData(**parsed.get("apr", {})),
                term=TermData(**parsed.get("term", {})),
                trade=TradeData(
                    trade_allowance=trade_data.trade_allowance if trade_data else None,
                    trade_payoff=trade_data.trade_payoff if trade_data else None,
                    equity=trade_data.equity if trade_data else None,
                    negative_equity=(-abs(trade_data.negative_equity) if (trade_data and trade_data.negative_equity is not None) else None),
                    status=(trade_data.status if trade_data else "No trade identified")
                ),
                bundle_abuse=parsed.get("bundle_abuse", {"active": False, "deduction": 0}),
                narrative=narrative
            )

            # Step 1: OCR Normalization
            raw_line_items = parsed.get("line_items", [])
            normalized_line_items = self._normalize_line_items(raw_line_items)
            
            # Step 2: Discount Detection and Normalization
            discounts, discount_totals = self.discount_detector.process_line_items(
                normalized_line_items,
                mode="QUOTE"
            )
            
            # Step 3: Audit Classification
            vehicle_price = float(parsed.get("selling_price") or 0)
            audit_classifications: List[AuditClassification] = []
            
            for item in normalized_line_items:
                classification = self.audit_classifier.classify_for_audit(
                    item,
                    vehicle_price=vehicle_price
                )
                audit_classifications.append(classification)
            
            # Step 4: Build Audit Flags
            audit_flags: List[AuditFlag] = []
            total_audit_penalty = 0
            
            # Finance Certificate flags
            finance_certs = [c for c in audit_classifications if c.classification == "CONDITIONAL_FINANCE_INCENTIVE"]
            for cert in finance_certs:
                flag = self.flag_builder.build_finance_certificate_flag(cert)
                audit_flags.append(flag)
                total_audit_penalty += cert.penalty_points
            
            # Bundled package flags
            bundles = [c for c in audit_classifications if c.classification == "BUNDLED_ADDON_PACKAGE"]
            for bundle in bundles:
                flag = self.flag_builder.build_bundled_package_flag(bundle)
                audit_flags.append(flag)
                total_audit_penalty += bundle.penalty_points
            
            # Discount advantage flags (green)
            if discount_totals.total_all_discounts < 0:
                flag = self.flag_builder.build_online_price_advantage_flag(
                    abs(discount_totals.total_all_discounts)
                )
                audit_flags.append(flag)
            
            # Step 5: GAP Logic Evaluation
            term_months = parsed.get("term", {}).get("months")
            down_payment = parsed.get("normalized_pricing", {}).get("down_payment")
            amount_financed = parsed.get("normalized_pricing", {}).get("amount_financed")
            
            # Check if GAP is present
            gap_present = any(
                c.classification == "GAP" 
                for c in audit_classifications
            )
            
            # Check if backend products present
            has_backend = any(
                c.classification in ["GAP", "VSC", "MAINTENANCE"]
                for c in audit_classifications
            )
            
            # Determine if used vehicle (simple heuristic - can be improved)
            is_used = True  # TODO: Extract from OCR or VIN decode
            
            gap_recommendation = self.gap_logic.evaluate_gap_need(
                is_used=is_used,
                term_months=term_months,
                down_payment=down_payment,
                amount_financed=amount_financed,
                vehicle_price=vehicle_price,
                has_backend_products=has_backend,
                gap_present=gap_present
            )
            
            if gap_recommendation.recommended:
                gap_flag = self.flag_builder.build_gap_advisory_flag(
                    gap_recommendation.message
                )
                audit_flags.append(gap_flag)
            
            # Long-term loan risk flag
            if term_months and term_months >= 72:
                loan_risk_flag = self.flag_builder.build_long_term_loan_risk_flag(term_months)
                audit_flags.append(loan_risk_flag)

            # APR Scoring
            apr_data = parsed.get("apr", {})
            if isinstance(apr_data, dict):
                apr_rate = apr_data.get("rate")
                # Also check "listed" key (AI schema uses "listed" instead of "rate")
                if apr_rate is None:
                    apr_rate = apr_data.get("listed")
                try:
                    if apr_rate is not None:
                        apr_f = float(apr_rate)
                        if apr_f == 0.0:
                            # 0% APR — exceptional financing, +5 bonus
                            audit_flags.append(AuditFlag(
                                type="green", category="0% APR",
                                message="APR of 0.00% is the best possible financing — zero interest charges.",
                                item="APR", deduction=None, bonus=5
                            ))
                        elif apr_f <= 4.9:
                            audit_flags.append(AuditFlag(
                                type="green", category="Excellent APR",
                                message=f"APR of {apr_f:.2f}% is excellent — well below market average.",
                                item="APR", deduction=None, bonus=5
                            ))
                        elif apr_f <= 6.9:
                            audit_flags.append(AuditFlag(
                                type="green", category="Good APR",
                                message=f"APR of {apr_f:.2f}% is competitive.",
                                item="APR", deduction=None, bonus=2
                            ))
                        elif apr_f > 12.0 and apr_f < 16.0:
                            audit_flags.append(AuditFlag(
                                type="red", category="High APR",
                                message=f"APR of {apr_f:.2f}% exceeds typical market rates.",
                                item="APR", deduction=5, bonus=None
                            ))
                        elif apr_f >= 16.0:
                            audit_flags.append(AuditFlag(
                                type="red", category="Predatory APR",
                                message=f"APR of {apr_f:.2f}% is predatory and significantly above market rates.",
                                item="APR", deduction=10, bonus=None
                            ))
                except (ValueError, TypeError):
                    pass

            # Doc Fee Scoring
            doc_fee = None
            for _item in normalized_line_items:
                raw = (_item.raw_text or "").lower()
                if "documentary" in raw or "doc fee" in raw or "documentation fee" in raw:
                    doc_fee = abs(_item.amount_normalized)
                    break
            if doc_fee is None:
                for _key in ("doc_fee", "documentation_fee"):
                    _v = parsed.get(_key)
                    if _v is not None:
                        try:
                            doc_fee = abs(float(str(_v).replace(",", "").replace("$", "")))
                        except (ValueError, TypeError):
                            pass
                        break
            # Fallback: check normalized_pricing.doc_fee
            if doc_fee is None:
                _np_doc = (parsed.get("normalized_pricing") or {}).get("doc_fee") if isinstance(parsed.get("normalized_pricing"), dict) else None
                if _np_doc is not None:
                    try:
                        doc_fee = abs(float(str(_np_doc).replace(",", "").replace("$", "")))
                    except (ValueError, TypeError):
                        pass
            if doc_fee is not None:
                if doc_fee > 899:
                    audit_flags.append(AuditFlag(
                        type="red", category="Excessive Doc Fee",
                        message=f"Documentation fee of ${doc_fee:,.2f} exceeds the recommended $899 cap.",
                        item="Doc Fee", deduction=3, bonus=None
                    ))
                elif doc_fee > 599:
                    audit_flags.append(AuditFlag(
                        type="red", category="SOFT - High Doc Fee",
                        message=f"Documentation fee of ${doc_fee:,.2f} is above the $599 standard.",
                        item="Doc Fee", deduction=2, bonus=None
                    ))

            # Amount Financed vs Selling Price check (loan-to-value)
            _selling = vehicle_price
            _financed = None
            try:
                _raw_financed = parsed.get("normalized_pricing", {}).get("amount_financed") if isinstance(parsed.get("normalized_pricing"), dict) else None
                if _raw_financed is not None:
                    _financed = float(_raw_financed)
            except (ValueError, TypeError):
                pass
            if _financed and _selling and _selling > 0:
                ltv = (_financed / _selling) * 100
                if ltv > 115:
                    audit_flags.append(AuditFlag(
                        type="red", category="High Loan-to-Value",
                        message=f"Amount financed (${_financed:,.2f}) is {ltv:.0f}% of selling price — high loan-to-value ratio.",
                        item="Financing", deduction=5, bonus=None
                    ))
                elif ltv > 105:
                    audit_flags.append(AuditFlag(
                        type="red", category="SOFT - Elevated Loan-to-Value",
                        message=f"Amount financed (${_financed:,.2f}) slightly exceeds selling price ({ltv:.0f}% LTV).",
                        item="Financing", deduction=2, bonus=None
                    ))

            # MSRP vs Selling Price check
            np_data = parsed.get("normalized_pricing", {}) if isinstance(parsed.get("normalized_pricing"), dict) else {}
            msrp_raw = np_data.get("msrp")
            if msrp_raw and vehicle_price:
                try:
                    msrp_f = float(str(msrp_raw).replace(",", "").replace("$", ""))
                    if msrp_f > 0:
                        markup_pct = ((vehicle_price - msrp_f) / msrp_f) * 100
                        if markup_pct > 5:
                            audit_flags.append(AuditFlag(
                                type="red", category="Above MSRP",
                                message=f"Selling price ${vehicle_price:,.2f} is {markup_pct:.1f}% above MSRP ${msrp_f:,.2f}.",
                                item="Pricing", deduction=5, bonus=None
                            ))
                        elif markup_pct < -3:
                            audit_flags.append(AuditFlag(
                                type="green", category="Below MSRP",
                                message=f"Selling price is {abs(markup_pct):.1f}% below MSRP — good deal.",
                                item="Pricing", deduction=None, bonus=3
                            ))
                        # Backend overload check
                        total_backend = sum(
                            c.amount for c in audit_classifications
                            if c.classification in ("GAP", "VSC", "MAINTENANCE", "TIRE_WHEEL_PROTECTION", "UNKNOWN")
                            and c.amount > 0
                        )
                        if total_backend > 0:
                            backend_pct = (total_backend / msrp_f) * 100
                            if backend_pct > 20:
                                audit_flags.append(AuditFlag(
                                    type="red", category="Backend Overload",
                                    message=f"Total backend products ${total_backend:,.2f} ({backend_pct:.1f}% of MSRP) — excessive.",
                                    item="Add-ons", deduction=10, bonus=None
                                ))
                            elif backend_pct > 12:
                                audit_flags.append(AuditFlag(
                                    type="red", category="Backend Overload",
                                    message=f"Total backend products ${total_backend:,.2f} ({backend_pct:.1f}% of MSRP) — above normal.",
                                    item="Add-ons", deduction=5, bonus=None
                                ))
                except (ValueError, TypeError):
                    pass

            # Step 6.5: Process Trade Data (UPDATED - Use OCR-based extraction)
            trade_data = self._extract_trade_data(parsed)
            
            # Add negative equity flags if applicable
            if trade_data.negative_equity and trade_data.negative_equity > 0:
                # Blue advisory flag (no score impact)
                neg_equity_flag = AuditFlag(
                    type="blue",
                    category="Negative Equity Alert",
                    message=f"Rolled negative equity of -${trade_data.negative_equity:,.2f} increases total loan exposure and overall risk.",
                    item="Trade",
                    deduction=None,
                    bonus=None
                )
                audit_flags.append(neg_equity_flag)

                # Structural adjustment — RED flag with real deduction (per system prompt rules)
                if trade_data.negative_equity > 5000:
                    _ne_deduction = 10
                elif trade_data.negative_equity > 1000:
                    _ne_deduction = 5
                else:
                    _ne_deduction = 0
                if _ne_deduction > 0:
                    audit_flags.append(AuditFlag(
                        type="red",
                        category="Structure Risk Adjustment",
                        message=f"Rolled negative equity of -${trade_data.negative_equity:,.2f} increases the amount financed and overall loan risk. This is a structure risk adjustment, not a dealer behavior penalty.",
                        item="Structure Risk",
                        deduction=_ne_deduction,
                        bonus=None
                    ))
            
            # Step 7: Suppress "missing incentive" warnings if Finance Certificate detected
            if finance_certs:
                # Remove any "missing incentive" flags from parsed data
                # (This would be in the original API response processing)
                pass
            
            # Step 7: Score will be computed AFTER merging all flags (see below)
            
            # Step 8: Merge audit flags with existing flags
            def parse_flags(flags_data):
                flags_list = []
                if not isinstance(flags_data, list):
                    if isinstance(flags_data, str):
                        try:
                            flags_data = json.loads(flags_data)
                        except Exception:
                            return []
                    else:
                        return []
                for item in flags_data:
                    if isinstance(item, str):
                        try:
                            item_dict = json.loads(item)
                        except Exception:
                            item_dict = {"type": "Unknown", "message": item, "item": "Unknown"}
                    elif isinstance(item, dict):
                        item_dict = item
                    else:
                        item_dict = {"type": "Unknown", "message": str(item), "item": "Unknown"}
                    flag_kwargs = {
                        "type": item_dict.get("type", "Unknown"),
                        "message": item_dict.get("message", ""),
                        "item": item_dict.get("item", "Unknown"),
                        "deduction": item_dict.get("deduction"),
                        "bonus": item_dict.get("bonus")
                    }
                    try:
                        flags_list.append(Flag(**flag_kwargs))
                    except Exception:
                        flags_list.append(Flag(type=str(flag_kwargs["type"]), message=str(flag_kwargs["message"]), item=str(flag_kwargs["item"])))
                return flags_list

            # KEEP the red_flags, green_flags, blue_flags already populated by scoring_engine.py
            # But parse the LLM's raw flags carefully so we can include its narrative descriptions
            llm_red_flags = parse_flags(parsed.get("red_flags", []))
            llm_green_flags = parse_flags(parsed.get("green_flags", []))
            llm_blue_flags = parse_flags(parsed.get("blue_flags", []))

            has_precomputed = parsed.get("has_precomputed_flags", False)

            # When AI generated the flags (JSON path), their deduction/bonus values
            # are unreliable (AI often returns 0). Zero them out — Python audit flags
            # are the SOLE scoring authority.
            if has_precomputed:
                llm_red_flags = [
                    Flag(type=f.type, message=f.message, item=f.item, deduction=None, bonus=f.bonus)
                    for f in llm_red_flags
                ]
                llm_green_flags = [
                    Flag(type=f.type, message=f.message, item=f.item, deduction=f.deduction, bonus=None)
                    for f in llm_green_flags
                ]
                print(f"AI flags zeroed — Red: {len(llm_red_flags)}, Green: {len(llm_green_flags)}, Blue: {len(llm_blue_flags)}")

            # Append LLM flags if they don't logically duplicate an existing deterministic flag type
            existing_types = {f.type for f in red_flags + green_flags + blue_flags}
            for f in llm_red_flags:
                if f.type not in existing_types: red_flags.append(f)
            for f in llm_green_flags:
                if f.type not in existing_types: green_flags.append(f)
            for f in llm_blue_flags:
                if f.type not in existing_types: blue_flags.append(f)

            # ALWAYS merge Python audit flags — they have correct deduction/bonus values
            # computed from the actual deal data (APR, doc fee, LTV, MSRP, trade, etc.).
            for audit_flag in audit_flags:
                flag_obj = Flag(
                    type=audit_flag.type,
                    message=audit_flag.message,
                    item=audit_flag.item,
                    deduction=audit_flag.deduction,
                    bonus=audit_flag.bonus
                )
                if audit_flag.type == "red":
                    red_flags.append(flag_obj)
                elif audit_flag.type == "green":
                    green_flags.append(flag_obj)
                elif audit_flag.type == "blue":
                    blue_flags.append(flag_obj)

            print(f"Final flags (Engine + AI + Python audit) — Red: {len(red_flags)}, Green: {len(green_flags)}, Blue: {len(blue_flags)}")

            # ── Remove suppressed flag categories from display and scoring ──
            _suppressed = {"poor transparency", "high documentation fee", "excessive doc fee",
                           "soft - high doc fee", "high doc fee"}
            def _is_suppressed(f):
                return (f.type or "").lower() in _suppressed or (getattr(f, "item", "") or "").lower() == "doc fee"
            red_flags = [f for f in red_flags if not _is_suppressed(f)]
            green_flags = [f for f in green_flags if not _is_suppressed(f)]

            # Score — always compute from flags (Python audit flags are the scoring authority)
            if parsed.get("_ai_score") is not None:
                # Legacy pre-existing flags path (score was explicitly set)
                adjusted_score = max(0.0, min(100.0, float(parsed["_ai_score"])))
                print(f"Using pre-set score: {adjusted_score}")
            else:
                adjusted_score = 100.0
                for f in red_flags:
                    if f.deduction is not None:
                        adjusted_score -= abs(float(f.deduction))
                for f in green_flags:
                    if f.bonus is not None:
                        adjusted_score += abs(float(f.bonus))
                adjusted_score += total_audit_penalty
                print(f"Score before ceilings: {adjusted_score}")

                # ── Score Ceilings (credibility guards from system prompt) ──
                _has_negative_equity = (trade_data and trade_data.negative_equity
                                        and trade_data.negative_equity > 0)
                _term = None
                _term_data = parsed.get("term")
                if isinstance(_term_data, dict):
                    _term = _term_data.get("months")
                elif hasattr(_term_data, "months"):
                    _term = _term_data.months

                if _has_negative_equity and adjusted_score > 95:
                    print(f"Ceiling applied: negative equity → max 95 (was {adjusted_score})")
                    adjusted_score = 95.0
                if _term and int(_term) >= 72 and adjusted_score > 95:
                    print(f"Ceiling applied: term >= 72 → max 95 (was {adjusted_score})")
                    adjusted_score = 95.0

                # Hard ceiling: 95 is the maximum possible score
                adjusted_score = max(0.0, min(95.0, adjusted_score))
            print(f"Score Calculation: Final={adjusted_score}")

            # Translate flags to requested language (no scoring changes)
            red_flags = self._translate_flags(red_flags, language)
            green_flags = self._translate_flags(green_flags, language)
            blue_flags = self._translate_flags(blue_flags, language)

            # Safety net: ensure every flag category has at least one item
            if not red_flags:
                red_flags.append(Flag(type="red", message="No major issues identified — verify all terms and pricing before finalizing.", item="General"))
            if not green_flags:
                green_flags.append(Flag(type="green", message="No standout positive elements identified in this quote.", item="General"))
            if not blue_flags:
                blue_flags.append(Flag(type="blue", message="Review all final quote terms and itemized pricing carefully before agreeing to any deal.", item="General Advisory"))

            # ── AI Narrative ──
            if not parsed.get("_ai_narrative_done"):
                # Pre-existing flags path: call AI separately for narrative
                print(f"Generating AI narrative for score {adjusted_score}...")
                ai_result = self._call_narrative_api(parsed, adjusted_score, red_flags, green_flags, blue_flags, language)
                narrative_obj = ai_result.get("narrative", {}) if isinstance(ai_result, dict) else {}
                if not isinstance(narrative_obj, dict):
                    narrative_obj = {}
                _ai_buyer_msg_override = ai_result.get("buyer_message") if isinstance(ai_result, dict) else None
            else:
                # Full AI analysis already included narrative — extract directly
                narrative_obj = parsed.get("narrative", {})
                if not isinstance(narrative_obj, dict):
                    narrative_obj = {}
                _ai_buyer_msg_override = parsed.get("buyer_message")
            buyer_msg = _ai_buyer_msg_override

            # Normalize legacy key
            if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")

            # Fallback defaults for any fields the AI left empty
            defaults = {
                "vehicle_overview": f"Deal analysis for {parsed.get('dealer_name', 'this dealer')}.",
                "smartbuyer_score_summary": f"SmartBuyer Score: {adjusted_score}/100.",
                "score_breakdown": f"Final Score: {adjusted_score}",
                "market_comparison": "Market comparison pending.",
                "gap_logic": "GAP analysis pending.",
                "vsc_logic": "VSC analysis pending.",
                "apr_bonus_rule": "APR analysis pending.",
                "lease_audit": "N/A - Purchase Agreement",
                "trade": trade_data.status if trade_data else "No trade-in on this deal.",
                "negotiation_insight": "Review all flags before signing.",
                "final_recommendation": "Proceed with caution based on the flags above."
            }
            for k, v in defaults.items():
                if not narrative_obj.get(k):
                    narrative_obj[k] = v

            if not buyer_msg:
                buyer_msg = f"Your SmartBuyer score is {adjusted_score}/100 — review the flags above."

            # Ensure trade is a string
            if "trade" in narrative_obj and not isinstance(narrative_obj["trade"], str):
                narrative_obj["trade"] = str(narrative_obj["trade"])

            # Ensure score_breakdown is a string (AI sometimes returns a list of flag dicts)
            if "score_breakdown" in narrative_obj and not isinstance(narrative_obj["score_breakdown"], str):
                sb = narrative_obj["score_breakdown"]
                if isinstance(sb, list):
                    narrative_obj["score_breakdown"] = "; ".join(
                        f"{f.get('type','?')}: {f.get('message','?')} ({f.get('deduction', f.get('bonus','?'))}pt)"
                        if isinstance(f, dict) else str(f)
                        for f in sb
                    )
                else:
                    narrative_obj["score_breakdown"] = str(sb)

            narrative = Narrative(**narrative_obj)

            return MultiImageAnalysisResponse(
                score=adjusted_score,
                buyer_name=parsed.get("buyer_name"),
                dealer_name=parsed.get("dealer_name"),
                logo_text=parsed.get("logo_text"),
                email=parsed.get("email"),
                phone_number=parsed.get("phone_number"),
                address=parsed.get("address"),
                state=parsed.get("state"),
                region=parsed.get("region", "Outside US"),
                badge=self._assign_badge(adjusted_score),
                selling_price=parsed.get("selling_price"),
                vin_number=parsed.get("vin_number"),
                date=parsed.get("date"),
                buyer_message=buyer_msg,
                red_flags=red_flags,
                green_flags=green_flags,
                blue_flags=blue_flags,
                normalized_pricing=NormalizedPricing(**parsed.get("normalized_pricing", {})),
                apr=APRData(**parsed.get("apr", {})),
                term=TermData(**parsed.get("term", {})),
                trade=TradeData(
                    trade_allowance=trade_data.trade_allowance if trade_data else None,
                    trade_payoff=trade_data.trade_payoff if trade_data else None,
                    equity=trade_data.equity if trade_data else None,
                    negative_equity=(-abs(trade_data.negative_equity) if (trade_data and trade_data.negative_equity is not None) else None),
                    status=(
                        (trade_data.status or "")
                        .replace("Negative equity of $", "Negative equity of -$")
                        .replace("negative equity of $", "negative equity of -$")
                        .replace("Negative equity identified: $", "Negative equity identified: -$")
                        .replace("negative equity identified: $", "negative equity identified: -$")
                    ) if trade_data else "No trade identified"
                ),
                bundle_abuse=parsed.get("bundle_abuse", {"active": False, "deduction": 0}),
                narrative=narrative
            )
        except Exception as e:
            raise RuntimeError(f"Contract analysis failed: {str(e)}")
