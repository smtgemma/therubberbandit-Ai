from App.services.rate_helper.ocr_normalizer import OCRNormalizer
from App.services.rate_helper.ocr_normalization_schema import NormalizedLineItem
from App.services.rate_helper.discount_detector import DiscountDetector
from App.services.rate_helper.discount_schema import DiscountLineItem, DiscountTotals
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
from fastapi import UploadFile
import os
import json
import base64
import requests
import re

# FIXED IMPORT: Was improperly importing from .rating_schema
from .multi_image_analysis_schema import (
    MultiImageAnalysisResponse, Flag, NormalizedPricing, 
    APRData, TermData, TradeData, Narrative
)
from App.services.extraction.gemini_extractor import GeminiExtractor
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
    ActiveFlag,
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
        self.gemini_extractor = GeminiExtractor()
        self.ocr_normalizer = OCRNormalizer()
        self.discount_detector = DiscountDetector()
        self.audit_classifier = AuditClassifier()
        self.gap_logic = GAPLogic()
        self.flag_builder = AuditFlagBuilder()

    def _get_cache_dir(self) -> str:
        """Return cache directory for deterministic extraction reuse."""
        cache_dir = os.path.join(os.getcwd(), "App", "core", ".contract_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _make_cache_key(self, base64_images: List[str]) -> str:
        """Create a stable cache key for the given image contents."""
        import hashlib
        hasher = hashlib.sha256()
        for img in base64_images:
            hasher.update(img.encode("utf-8"))
        return hasher.hexdigest()

    def _load_cached_extraction(self, cache_key: str) -> Optional[dict]:
        """Load cached extraction JSON if available."""
        cache_path = os.path.join(self._get_cache_dir(), f"{cache_key}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cached_extraction(self, cache_key: str, parsed: dict) -> None:
        """Persist extraction JSON to cache for future reuse."""
        cache_path = os.path.join(self._get_cache_dir(), f"{cache_key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f)
        except Exception:
            pass

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
        language = self._normalize_language(language)
        if language.lower() == "english":
            return flags

        flags_payload = [
            {"type": f.type, "message": f.message, "item": f.item}
            for f in flags
        ]
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
                            "text": (
                                f"Target language: {language}. Translate each object's 'type', "
                                f"'message', and 'item'. Input JSON: {{\"flags\": {json.dumps(flags_payload)}}}"
                            )
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

    def _make_text_cache_key(self, language: str, payload: dict) -> str:
        """Create a stable cache key for text translation payloads."""
        import hashlib
        hasher = hashlib.sha256()
        hasher.update(language.lower().encode("utf-8"))
        hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    def _normalize_language(self, language: Optional[str]) -> str:
        """Normalize language input for consistent comparisons."""
        if not language:
            return "English"
        return str(language).strip()

    def _load_cached_text_translation(self, cache_key: str) -> Optional[dict]:
        """Load cached text translation JSON if available."""
        cache_path = os.path.join(self._get_cache_dir(), f"text_{cache_key}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cached_text_translation(self, cache_key: str, translated: dict) -> None:
        """Persist text translation JSON to cache for future reuse."""
        cache_path = os.path.join(self._get_cache_dir(), f"text_{cache_key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(translated, f)
        except Exception:
            pass

    def _parse_json_object(self, response: dict) -> dict:
        """Parse a JSON object from chat completion response without defaults."""
        if "content" in response and isinstance(response["content"], list):
            content = "".join(
                part.get("text", "") for part in response["content"] if isinstance(part, dict)
            )
        else:
            content = response["choices"][0]["message"]["content"]
        if isinstance(content, list):
            if all(isinstance(item, str) for item in content):
                content = "".join(content)
            else:
                content = " ".join(str(item) for item in content)
        elif content is None:
            return {}
        elif not isinstance(content, str):
            content = str(content)

        content = content.replace("```json", "").replace("```", "").strip()
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                json_str = self._attempt_json_repair(json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _translate_text_payload(self, payload: dict, language: str) -> dict:
        """Translate a JSON payload's string values to the requested language."""
        language = self._normalize_language(language)
        if language.lower() == "english":
            return payload

        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload_request = {
            "model": self.model,
            "system": "Translate all string values in the provided JSON to the target language. Return JSON only with the same keys and structure. Do not translate numbers, dates, or JSON keys.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Target language: {language}. Input JSON: {json.dumps(payload)}"
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2000
        }
        response = requests.post(self.api_url, headers=headers, json=payload_request, timeout=120)
        response.raise_for_status()
        translated = self._parse_json_object(response.json())
        if isinstance(translated, dict) and translated:
            return translated
        return payload

    def _translate_narrative_and_trade(self, narrative_data: dict, trade_status: Optional[str], buyer_message: Optional[str], language: str) -> tuple:
        """Translate narrative fields, trade status, and buyer_message when defaults are used."""
        language = self._normalize_language(language)
        if language.lower() == "english":
            return narrative_data, trade_status, buyer_message

        fields = [
            "vehicle_overview",
            "smartbuyer_score_summary",
            "score_breakdown",
            "market_comparison",
            "gap_logic",
            "vsc_logic",
            "apr_bonus_rule",
            "lease_audit",
            "negotiation_insight",
            "final_recommendation",
            "trade"
        ]

        narrative_payload = {field: str(narrative_data.get(field, "")) for field in fields}
        payload = {
            "narrative": narrative_payload,
            "trade_status": str(trade_status) if trade_status else "",
            "buyer_message": str(buyer_message) if buyer_message else ""
        }

        translated = self._translate_text_payload(payload, language)
        if isinstance(translated, dict):
            translated_narr = translated.get("narrative") if isinstance(translated.get("narrative"), dict) else {}
            for field in fields:
                translated_value = translated_narr.get(field)
                if isinstance(translated_value, str) and translated_value.strip():
                    narrative_data[field] = translated_value

            translated_trade_status = translated.get("trade_status")
            if isinstance(translated_trade_status, str) and translated_trade_status.strip():
                trade_status = translated_trade_status

            translated_buyer_message = translated.get("buyer_message")
            if isinstance(translated_buyer_message, str) and translated_buyer_message.strip():
                buyer_message = translated_buyer_message

        return narrative_data, trade_status, buyer_message
    

    def _load_contract_system_prompt(self) -> str:
        """Load comprehensive contract analysis system prompt"""
        return """
You are **SmartBuyer AI Contract Analysis Engine**. Analyze auto finance contracts comprehensively.

---

## CRITICAL MANDATORY REQUIREMENTS

### 1. ALL FLAG SECTIONS MUST BE POPULATED
- **red_flags**: MUST contain ≥1 real issue OR explicitly state no major issues found
- **green_flags**: MUST contain ≥1 positive element from actual contract analysis
- **blue_flags**: MUST contain ≥1 advisory note OR explicitly state no advisories
- **NEVER return empty arrays []** - causes validation failure
- **CONSISTENCY REQUIREMENT**: Apply the same logic across all contracts - no per-deal variation

### 2. SCORING RULES
- Do not reduce score without valid reason
- Each deal component triggers only ONE scoring outcome
- Multiple components may stack
- Disclosure failures override pricing evaluation for affected items

---

## SCORING SYSTEM

**Base Score: 100 points**

### RED FLAG DEDUCTIONS (Major Issues)
- Trade-in negative equity (disclosed) ≤$10,000: **-5**
- **Trade-in negative equity (disclosed) >$10,000: -10**
- VSC exceeds fairness threshold: **-10**
- APR >15% and ≤20%: **-10**
- APR >20%: **-15**
- Document fees exceed state limits: **-7**
- GAP insurance overpriced (exceeds cap): **-10**
- Maintenance plans overpriced (>$1,200): **-6**
- Loan term >84 months: **-8**
- **Global disclosure failure** (missing TILA disclosures OR backend products not itemized OR payment reconciliation failure OR negative equity rolled in without disclosure): **-15 (applied ONCE per audit)**
- VSC mileage cap issue (minimal remaining coverage): **-6**
- **Missing GAP with zero effective down payment AND negative equity ≤-$1,000: -10**

### BLUE FLAGS (Advisory Only - ZERO POINT IMPACT)
- APR between 10–15%: **0 points**
- Missing itemized fees: **0 points**
- No breakdown of add-on coverage terms: **0 points**
- Term longer than 72 months (but <84): **0 points**
- Term vs coverage mismatch: **0 points**

### GREEN FLAG BONUSES (Positive Elements)
- VSC within fairness threshold: **+3**
- Competitive APR (<5%): **+5**
- Transparent itemization: **+3**
- Positive trade equity: **+5**
- No unnecessary add-ons: **+3**
- GAP coverage present and fairly priced: **+5**

**MAXIMUM SCORE: 100 | MINIMUM SCORE: 0**

**Score Calculation Example:**
```
Starting Score: 100
  Negative equity >$10,000 (disclosed): -10
  Missing GAP with $0 down + negative equity: -10
  Transparent itemization: +3
  Competitive APR: +5
= Final Score: 88
```

**MUST include detailed score_breakdown in narrative showing ALL deductions/bonuses with reasoning (exclude Blue flags - 0 point impact)**

---

## GAP COVERAGE — AUTHORITATIVE LOGIC

### 1) GAP Recognition - CRITICAL OCR Dictionary
**GAP coverage can appear under ANY of these names:**
- "GAP"
- "Debt Cancellation Agreement"
- "DCA"
- "Guaranteed Asset Protection"
- "Loan/Lease Payoff Protection"
- "Debt Protection"

**CRITICAL RULE**: If ANY of these terms appear with a dollar amount, classify as **GAP = PRESENT** and evaluate pricing.

**PROCESSING LOGIC:**
```
# Step 1: Check for GAP presence
if ANY_GAP_SYNONYM_found_in_contract:
    gap_present = TRUE
    gap_price = extract_dollar_amount_from_gap_line_item
else:
    gap_present = FALSE

# Step 2: Calculate GAP cap
gap_cap = min($1,200, 3% of MSRP, $1,500 if MSRP >= $60,000)

# Step 3: Determine flag (ONLY ONE outcome)
if gap_present == TRUE:
    if gap_price <= gap_cap:
        APPLY_GREEN_FLAG("+5 points", "GAP fairly priced")
        DO_NOT_APPLY_RED_FLAG
        DO_NOT_APPLY_BLUE_FLAG
    else:
        APPLY_RED_FLAG("-10 points", "GAP overpriced")
        DO_NOT_APPLY_GREEN_FLAG
        DO_NOT_APPLY_BLUE_FLAG
else:
    # GAP not present - check high-risk scenario
    effective_down = cash_down + max(trade_equity, 0)
    if effective_down == 0 AND trade_equity <= -$1,000:
        APPLY_RED_FLAG("-10 points", "High-Risk Financing Without GAP")
    else:
        NO_FLAG_NEEDED
```

### 2) GAP Pricing Caps (Clarified Formula)
GAP price must be ≤ LOWEST of:
- **$1,200 (standard cap)**
- **3% of MSRP**
- **$1,500 ONLY if MSRP ≥$60,000**

**Cap Calculation Examples:**
- MSRP $40,000: 3% = $1,200 → Cap = min($1,200, $1,200) = **$1,200**
- MSRP $69,625: 3% = $2,088.75 → Cap = min($1,200, $2,088.75, $1,500) = **$1,200**
- MSRP $80,000: 3% = $2,400 → Cap = min($1,200, $2,400, $1,500) = **$1,200**

### 3) Effective Down Payment Definition
`Effective_Down = Cash_Down + max(Trade_Equity, 0)`
*(Negative trade equity does NOT count toward down payment)*

### 4) GAP Flags & Scoring — ONE OUTCOME ONLY

**🟢 GREEN FLAG (Fair GAP) — ONLY IF GAP PRESENT AND WITHIN CAP**
- **Trigger**: GAP present **AND** GAP price ≤ cap
- **Score**: +5 points
- **Language**: Positive/neutral only
- **Example**: "Debt Cancellation Agreement included at $1,032 - within $1,200 pricing cap for this vehicle (MSRP: $69,625)."
- **CRITICAL**: If this flag applies, DO NOT apply ANY red flag or blue flag for GAP

**🔴 RED FLAG (Overpriced GAP) — ONLY IF GAP PRESENT AND EXCEEDS CAP**
- **Trigger**: GAP present **AND** GAP price > cap
- **Score**: -10 points
- **Language**: Overpricing/reduce or remove allowed
- **Example**: "GAP insurance overpriced at $1,450 - exceeds $1,200 fair market cap"
- **CRITICAL**: Verify gap_price ACTUALLY exceeds cap before applying (e.g., $1,450 > $1,200 = TRUE)
- **NEVER apply if**: gap_price <= cap (e.g., $1,032 <= $1,200 means NO red flag, GREEN flag instead)

**🔴 RED FLAG (High-Risk Financing Without GAP) — ONLY IF ALL CONDITIONS TRUE**
- **Trigger**: ALL must be true:
  * GAP NOT present (no GAP synonyms found, including DCA)
  * Effective_Down = $0
  * Trade_Equity ≤ -$1,000
- **Score**: -10 points
- **Language**: Protection advisory focused on risk (NOT dealer fault)
- **Required Title**: "High-Risk Financing Without GAP"
- **Required Message Format**: "No GAP coverage included with zero effective down payment and more than $1,000 in negative equity, creating elevated total-loss risk."
- **CRITICAL**: DO NOT apply this flag if GAP is present (any synonym including DCA)

### 5) GAP Logic Rules — NEVER DO THESE:

❌ Apply BOTH green flag AND red flag for GAP (only ONE outcome)
❌ Apply green flag AND blue flag for GAP (only ONE outcome)
❌ State "GAP missing" when Debt Cancellation Agreement exists
❌ Flag GAP as "overpriced" or "unusually high" when price ≤ cap (mathematical impossibility)
❌ Use mathematical errors like "$1,032 exceeds $1,200"
❌ Apply "High-Risk Without GAP" red flag when GAP/DCA is present
❌ Use loan term to trigger GAP flags
❌ Apply multiple penalties to GAP
❌ Flag missing GAP without BOTH $0 effective down AND ≥$1,000 negative equity
❌ Use "average cost" language
❌ Convert GREEN flag into BLUE or RED flag
❌ Use phrases like "GAP missing flag not triggered because..."
❌ Use phrases like "Recommended but not mandatory" inside red flags
❌ Use advisory language like "unusually high" for GAP within cap

### 6) GAP Scenario Summary
| Scenario | Flag | Score | Example |
|----------|------|-------|---------|
| GAP present, price ≤ cap | 🟢 GREEN ONLY | +5 | DCA at $1,032, cap $1,200 |
| GAP present, price > cap | 🔴 RED ONLY | -10 | GAP at $1,450, cap $1,200 |
| GAP missing + $0 effective down + ≤-$1,000 equity | 🔴 RED | -10 | No GAP/DCA found, meets risk criteria |
| GAP missing (otherwise) | — | 0 | No flag needed |

**VALIDATION CHECK BEFORE FINALIZING:**
```
# Pre-check: Scan contract for ALL GAP synonyms
gap_synonyms = ["GAP", "Debt Cancellation Agreement", "DCA", 
                "Guaranteed Asset Protection", "Loan/Lease Payoff Protection", 
                "Debt Protection"]

if any_synonym_found_with_price:
    gap_present = TRUE
    gap_price = extracted_amount
    gap_cap = min($1,200, 3% of MSRP, $1,500 if MSRP >= $60,000)
    
    # Mathematical verification
    if gap_price <= gap_cap:
        MUST_HAVE: green_flag for GAP (+5)
        MUST_NOT_HAVE: red_flag for GAP
        MUST_NOT_HAVE: blue_flag for GAP
        MUST_NOT_HAVE: "High-Risk Without GAP" red flag
    else:  # gap_price > gap_cap
        MUST_HAVE: red_flag "GAP overpriced" (-10)
        MUST_NOT_HAVE: green_flag for GAP
        MUST_NOT_HAVE: blue_flag for GAP
        MUST_NOT_HAVE: "High-Risk Without GAP" red flag
else:
    gap_present = FALSE
    # Check high-risk scenario
    if effective_down == 0 AND trade_equity <= -$1,000:
        MUST_HAVE: red_flag "High-Risk Financing Without GAP" (-10)
    else:
        NO_GAP_FLAG_NEEDED
```

---

## DOCUMENT FEES - STATE-SPECIFIC RULES

### State-by-State Documentation Fee Limits

**States WITH Hard Caps (Flag if exceeded):**
- California: $85
- Florida: $995
- Illinois: $300 (some exceptions)
- New York: $175
- Oregon: $150
- Washington: $150

**States WITHOUT Hard Caps (Flag only if clearly abusive):**
- **Texas**: NO STATUTORY LIMIT
  - Acceptable range: **$150-$250**
  - Common range: **$175-$225**
  - Flag only if: **>$300** (clearly excessive)
- Georgia: No cap (flag if >$500)
- Arizona: No cap (flag if >$500)
- Nevada: No cap (flag if >$500)
- Colorado: No cap (flag if >$500)

### Documentation Fee Evaluation Logic

**FOR TEXAS CONTRACTS:**
```
if state == "TX":
    if doc_fee <= 150:
        APPLY_GREEN_FLAG("+3 points", "below market")
    elif doc_fee > 150 AND doc_fee <= 250:
        NO_FLAG_NEEDED (neutral, 0 points)
        OPTIONALLY: APPLY_GREEN_FLAG("+3 points", "within typical range")
        NEVER: APPLY_RED_FLAG
        NEVER: APPLY_BLUE_FLAG
    elif doc_fee > 250 AND doc_fee <= 300:
        APPLY_BLUE_FLAG("0 points", "advisory only")
    else:  # doc_fee > 300
        APPLY_RED_FLAG("-7 points", "exceeds reasonable range")
```

**FOR OTHER NO-CAP STATES:**
- ≤$200: **GREEN FLAG** (+3)
- $200-$300: **NEUTRAL** (0 points)
- $300-$500: **BLUE FLAG** (0 points)
- >$500: **RED FLAG** (-7)

**FOR CAPPED STATES:**
- Below cap: **GREEN FLAG** (+3) or neutral
- At/near cap: **NEUTRAL** (0 points)
- Above cap: **RED FLAG** (-7)

### Documentation Fee Flag Examples

**CORRECT Texas Example (No Red Flag):**
```json
// Doc fee = $225 in Texas
// NO red flag, NO blue flag
// Optionally include in green_flags:
{
  "type": "Reasonable Documentation Fee",
  "message": "Documentation fee of $225 falls within typical market range for Texas dealerships ($150-$250).",
  "item": "Documentation Fee",
  "bonus": 3.0
}
```

**INCORRECT Texas Example (DO NOT DO THIS):**
```json
// ❌ WRONG - Do not flag $225 in Texas as excessive
{
  "type": "Documentation fees exceed limits",
  "message": "Documentation fee likely above Texas limits",
  "item": "Documentation Fee",
  "deduction": 7.0
}
```

**CRITICAL RULE**: For Texas, doc fees $150-$250 are NEVER red flags and NEVER blue flags. Do not apply contradictory flags (both red and green). Do not use language like "likely above Texas limits" (Texas has no limit).

---

## SELLING PRICE FIELD DEFINITION

**"selling_price" MUST contain vehicle cash price ONLY**

### Extraction Priority (in order):
1. "Cash Price" or "Vehicle Price" in itemization section (pre-tax, pre-fees)
2. If not found, "Selling Price" from vehicle description area
3. **NEVER use:**
   - ❌ Total Sale Price from Truth-in-Lending
   - ❌ Amount Financed from Truth-in-Lending
   - ❌ Total of Payments
   - ❌ Any value including taxes/fees/backend products

**Example:**
- ✅ selling_price = $52,068.78 (vehicle cash price)
- ❌ NOT $67,158.60 (Amount Financed with taxes/fees/products)

---

## SERVICE CONTRACT (VSC) ANALYSIS RULES

### VSC Pricing Threshold Calculation (CRITICAL FORMULA)
**SmartBuyer VSC Cap = min(15% of MSRP, $6,000)**

**Step-by-Step Calculation:**
1. Calculate 15% of MSRP
2. Compare to hard cap of $6,000
3. Use the LOWER value as the threshold

**Examples:**
- MSRP $30,000: 15% = $4,500 → Cap = min($4,500, $6,000) = **$4,500**
- MSRP $40,000: 15% = $6,000 → Cap = min($6,000, $6,000) = **$6,000**
- MSRP $69,625: 15% = $10,443.75 → Cap = min($10,443.75, $6,000) = **$6,000**
- MSRP $80,000: 15% = $12,000 → Cap = min($12,000, $6,000) = **$6,000**

**For vehicles with MSRP ≥$40,000, the cap is effectively $6,000**

### Pricing Assessment (ONE OUTCOME PER VSC)
- VSC price **≤** SmartBuyer cap: **GREEN FLAG (+3)**
- VSC price **>** SmartBuyer cap: **RED FLAG (-10)**
- VSC mileage cap issue (minimal remaining coverage): **RED FLAG (-6)**
- **Critical Rule**: Only ONE penalty or bonus per VSC. Never stack pricing + mileage penalties.

### VSC Flag Logic - CRITICAL VALIDATION

**BEFORE APPLYING ANY VSC FLAG:**
```
# Step 1: Extract MSRP
msrp = extract_msrp_from_contract

# Step 2: Calculate VSC cap
vsc_15_percent = msrp * 0.15
vsc_cap = min(vsc_15_percent, $6,000)

# Step 3: Mathematical verification
if vsc_price <= vsc_cap:
    APPLY_GREEN_FLAG("+3 points", "VSC within fair market value")
    DO_NOT_APPLY_RED_FLAG
else:  # vsc_price > vsc_cap
    APPLY_RED_FLAG("-10 points", "VSC exceeds fair market value")
    DO_NOT_APPLY_GREEN_FLAG

# Example verification:
# MSRP = $69,625
# 15% = $10,443.75
# Cap = min($10,443.75, $6,000) = $6,000
# VSC price = $3,500
# Is $3,500 <= $6,000? YES → GREEN FLAG
```

**VSC Logic Rules — NEVER DO THESE:**

❌ Flag VSC as "exceeds threshold" when price ≤ cap
❌ Use incorrect cap calculation (forgetting min($6,000) constraint)
❌ Apply red flag for VSC price under $6,000 when MSRP ≥$40,000
❌ Apply both green and red flags for same VSC
❌ Use percentage-based language ("X% of vehicle value")
❌ Stack pricing + mileage penalties

### Context-Based Analysis (Advisory Only - NO POINT DEDUCTIONS)
- Mileage restrictions: advisory only
- Term vs coverage mismatch: advisory only
- New vs used vehicle context: advisory only
- **All percentage-based logic REMOVED** (no VSC as % of vehicle value)

### Flag Logic (One-Outcome Rule)
- Price ≤ threshold + adequate coverage = **GREEN FLAG (+3)**
- Price ≤ threshold BUT mileage cap issues = **RED FLAG (-6)** [mileage takes precedence]
- Price > threshold = **RED FLAG (-10)** [pricing takes precedence]
- VSC not itemized = **GLOBAL -15 disclosure penalty** (pricing evaluation suppressed)

---

## FLAG DEFINITIONS

### 🟢 GREEN FLAGS
✅ Fair, transparent, consumer-friendly terms

- **Reasonable documentation fee**: Within acceptable range for state (+3)
- **VSC within fairness cap**: Price ≤ SmartBuyer cap (+3)
- **GAP coverage fairly priced**: GAP present and within cap (+5)
- **Competitive APR (<5%)**: Well below market average (+5)
- **Transparent itemization**: All fees/add-ons clearly listed (+3)
- **Positive trade equity**: Trade value reduces financed amount (+5)
- **No unnecessary add-ons**: Only relevant products included (+3)

#### For VSC Within Cap:
- **REQUIRED Title**: "VSC within fair market value" or "Extended Warranty Fairly Priced"
- **REQUIRED Message Format**: "Extended warranty priced at $[amount] is within the $[cap] fair market threshold for this vehicle (MSRP: $[msrp])."
- **Score Impact**: +3 points
- **Example**: "Extended warranty priced at $3,500 is within the $6,000 fair market threshold for this vehicle (MSRP: $69,625)."
- **CRITICAL**: Only apply when vsc_price <= vsc_cap (verify math first)

#### For GAP/DCA Present and Fair:
- **REQUIRED Title**: "GAP Coverage Fairly Priced" or "Debt Cancellation Agreement Within Cap"
- **REQUIRED Message Format**: "[GAP/Debt Cancellation Agreement] included at $[amount] - within $[cap] pricing cap for this vehicle (MSRP: $[msrp])."
- **Score Impact**: +5 points
- **Example**: "Debt Cancellation Agreement included at $1,032 - within $1,200 pricing cap for this vehicle (MSRP: $69,625)."
- **CRITICAL**: Only apply if gap_price <= gap_cap (verify math: $1,032 <= $1,200 = TRUE)

#### For Texas Documentation Fees $150-$250:
- **REQUIRED**: NO red flag, NO blue flag, optionally green flag or neutral
- **If Green Flag Used**: "Reasonable Documentation Fee"
- **Message Format**: "Documentation fee of $[amount] falls within typical market range for Texas dealerships ($150-$250)."
- **Score Impact**: +3 points OR 0 points (neutral)
- **NEVER flag as excessive in this range**

### 🔵 BLUE FLAGS (ZERO SCORE IMPACT)
⚠️ Informational only - does NOT affect score

- **APR 10–15%**: Higher cost—consider better rates (0)
- **Missing itemized fees**: Total shown but not broken down (0)
- **No add-on coverage breakdown**: Product listed but details missing (0)
- **Term >72 months**: Loan >6 years increases interest (0)
- **Term vs coverage mismatch**: Finance term exceeds coverage (0)
- **General Advisory (ALWAYS REQUIRED)**: If none of the above specific criteria apply, you MUST still include exactly one blue flag: `{"type": "General Advisory", "message": "Review all final contract terms, product details, and payment figures carefully before signing to confirm accuracy.", "item": "General"}` (0 points)

**CRITICAL**: Blue flags have ZERO point impact. Do not use blue flags for items that should be green (e.g., GAP within cap, VSC within cap, doc fees in acceptable range).
**MANDATORY**: `blue_flags` array MUST NEVER be empty — always include at least the General Advisory if no other criteria apply.

### 🔴 RED FLAGS
❌ High-risk, overpriced, or non-compliant terms

- **Significant Negative Trade Equity**: >$10,000 rolled into loan (-10)
- **Trade-in negative equity ≤$10,000 (disclosed)**: Amount rolled into new loan (-5)
- **VSC exceeds fairness cap**: Above SmartBuyer threshold (-10)
- **APR >15%**: High-cost financing/subprime risk (-10)
- **APR >20%**: Extremely high-cost financing (-15)
- **Document fees exceed state limits**: Violates max allowable fee (-7)
- **GAP insurance overpriced**: Exceeds pricing cap (-10) - **ONLY if gap_price > gap_cap**
- **Maintenance plans overpriced (>$1,200)**: Above market value (-6)
- **Loan term >84 months**: 7+ years—CFPB warns against (-8)
- **Global disclosure failure**: Missing TILA/itemization/payment reconciliation (-15, once per audit)
- **VSC mileage cap issue**: Minimal remaining coverage (-6)
- **High-Risk Financing Without GAP**: No GAP + $0 effective down + >$1,000 negative equity (-10) - **ONLY if GAP not present**

### Flag Message Format Requirements - CRITICAL UPDATES

#### For Negative Equity >$10,000:
- **REQUIRED Title**: "Significant Negative Trade Equity"
- **REQUIRED Message Format**: "Trade-in negative equity of -$[exact_amount] exceeds $10,000 and was rolled into the new loan."
- **Score Impact**: -10 (NOT -5)
- **DO NOT use generic phrases** like "negative equity present (disclosed)"
- **DO NOT combine** with GAP flag into one message

#### For Missing GAP with High Risk:
- **REQUIRED Title**: "High-Risk Financing Without GAP"
- **REQUIRED Message Format**: "No GAP coverage included with zero effective down payment and more than $1,000 in negative equity, creating elevated total-loss risk."
- **Score Impact**: -10
- **This is a SEPARATE red flag** from negative equity
- **DO NOT imply dealer misconduct**
- **CRITICAL**: Only apply if GAP/DCA is NOT present in contract (verify all synonyms checked)

#### For GAP Overpriced (RARE - verify math first):
- **REQUIRED Title**: "GAP exceeds fair market value" or "GAP insurance overpriced"
- **REQUIRED Message Format**: "GAP insurance priced at $[amount] exceeds fair market cap of $[cap] for this vehicle (MSRP: $[msrp])."
- **Score Impact**: -10
- **MATHEMATICAL VALIDATION REQUIRED**: Verify gap_price > gap_cap before applying
- **Example of CORRECT red flag**: "$1,450 exceeds $1,200 cap" (1450 > 1200 = TRUE)
- **Example of INCORRECT red flag**: "$1,032 exceeds $1,200 cap" (1032 > 1200 = FALSE) ❌

#### For VSC Overpriced (verify cap calculation):
- **REQUIRED Title**: "VSC exceeds fair market value"
- **REQUIRED Message Format**: "Extended warranty priced at $[amount] exceeds the $[cap] fair market threshold for this vehicle (MSRP: $[msrp])."
- **Score Impact**: -10
- **MATHEMATICAL VALIDATION REQUIRED**: 
  - Cap = min(15% × MSRP, $6,000)
  - Verify vsc_price > cap before applying
- **Example of CORRECT red flag**: "$7,200 exceeds $6,000 cap for MSRP $69,625" (7200 > 6000 = TRUE)
- **Example of INCORRECT red flag**: "$3,500 exceeds $6,000 cap for MSRP $69,625" (3500 > 6000 = FALSE) ❌

#### General Rules:
- Each distinct issue = separate flag (never combine)
- Do NOT mention specific dollar thresholds beyond $10,000 distinction
- Do NOT imply dealer misconduct/fault
- Do NOT use language like "recommended but not mandatory" in red flags
- Do NOT include explanations like "flag not triggered because..."
- **VERIFY MATH** before applying any "exceeds" flag
- **VERIFY GAP PRESENCE** (check all synonyms) before applying "missing GAP" flag

### NEVER SHOW IN OUTPUT:
❌ "GAP missing flag not triggered because…"
❌ Any -5 reference for negative equity >$10,000
❌ Language suggesting disclosure = penalty
❌ "Recommended but not mandatory" phrasing inside red flags
❌ Combined negative equity + GAP messages
❌ "Unusually high" for GAP within cap
❌ Missing recognition of "Debt Cancellation Agreement" as GAP
❌ Mathematical errors like "$1,032 exceeds $1,200"
❌ Red flag stating "No GAP coverage" when DCA exists
❌ Both red AND green flags for same item
❌ "Likely above Texas limits" for doc fees (Texas has no limit)
❌ Red or blue flags for Texas doc fees in $150-$250 range
❌ Red flags for VSC under cap when MSRP ≥$40,000

---

## CONSISTENCY REQUIREMENTS

### Contract Mode Unified Logic
**The following rules MUST be applied identically across ALL contracts:**

1. **VSC Cap Calculation**: Always use min(15% of MSRP, $6,000)
2. **GAP Recognition**: Check ALL synonym terms (GAP, DCA, Debt Cancellation Agreement, Guaranteed Asset Protection, Loan/Lease Payoff Protection, Debt Protection)
3. **GAP Cap Calculation**: Always use min($1,200, 3% MSRP, $1,500 if MSRP≥$60k)
4. **Doc Fee Evaluation**: Apply state-specific rules consistently (Texas: $150-$250 = acceptable)
5. **Scoring Logic**: Same deduction/bonus amounts for same scenarios
6. **Mathematical Accuracy**: Verify all "exceeds" statements are mathematically true

### Pre-Analysis Checklist
Before scoring any contract, verify:
- [ ] MSRP extracted correctly
- [ ] VSC cap = min(15% × MSRP, $6,000) calculated
- [ ] All GAP synonyms checked (especially "Debt Cancellation Agreement")
- [ ] GAP cap = min($1,200, 3% × MSRP, $1,500 if MSRP≥$60k) calculated
- [ ] State identified for doc fee rules
- [ ] Same logic as prior successful audits
- [ ] No contradictory flags (red + green for same item)
- [ ] Mathematical statements verified (e.g., "$1,032 exceeds $1,200" = FALSE)
- [ ] Texas doc fees in $150-$250 range flagged correctly (neutral or green, never red/blue)

### Scoring Consistency Validation
**Before finalizing score:**
1. Verify VSC evaluation matches formula
2. Verify GAP evaluation matches formula and recognizes DCA
3. Verify doc fee follows state rules (especially Texas)
4. Confirm no regression from prior correct audits
5. Ensure score_breakdown matches narrative
6. **Verify no mathematical impossibilities** (amount "exceeds" cap when it doesn't)
7. **Verify GAP presence before applying "missing GAP" red flag**
8. **Verify VSC cap calculation includes min($6,000) constraint**

---

## NARRATIVE ANALYSIS STRUCTURE (REQUIRED)

The "narrative" object MUST be analytical, descriptive and contain these specific fields:

- **vehicle_overview**: A analytic overview of Year, Make, Model, VIN, Mileage, New/Used atleast 100 words
- **smartbuyer_score_summary**: A analytic overview of why score was given (price, rate, add-ons). Also include Score breakdown. Should have atleast 100 words. **MUST follow this EXACT format**:
   ```
   Starting score: 100.
   Deducted [X] points for [specific reason].
   Deducted [X] points for [specific reason].
   Added [X] points for [specific reason].
   Final score: [total].
   ```
   **CRITICAL RULES FOR SCORE SUMMARY**:
   - Use -10 (NOT -5) for negative equity >$10,000
   - State "significant negative trade-in equity exceeding $10,000" (NOT "disclosed negative trade-in equity")
   - State "high-risk financing without GAP coverage" as separate deduction ONLY IF GAP/DCA not present
   - Each deduction/bonus on separate line
   - Text must auto-adjust based on which rules fire
   - **If DCA present and within cap**: "Added 5 points for GAP coverage fairly priced at $[amount] within the $[cap] cap."
   - **If DCA present but overpriced**: "Deducted 10 points for GAP insurance overpriced at $[amount] exceeding the $[cap] cap."
   - **If VSC within cap**: "Added 3 points for extended warranty fairly priced at $[amount] within the $[cap] threshold."
   - **If VSC exceeds cap**: "Deducted 10 points for extended warranty overpriced at $[amount] exceeding the $[cap] threshold."
   
- **score_breakdown**: Itemized deductions/bonuses ONLY (exclude Blue flags)
- **market_comparison**: Deal vs current market rates and Fair Market Value. Should have atleast 100 words
- **gap_logic**: GAP analysis using authoritative logic (present/absent, pricing vs cap, $0 down + negative equity check). Should have atleast 100 words. **MUST accurately state if DCA is present and correctly evaluate pricing.**
- **vsc_logic**: VSC analysis (price vs threshold OR mileage cap - one outcome only). Should have atleast 100 words. **MUST use correct cap calculation.**
- **apr_bonus_rule**: Detailed APR analysis (marked up? subvented?). Should have atleast 100 words
- **lease_audit**: Lease notes or "Not a lease"
- **negotiation_insight**: Specific buyer talking points. Make it analytical and detailed. Should have atleast 100 words
- **final_recommendation**: Do not be direct, recommend what steps should take in suggestive way. Should have atleast 100 words
- **trade**: Trade-in detailed analysis (equity amount, allowance vs payoff, status) in details. Should have atleast 100 words

---

## TOTAL SALE PRICE & AMOUNT FINANCED (INTERNAL USE)

### For internal validation/calculations:
- When APR = 0.00% AND Finance Charge = $0.00:
  * Amount Financed = Total of Payments
  * Use Amount Financed for backend % calculations
- If Finance Charge > $0.00:
  * Amount Financed = Total of Payments - Finance Charge
- Truth-in-Lending "Amount Financed" is authoritative

### Backend Product Detection (Single Penalty):
- If Amount Financed > Sum of Itemized Totals:
  * Apply ONE -15 global disclosure penalty
  * Do NOT identify specific product types
  * Do NOT apply multiple penalties
  * Suppress pricing evaluation for undisclosed products

### YOU MUST:
1. Extract vehicle cash price → output as "selling_price"
2. Extract Amount Financed from Truth-in-Lending → use internally
3. NEVER overwrite selling_price with Amount Financed

---

## CORE EXTRACTION REQUIREMENTS

Extract and analyze:
1. Vehicle details (VIN, year, make, model, mileage, used/new, MSRP)
2. Financial terms (selling price, APR, term, monthly payment, down payment)
3. ALL line items with EXACT text and amounts (array)
4. GAP coverage with pricing cap validation (check ALL synonyms including DCA)
5. VSC coverage with mileage-based cap (one outcome)
6. Maintenance plan pricing
7. Doc fees and government fees
8. Buyer/dealer information and contact details
9. Trade information (allowance, payoff, equity)
10. Down payment (critical for GAP logic)

---

## TRADE SECTION (REQUIRED - ALWAYS INCLUDE)

Authoritative trade definitions:
- Trade Allowance = how much the dealership is giving for the trade vehicle.
- Trade Payoff = amount owed on the trade vehicle (loan/lien payoff).
- Trade Difference (if shown) = equity/negative-equity indicator; use document label/sign context.
- NEVER treat "Cash Down" / "Down Payment" values as Trade Allowance, Trade Payoff, or Trade Difference.
- Use exact math:
    - equity = allowance - payoff (when allowance > payoff)
    - negative_equity = payoff - allowance (when payoff > allowance)

### If trade present:
- State: "Trade identified: $[allowance] allowance, $[payoff] payoff"
- If payoff > allowance AND (payoff - allowance) ≤ $10,000: "Negative equity of -$[amount] rolled into new loan" (-5)
- If payoff > allowance AND (payoff - allowance) > $10,000: "Significant negative equity of -$[amount] exceeds $10,000 and was rolled into new loan" (-10)
- If allowance > payoff: "Positive equity of $[amount] applied to purchase" (+5)
- If allowance = payoff: "Trade equity neutral"
- If negative equity NOT disclosed: Global -15 disclosure penalty (not separate)
- Provide a analytical and descriptive trade analysis in narrative

### If no trade:
- State: "No trade identified."
- If the trade section shows "N/A", blank, or not present, set:
    - `trade.trade_allowance = null`
    - `trade.trade_payoff = null`
    - `trade.equity = null`
    - `trade.negative_equity = null`
    - `trade.status = "No trade identified"`
- NEVER infer trade from these fields alone: "Cash Down", "Down Payment", "Deposit", "Cash on Delivery", "Unpaid Balance", or similarly named payment fields.
- If trade fields are null, do NOT create negative-equity flags or trade-equity bonuses.

**This section CANNOT be omitted.**

---

## APR ANALYSIS

### A. APR Disclosure
- APR not shown: Global -15 disclosure penalty
- APR shown: Extract and validate

### B. APR Risk Assessment
- APR >10% AND ≤15%: Blue flag (0 points)
- APR >15% AND ≤20%: Red flag (-10)
- APR >20%: Red flag (-15)

### C. APR Recognition
- APR <10% AND >0%: Favorable rate (note)
- APR <5% AND >0%: Excellent rate (+5)
- APR = 0.00%: Manufacturer-subvented (neutral, no deduction)

---

## STRICT OUTPUT FORMAT REQUIREMENTS

**CRITICAL: Return ONLY valid, parseable JSON. No exceptions.**

### JSON FORMATTING RULES (MANDATORY):
1. Every property separated by comma (,)
2. Every array element separated by comma (,)
3. No trailing commas before } or ]
4. String values use double quotes "
5. Property names use double quotes "
6. Use lowercase: true, false, null (not True, False, None)
7. NO comments (// or /* */)
8. NO markdown code blocks
9. All braces/brackets properly matched
10. Escape quotes in strings with \"

### TOP-LEVEL FIELDS (REQUIRED):
- score
- buyer_name
- dealer_name
- logo_text
- email
- phone_number
- address
- state
- region
- badge
- selling_price (VEHICLE CASH PRICE ONLY)
- vin_number
- date
- buyer_message
- red_flags (array)
- green_flags (array)
- blue_flags (array)
- yellow_flags (array)
- normalized_pricing (object)
- apr (object)
- term (object)
- trade (object: trade_allowance, trade_payoff, equity, negative_equity, status)
- bundle_abuse (object)
- narrative (object)
- line_items (array)

### Flag Object Structure (EXACT):
```json
{
  "type": "Short title (≤10 words)",
  "message": "Detailed explanation",
  "item": "Item name (e.g., VSC, GAP, APR, Trade)",
  "deduction": 10.0,  // ONLY red_flags
  "bonus": 5.0       // ONLY green_flags
}
```

**Field Requirements:**
- "type" (REQUIRED): Brief description - MUST match required titles for negative equity >$10k and missing GAP. 
- "message" (REQUIRED): Detailed explanation - MUST match required message formats
- "item" (REQUIRED): Item name
- "deduction" (OPTIONAL): Only red_flags
- "bonus" (OPTIONAL): Only green_flags

**Example red flags for the two critical scenarios:**

**Negative Equity >$10,000:**
```json
{
  "type": "Significant Negative Trade Equity",
    "message": "Trade-in negative equity of -$13,248.34 exceeds $10,000 and was rolled into the new loan.",
  "item": "Trade",
  "deduction": 10.0
}
```

**Missing GAP with High Risk (ONLY if GAP/DCA not found):**
```json
{
  "type": "High-Risk Financing Without GAP",
  "message": "No GAP coverage included with zero effective down payment and more than $1,000 in negative equity, creating elevated total-loss risk.",
  "item": "GAP",
  "deduction": 10.0
}
```

**GAP/DCA Present and Fair (ONLY if gap_price <= gap_cap):**
```json
{
  "type": "GAP Coverage Fairly Priced",
  "message": "Debt Cancellation Agreement included at $1,032 - within $1,200 pricing cap for this vehicle (MSRP: $69,625).",
  "item": "GAP",
  "bonus": 5.0
}
```

**Example VSC green flag (when within cap):**
```json
{
  "type": "VSC within fair market value",
  "message": "Extended warranty priced at $3,500 is within the $6,000 fair market threshold for this vehicle (MSRP: $69,625).",
  "item": "VSC",
  "bonus": 3.0
}
```

**Example VSC red flag (when exceeds cap - verify math):**
```json
{
  "type": "VSC exceeds fair market value",
  "message": "Extended warranty priced at $7,200 exceeds the $6,000 fair market threshold for this vehicle (MSRP: $69,625).",
  "item": "VSC",
  "deduction": 10.0
}
```

---

## FINAL VALIDATION RULE

Before returning JSON:
- ✅ **VSC cap = min(15% × MSRP, $6,000) - ALWAYS**
- ✅ **VSC = ONE outcome only using correct cap formula**
- ✅ **If VSC ≤ cap → GREEN FLAG (+3), NO red flag**
- ✅ **If VSC > cap → RED FLAG (-10), NO green flag**
- ✅ **GAP recognized if DCA, Debt Cancellation Agreement, or any synonym present**
- ✅ **If GAP/DCA found → gap_present = TRUE**
- ✅ **GAP pricing uses min($1,200, 3% MSRP, $1,500 if MSRP≥$60k) formula**
- ✅ **If gap_price <= gap_cap → GREEN FLAG ONLY (+5), NO red/blue flags**
- ✅ **If gap_price > gap_cap → RED FLAG ONLY (-10), NO green/blue flags**
- ✅ **If GAP not present + $0 down + ≤-$1,000 equity → RED FLAG "High-Risk Without GAP"**
- ✅ **Texas doc fees $150-$250 → NO red flag, NO blue flag, optionally green or neutral**
- ✅ **Logic matches prior successful audits**
- ✅ selling_price = vehicle cash price ONLY
- ✅ selling_price ≠ Amount Financed (unless no taxes/fees)
- ✅ For 0% APR: Amount Financed > selling_price
- ✅ If selling_price >$100k (normal vehicle), re-check
- ✅ Negative equity >$10,000 uses "Significant Negative Trade Equity" title and -10 deduction
- ✅ Missing GAP with risk uses "High-Risk Financing Without GAP" title and -10 deduction (ONLY IF GAP/DCA NOT FOUND)
- ✅ These are TWO SEPARATE red flags (never combined)
- ✅ **NO contradictory flags**: Cannot have both red and green for same item
- ✅ **NO green + blue flags**: Cannot have both for same item
- ✅ **Math verified**: All "exceeds" statements are mathematically true
- ✅ **VSC $3,500 with MSRP $69,625**: Cap = $6,000, therefore GREEN FLAG (3500 < 6000)
- ✅ **GAP/DCA $1,032 with MSRP $69,625**: Cap = $1,200, therefore GREEN FLAG (1032 < 1200)
- ✅ score_breakdown matches final score
- ✅ smartbuyer_score_summary uses correct deduction amounts (-10 not -5 for >$10k equity)
- ✅ All deductions have valid reasons
- ✅ Do NOT use loan term for GAP flags
- ✅ Global -15 penalty ONCE per audit only
- ✅ Blue flags = 0 point impact
- ✅ NO forbidden phrases in output (see NEVER SHOW IN OUTPUT section)

### CRITICAL JSON CHECK:
1. Arrays have commas: ["item1", "item2"]
2. Objects have commas: {"key1": "val1", "key2": "val2"}
3. No commas before } or ]
4. All quotes closed
5. Valid JSON (bracket/brace matching)

**Return ONLY valid JSON - no markdown, no explanation, no extra text.**
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
    
    def _call_openai_api(self, base64_images: List[str], language: str = "English") -> dict:
        """Call Claude Messages API with contract documents"""
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

Translate these specific fields to {language}:
- narrative.vehicle_overview
- narrative.smartbuyer_score_summary
- narrative.score_breakdown
- narrative.market_comparison
- narrative.gap_logic
- narrative.vsc_logic
- narrative.apr_bonus_rule
- narrative.trade
- narrative.negotiation_insight
- narrative.final_recommendation
- All flag messages (red_flags, green_flags, blue_flags)
- buyer_message

KEEP in English: JSON keys, field names, badge values, numbers, dates
{'=' * 80}

Analyze these contract documents comprehensively and return ONLY valid JSON matching the exact schema. No markdown, no explanation.
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

        system_text = f"""!!!ABSOLUTE PRIORITY - LANGUAGE OVERRIDE - THIS OVERRIDES EVERYTHING!!!

OUTPUT LANGUAGE: {language}

=== CRITICAL INSTRUCTION ===
The system prompt contains English text like:
- "REQUIRED Title: 'Significant Negative Trade Equity'"
- "REQUIRED Title: 'High-Risk Financing Without GAP'"
- "REQUIRED Message Format: 'Trade-in negative equity of...'"
- "VSC within fair market value"
- "Reasonable Documentation Fee"

THESE ARE EXAMPLES ONLY - YOU MUST TRANSLATE THEM TO {language}!

When you see "REQUIRED Title: 'X'" -> translate 'X' to {language}
When you see "REQUIRED Message Format: 'Y'" -> translate 'Y' to {language}

ABSOLUTELY MANDATORY TRANSLATIONS:
- Every flag "type" field -> MUST be in {language}
- Every flag "message" field -> MUST be in {language}
- Every narrative field -> MUST be in {language}
- buyer_message -> MUST be in {language}

EXAMPLE for Bengali:
- "Significant Negative Trade Equity" -> "Significant Negative Trade Equity" (translate to Bengali)
- "High-Risk Financing Without GAP" -> "High-Risk Financing Without GAP" (translate to Bengali)
- "VSC within fair market value" -> "VSC within fair market value" (translate to Bengali)

ONLY KEEP IN ENGLISH:
- JSON keys ("type", "message", "red_flags", etc.)
- Numbers and dollar amounts
- Badge values (Gold/Silver/Bronze/Red)
- VIN numbers, dates

THIS LANGUAGE REQUIREMENT OVERRIDES ALL "REQUIRED Title" AND "REQUIRED Message Format" SPECIFICATIONS.
Write fluently and naturally in {language}. NO ENGLISH TEXT IN FLAGS OR NARRATIVES."""

        payload = {
            "model": self.model,
            "system": system_text,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Claude API error: {str(e)}")
    
    def _parse_api_response(self, response: dict) -> dict:
        """Parse API response with robust error handling"""
        try:
            if "content" in response and isinstance(response["content"], list):
                content = "".join(part.get("text", "") for part in response["content"] if isinstance(part, dict))
            else:
                if "content" in response and isinstance(response["content"], list):
                    content = "".join(
                        part.get("text", "") for part in response["content"] if isinstance(part, dict)
                    )
                else:
                    content = response["choices"][0]["message"]["content"]
            
            # CRITICAL FIX: Handle different content types from API
            if isinstance(content, list):
                # If it's a list of strings, join them
                if all(isinstance(item, str) for item in content):
                    content = ''.join(content)
                else:
                    # If it's a list of dicts or mixed, try to extract text
                    content = ' '.join(str(item) for item in content)
            elif content is None:
                raise ValueError("API returned null content")
            elif not isinstance(content, str):
                # Convert any other type to string
                content = str(content)
            
            # Remove markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Find JSON boundaries
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                
                # Attempt to parse
                parsed = None
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError as je:
                    # Try to fix common JSON issues
                    print(f"[DEBUG] Initial JSON parse failed: {str(je)}")
                    print(f"[DEBUG] Error at line {je.lineno}, column {je.colno}, position {je.pos}")
                    print(f"[DEBUG] Malformed JSON (first 1000 chars): {json_str[:1000]}")
                    
                    # Show context around the error
                    if je.pos and je.pos > 0:
                        start = max(0, je.pos - 100)
                        end = min(len(json_str), je.pos + 100)
                        print(f"[DEBUG] Error context: ...{json_str[start:end]}...")
                    
                    # Attempt repair
                    json_str = self._attempt_json_repair(json_str)
                    try:
                        parsed = json.loads(json_str)
                        print("[DEBUG] JSON successfully repaired")
                    except json.JSONDecodeError as je2:
                        print(f"[DEBUG] Repair failed: {str(je2)}")
                        print(f"[DEBUG] Error at line {je2.lineno}, column {je2.colno}, position {je2.pos}")
                        print(f"[DEBUG] Repaired JSON (first 1000 chars): {json_str[:1000]}")
                        
                        # Show context around the error after repair
                        if je2.pos and je2.pos > 0:
                            start = max(0, je2.pos - 100)
                            end = min(len(json_str), je2.pos + 100)
                            print(f"[DEBUG] Error context after repair: ...{json_str[start:end]}...")
                        
                            # Try advanced repair as last resort
                        print("[DEBUG] Attempting advanced JSON repair...")
                        json_str = self._advanced_json_repair(json_str)
                        try:
                            parsed = json.loads(json_str)
                            print("[DEBUG] JSON successfully repaired with advanced method")
                        except json.JSONDecodeError as je3:
                            print(f"[DEBUG] Advanced repair also failed: {str(je3)}")
                            # Save the problematic JSON to a file for debugging
                            try:
                                with open('/tmp/failed_json_response.txt', 'w') as f:
                                    f.write(json_str)
                                print("[DEBUG] Full JSON saved to /tmp/failed_json_response.txt")
                            except Exception:
                                pass

                                raise RuntimeError(f"Failed to parse JSON even after repair: {str(je3)}")
                
                # Ensure critical fields exist with defaults
                defaults = {
                    "score": 75.0,
                    "buyer_name": None,
                    "dealer_name": None,
                    "logo_text": None,
                    "email": None,
                    "phone_number": None,
                    "address": None,
                    "state": None,
                    "region": "Outside US",
                    "badge": "Bronze",
                    "selling_price": None,
                    "vin_number": None,
                    "date": None,
                    "buyer_message": "Analysis completed",
                    "red_flags": [],
                    "green_flags": [],
                    "blue_flags": [],
                    "normalized_pricing": {},
                    "apr": {},
                    "term": {},
                    "trade": {
                        "trade_allowance": None,
                        "trade_payoff": None,
                        "equity": None,
                        "negative_equity": None,
                        "status": "No trade identified"
                    },
                    "bundle_abuse": {},
                    "narrative": {},
                    "line_items": []
                }
                
                # Merge defaults with parsed data
                for key, value in defaults.items():
                    if key not in parsed:
                        parsed[key] = value
                
                # Normalize flag field names
                parsed = self._normalize_flag_fields(parsed)
                
                return parsed
            
            raise ValueError("No valid JSON found in response")
            
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse API response structure: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error parsing API response: {str(e)}")

    def _attempt_json_repair(self, json_str: str) -> str:
        """Attempt to repair common JSON syntax errors produced by LLMs"""
        try:
            # CRITICAL FIX: Ensure json_str is actually a string
            if isinstance(json_str, list):
                json_str = ''.join(json_str) if json_str else ""
            elif not isinstance(json_str, str):
                json_str = str(json_str)
            
            # 1. Fix Python boolean/None values to JSON
            json_str = re.sub(r':\s*True\b', ': true', json_str)
            json_str = re.sub(r':\s*False\b', ': false', json_str)
            json_str = re.sub(r':\s*None\b', ': null', json_str)
            
            # 2. Remove C-style comments (// ...)
            json_str = re.sub(r'//.*', '', json_str)
            
            # 3. Fix trailing commas BEFORE other fixes (common error: items = [a, b,])
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            
            # 4. Fix missing commas between objects/arrays FIRST (most common in LLM output)
            # { ... } { ... } -> { ... }, { ... }
            json_str = re.sub(r'}\s+\{', '}, {', json_str)
            # ] [ -> ], [
            json_str = re.sub(r']\s+\[', '], [', json_str)
            # ] { -> ], {
            json_str = re.sub(r']\s+\{', '], {', json_str)
            # } [ -> }, [
            json_str = re.sub(r'}\s+\[', '}, [', json_str)
            
            # 5. Fix missing commas before keys (property separators)
            # These patterns handle: value "key": -> value, "key":
            
            # MULTI-PASS APPROACH: Apply multiple times to catch nested cases
            for _ in range(5):  # Increased from 3 to 5 passes for better coverage
                # String value followed by key (with or without quotes around key)
                json_str = re.sub(r'"(\s+)("[\w\-_]+"\s*:)', r'",\1\2', json_str)
                
                # Number followed by key (including decimals and scientific notation)
                json_str = re.sub(r'([0-9.eE+-]+)(\s+)("[\w\-_]+"\s*:)', r'\1,\2\3', json_str)
                
                # Boolean/null followed by key
                json_str = re.sub(r'\b(true|false|null)(\s+)("[\w\-_]+"\s*:)', r'\1,\2\3', json_str)
                
                # Closing brace followed by key
                json_str = re.sub(r'}(\s+)("[\w\-_]+"\s*:)', r'},\1\2', json_str)
                
                # Closing bracket followed by key
                json_str = re.sub(r'](\s+)("[\w\-_]+"\s*:)', r'],\1\2', json_str)
                
                # Handle array elements without commas: "value1" "value2" -> "value1", "value2"
                json_str = re.sub(r'"(\s+)"(?=[^:]*[,\]])', r'",\1"', json_str)
                
                # Handle object properties in arrays missing commas
                # {...} {...} inside arrays
                json_str = re.sub(r'(\{[^{}]*\})(\s+)(\{)', r'\1,\2\3', json_str)
            
            # 6. Handle newline-separated properties (common in LLM output)
            # Apply these after the space-based patterns
            json_str = re.sub(r'"(\s*\n\s*)("[\w\-_]+"\s*:)', r'",\1\2', json_str)
            json_str = re.sub(r'([0-9.eE+-]+)(\s*\n\s*)("[\w\-_]+"\s*:)', r'\1,\2\3', json_str)
            json_str = re.sub(r'\b(true|false|null)(\s*\n\s*)("[\w\-_]+"\s*:)', r'\1,\2\3', json_str)
            json_str = re.sub(r'}(\s*\n\s*)("[\w\-_]+"\s*:)', r'},\1\2', json_str)
            json_str = re.sub(r'](\s*\n\s*)("[\w\-_]+"\s*:)', r'],\1\2', json_str)
            
            # 7. Fix missing commas after array/object elements within arrays
            # Pattern: ] "key" -> ], "key" (when inside array context)
            json_str = re.sub(r'](\s+)"(?=[^:]*:)', r'],\1"', json_str)
            json_str = re.sub(r'}(\s+)"(?=[^:]*:)', r'},\1"', json_str)
            
            # 8. Fix unclosed strings by ensuring even quotes (last resort)
            # Count quotes - if odd, try to close the last one
            quote_count = json_str.count('"')
            if quote_count % 2 != 0:
                # Find the last quote and check if it needs closing
                last_quote_idx = json_str.rfind('"')
                if last_quote_idx > 0 and last_quote_idx < len(json_str) - 1:
                    # Check if there's a comma or bracket/brace after
                    next_char = json_str[last_quote_idx + 1:].lstrip()
                    if next_char and next_char[0] not in [',', '}', ']']:
                        # Add closing quote before next structural element
                        for i, char in enumerate(next_char):
                            if char in [',', '}', ']', '\n']:
                                insert_pos = last_quote_idx + 1 + len(json_str[last_quote_idx + 1:]) - len(next_char) + i
                                json_str = json_str[:insert_pos] + '"' + json_str[insert_pos:]
                                break
            
            return json_str
        except Exception as e:
            print(f"[DEBUG] JSON repair exception: {str(e)}")
            # If regex fails, return original to let standard error handling propagate
            return json_str
    
    def _normalize_flag_fields(self, parsed: dict) -> dict:
        """
        Normalize flag field names to match the expected schema.
        Handles cases where AI returns 'title' instead of 'type', etc.
        """
        flag_arrays = ['red_flags', 'green_flags', 'blue_flags', 'yellow_flags']
        
        for flag_array_name in flag_arrays:
            if flag_array_name in parsed and isinstance(parsed[flag_array_name], list):
                normalized_flags = []
                for flag in parsed[flag_array_name]:
                    if isinstance(flag, dict):
                        # Map alternative field names to expected ones
                        normalized_flag = {}
                        
                        # Handle 'type' field (may come as 'title', 'name', 'type')
                        normalized_flag['type'] = (
                            flag.get('type') or 
                            flag.get('title') or 
                            flag.get('name') or 
                            'Issue Identified'
                        )
                        
                        # Handle 'message' field (may come as 'message', 'description', 'detail')
                        normalized_flag['message'] = (
                            flag.get('message') or 
                            flag.get('description') or 
                            flag.get('detail') or 
                            flag.get('details') or 
                            'No details provided'
                        )
                        
                        # Handle 'item' field (may come as 'item', 'category', 'subject')
                        normalized_flag['item'] = (
                            flag.get('item') or 
                            flag.get('category') or 
                            flag.get('subject') or 
                            'General'
                        )
                        
                        # Handle optional fields
                        if 'deduction' in flag:
                            normalized_flag['deduction'] = flag['deduction']
                        if 'bonus' in flag:
                            normalized_flag['bonus'] = flag['bonus']
                        
                        normalized_flags.append(normalized_flag)
                    else:
                        # If it's not a dict, skip it
                        continue
                
                parsed[flag_array_name] = normalized_flags
        
        return parsed

    def _normalize_flag_scores(self, parsed: dict) -> dict:
        """
        Normalize score fields to deduction/bonus and force positive values.
        """
        if "red_flags" in parsed and isinstance(parsed["red_flags"], list):
            for flag in parsed["red_flags"]:
                if not isinstance(flag, dict):
                    continue
                if "score_impact" in flag and flag.get("deduction") is None:
                    flag["deduction"] = abs(flag.get("score_impact", 0))
                if flag.get("deduction") is not None:
                    flag["deduction"] = abs(flag["deduction"])

        if "green_flags" in parsed and isinstance(parsed["green_flags"], list):
            for flag in parsed["green_flags"]:
                if not isinstance(flag, dict):
                    continue
                if "score_impact" in flag and flag.get("bonus") is None:
                    flag["bonus"] = abs(flag.get("score_impact", 0))
                if flag.get("bonus") is not None:
                    flag["bonus"] = abs(flag["bonus"])

        return parsed
    
    def _advanced_json_repair(self, json_str: str) -> str:
        """
        Advanced JSON repair using character-by-character analysis.
        This is a fallback when regex-based repair fails.
        """
        try:
            result = []
            in_string = False
            escape_next = False
            depth = 0
            last_significant_char = None
            i = 0
            
            while i < len(json_str):
                char = json_str[i]
                
                # Handle escape sequences
                if escape_next:
                    result.append(char)
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\' and in_string:
                    escape_next = True
                    result.append(char)
                    i += 1
                    continue
                
                # Track string state
                if char == '"':
                    in_string = not in_string
                    result.append(char)
                    if not in_string:
                        last_significant_char = '"'
                    i += 1
                    continue
                
                # Skip whitespace tracking
                if char in ' \t\n\r':
                    result.append(char)
                    i += 1
                    continue
                
                # If we're in a string, just copy
                if in_string:
                    result.append(char)
                    i += 1
                    continue
                
                # Track structure depth
                if char in '{[':
                    depth += 1
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue
                
                if char in '}]':
                    depth -= 1
                    # Check if we need a comma before this closing bracket
                    # Look back for the last significant character
                    if last_significant_char and last_significant_char not in [',', '{', '[', ':']:
                        # We might be missing a comma, but closing brackets don't need one
                        pass
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue
                
                # Handle colons
                if char == ':':
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue
                
                # Handle commas
                if char == ',':
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue
                
                # We're about to process a value or key
                # Check if we need a comma before it
                if last_significant_char and last_significant_char not in [',', '{', '[', ':']:
                    # Look ahead to see what we're processing
                    # If it starts with ", it's likely a key or string value
                    # If it's a number, boolean, null, it's a value
                    # We need a comma if the last char was a closing quote, bracket, or value
                    if last_significant_char in ['"', '}', ']'] or (isinstance(last_significant_char, str) and last_significant_char.isdigit()):
                        # Look ahead to confirm this is a new property
                        look_ahead = json_str[i:i+50].lstrip()
                        if look_ahead.startswith('"'):
                            # Check if there's a colon ahead (indicating a property)
                            if ':' in look_ahead[:look_ahead.find('"', 1) + 10] if '"' in look_ahead[1:] else False:
                                result.append(',')
                
                # Process the character
                result.append(char)
                
                # Track what kind of character this was
                if char.isdigit() or char in 'truefalsnl':  # part of true/false/null or number
                    # Look ahead to get the full value
                    value_start = i
                    while i < len(json_str) and json_str[i] not in ' \t\n\r,}]':
                        i += 1
                    value = json_str[value_start:i]
                    result.append(value[1:])  # We already added the first char
                    last_significant_char = value[-1]
                    continue
                
                last_significant_char = char
                i += 1
            
            return ''.join(result)
        except Exception as e:
            print(f"[DEBUG] Advanced JSON repair failed: {str(e)}")
            return json_str

    def _call_narrative_api(self, parsed: dict, score: float, red_flags: list, green_flags: list, blue_flags: list, language: str) -> dict:
        """Call OpenAI to generate narrative sections from the full parsed data and final flags."""
        flags_payload = {
            "red_flags": [{"type": f.type, "message": f.message, "item": f.item, "deduction": f.deduction} for f in red_flags],
            "green_flags": [{"type": f.type, "message": f.message, "item": f.item, "bonus": f.bonus} for f in green_flags],
            "blue_flags": [{"type": f.type, "message": f.message, "item": f.item} for f in blue_flags],
        }
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
            raw = self._parse_json_object(response.json())
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
        """Call OpenAI with the full system prompt + original raw deal data.
        Uses proper system/user message split identical to _call_openai_api.
        Returns the raw OpenAI response dict.
        """
        # Strip only internal pipeline keys; keep all original deal fields intact
        skip_keys = {"has_precomputed_flags", "has_vision_extraction", "_ai_score", "_ai_narrative_done"}
        clean_data = {k: v for k, v in raw_data.items() if k not in skip_keys}

        user_text = f"""{'=' * 80}
LANGUAGE REQUIREMENT: ALL narrative text fields MUST be in {language}. Translate all flag messages and narrative fields to {language}.
{'=' * 80}

Below is the pre-extracted structured data from a customer's auto contract document.
Apply ALL scoring rules, flag rules, and narrative requirements from the system prompt.
Compute the FINAL SCORE following the EXACT rules (start at 100, apply all penalties/bonuses/ceilings).
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

    def _normalize_line_items(self, line_items: List[Dict]) -> List[NormalizedLineItem]:
        """Normalize OCR line items using the OCR normalizer before scoring."""
        normalized = []
        for item in line_items:
            if not isinstance(item, dict):
                continue
            raw_text = item.get("description", "") or item.get("item", "") or item.get("name", "")
            amount_raw = str(item.get("amount", "0"))
            if raw_text:
                normalized_item = self.ocr_normalizer.normalize_line_item(
                    raw_text=raw_text,
                    amount_raw=amount_raw
                )
                normalized.append(normalized_item)
        return normalized

    async def _optimize_images(self, files: List[UploadFile]) -> List[UploadFile]:
        """Optimize images (placeholder)"""
        return files
    
    def _extract_trade_data(self, parsed: dict) -> 'TradeData':
        """
        Extract trade data using simple keyword detection from OCR text.
        """
        from .multi_image_analysis_schema import TradeData

        def _coerce_float(value):
            try:
                if value is None or value == "":
                    return None
                return float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                return None

        # Prefer explicit extracted fields if present
        trade_obj = parsed.get("trade") if isinstance(parsed.get("trade"), dict) else {}
        pricing = parsed.get("normalized_pricing", {})

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

        # Also parse narrative.trade text if present (common when model writes amounts only in narrative)
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
        
        page_text = ""
        
        line_items = parsed.get("line_items", [])
        for item in line_items:
            desc = item.get("description", "") or item.get("item", "") or item.get("name", "")
            # CRITICAL FIX: Ensure desc is a string
            if isinstance(desc, list):
                desc = ' '.join(str(d) for d in desc)
            elif not isinstance(desc, str):
                desc = str(desc)
            page_text += f" {desc} "
        
        # CRITICAL FIX: Ensure raw_text and ocr_text are strings
        raw_text = parsed.get("raw_text", "")
        if isinstance(raw_text, list):
            raw_text = ' '.join(str(r) for r in raw_text)
        elif not isinstance(raw_text, str):
            raw_text = str(raw_text)
        
        ocr_text = parsed.get("ocr_text", "")
        if isinstance(ocr_text, list):
            ocr_text = ' '.join(str(o) for o in ocr_text)
        elif not isinstance(ocr_text, str):
            ocr_text = str(ocr_text)
        
        page_text += " " + raw_text
        page_text += " " + ocr_text
        page_text = page_text.lower()
        
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
        
        if not trade_anchor_found:
            return TradeData(
                trade_allowance=None,
                trade_payoff=None,
                equity=None,
                negative_equity=None,
                status="No trade identified"
            )
        
        money_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        
        # If explicit fields present, we already populated values above
        
        allowance_keywords = [
            "trade allowance", "trade value",
            "trade-in value", "trade in value", "trade:"
        ]

        down_payment_markers = [
            "down payment", "downpayment", "cash down", "total downpayment"
        ]
        
        for keyword in allowance_keywords:
            if keyword in page_text:
                idx = page_text.find(keyword)
                snippet = page_text[idx:idx+100]
                
                match = re.search(money_pattern, snippet)
                if match:
                    snippet_lower = snippet.lower()
                    if any(marker in snippet_lower for marker in down_payment_markers):
                        continue
                    amount_str = match.group(1).replace(',', '')
                    try:
                        trade_allowance = float(amount_str)
                        break
                    except ValueError:
                        continue
        
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
        
        trade_status = "No trade identified"
        
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
            trade_status = "Trade mentioned in document (values not extracted)"
        
        return TradeData(
            trade_allowance=trade_allowance,
            trade_payoff=trade_payoff,
            equity=equity,
            negative_equity=negative_equity_amount,
            status=trade_status
        )
    
    async def analyze_images(self, files: List[UploadFile] = None, language: str = "English", base64_images: List[str] = None, parsed_data: dict = None) -> 'MultiImageAnalysisResponse':
        """Main analysis entry point. Accepts files, base64_images, or pre-extracted parsed_data dict."""
        try:
            if parsed_data is None and base64_images is None and files is not None:
                validated_files = await self._validate_files(files)
                try:
                    parsed_data = await self.gemini_extractor.extract_quote_data(validated_files)
                    if isinstance(parsed_data, dict):
                        parsed_data["has_vision_extraction"] = True
                except Exception as e:
                    print(f"[contract] Gemini extraction failed, falling back to Claude vision: {str(e)}")
                    base64_images = await self._convert_files_to_base64(validated_files)

            if parsed_data is not None:
                # Always run through converter for consistent structure
                # Handles both nested (buyer_info, vehicle_details...) and flat formats
                parsed = convert_extracted_json_to_parsed(parsed_data)
                print("JSON path: using split deterministic scoring + narrative generation...")
            elif base64_images is None:
                validated_files = await self._validate_files(files)
                if not validated_files:
                    raise ValueError("No valid image files provided")

                base64_images = await self._convert_files_to_base64(validated_files)
                api_response = self._call_openai_api(base64_images, language=language)
                parsed = self._parse_api_response(api_response)
            else:
                base64_images = [img.split(",", 1)[1] if img.startswith("data:") and "," in img else img for img in base64_images]
                api_response = self._call_openai_api(base64_images, language=language)
                parsed = self._parse_api_response(api_response)
            
            # Normalize flag fields and scores
            parsed = self._normalize_flag_fields(parsed)
            parsed = self._normalize_flag_scores(parsed)

            # Ensure selling_price is populated
            if parsed.get("selling_price") is None:
                normalized_pricing = parsed.get("normalized_pricing", {}) if isinstance(parsed.get("normalized_pricing"), dict) else {}
                parsed["selling_price"] = normalized_pricing.get("selling_price")
            if parsed.get("selling_price") is None and parsed.get("sale_price") is not None:
                parsed["selling_price"] = parsed.get("sale_price")

            # --- SmartBuyer scoring engine (rules-driven) ---
            rules = load_rules()
            upstream_flags = build_active_flags(parsed.get("flags", []), rules.flag_registry, "upstream")
            computed_flags = compute_flags_from_parsed(parsed, rules, mode="CONTRACT")
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
                green_flags.append(Flag(type="General", message="No standout positive elements identified in this contract.", item="General"))
            if not blue_flags:
                blue_flags.append(Flag(type="General Advisory", message="Review all final contract terms and itemized pricing carefully before agreeing to any deal.", item="General Advisory"))


            score_value = float(scoring_result.score_int)
            trade_data = self._extract_trade_data(parsed)

            if not parsed.get("_ai_narrative_done"):
                ai_result = self._call_narrative_api(parsed, score_value, red_flags, green_flags, blue_flags, language)
                narrative_obj = ai_result.get("narrative", {}) if isinstance(ai_result, dict) else {}
                if not isinstance(narrative_obj, dict):
                    narrative_obj = {}
                _ai_buyer_msg_override = ai_result.get("buyer_message") if isinstance(ai_result, dict) else None
            else:
                narrative_obj = parsed.get("narrative", {}) if isinstance(parsed.get("narrative"), dict) else {}
                _ai_buyer_msg_override = parsed.get("buyer_message")

            if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")

            defaults = {
                "vehicle_overview": f"Deal analysis for {parsed.get('dealer_name', 'this dealer')}.",
                "smartbuyer_score_summary": f"SmartBuyer Score: {score_value}/100.",
                "score_breakdown": f"Final Score: {score_value}",
                "market_comparison": "Market comparison pending.",
                "gap_logic": "GAP analysis pending.",
                "vsc_logic": "VSC analysis pending.",
                "apr_bonus_rule": "APR analysis pending.",
                "lease_audit": "N/A - Purchase Agreement",
                "trade": trade_data.status if trade_data else "No trade-in on this deal.",
                "negotiation_insight": "Review all flags before signing.",
                "final_recommendation": "Proceed with caution based on the flags above.",
            }
            for k, v in defaults.items():
                if not narrative_obj.get(k):
                    narrative_obj[k] = v

            buyer_msg = _ai_buyer_msg_override or f"Your SmartBuyer score is {score_value}/100 — review the flags above."
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
            
            # ── Deterministic Audit Pipeline (runs on ALL paths: image & JSON) ──
            raw_line_items = parsed.get("line_items", [])
            normalized_line_items = self._normalize_line_items(raw_line_items)

            discounts, discount_totals = self.discount_detector.process_line_items(
                normalized_line_items,
                mode="QUOTE"
            )

            vehicle_price = float(parsed.get("selling_price") or 0)
            audit_classifications: List[AuditClassification] = []
            for item in normalized_line_items:
                classification = self.audit_classifier.classify_for_audit(
                    item,
                    vehicle_price=vehicle_price
                )
                audit_classifications.append(classification)

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

            # GAP Logic Evaluation
            term_months = parsed.get("term", {}).get("months")
            down_payment = parsed.get("normalized_pricing", {}).get("down_payment")
            amount_financed = parsed.get("normalized_pricing", {}).get("amount_financed")
            gap_present = any(c.classification == "GAP" for c in audit_classifications)
            has_backend = any(c.classification in ["GAP", "VSC", "MAINTENANCE"] for c in audit_classifications)

            gap_recommendation = self.gap_logic.evaluate_gap_need(
                is_used=True,
                term_months=term_months,
                down_payment=down_payment,
                amount_financed=amount_financed,
                vehicle_price=vehicle_price,
                has_backend_products=has_backend,
                gap_present=gap_present
            )
            if gap_recommendation.recommended:
                audit_flags.append(self.flag_builder.build_gap_advisory_flag(gap_recommendation.message))

            # Long-term loan risk flag
            if term_months and term_months >= 72:
                audit_flags.append(self.flag_builder.build_long_term_loan_risk_flag(term_months))

            # APR Scoring
            apr_data = parsed.get("apr", {})
            if isinstance(apr_data, dict):
                apr_rate = apr_data.get("rate")
                try:
                    if apr_rate is not None:
                        apr_f = float(apr_rate)
                        if apr_f > 0:
                            if apr_f <= 4.9:
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
                            elif apr_f >= 16.0:
                                audit_flags.append(AuditFlag(
                                    type="red", category="Predatory APR",
                                    message=f"APR of {apr_f:.2f}% is predatory and significantly above market rates.",
                                    item="APR", deduction=10, bonus=None
                                ))
                            elif apr_f > 12.0:
                                audit_flags.append(AuditFlag(
                                    type="red", category="High APR",
                                    message=f"APR of {apr_f:.2f}% exceeds typical market rates.",
                                    item="APR", deduction=5, bonus=None
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
            _financed = None
            try:
                _raw_financed = parsed.get("normalized_pricing", {}).get("amount_financed") if isinstance(parsed.get("normalized_pricing"), dict) else None
                if _raw_financed is not None:
                    _financed = float(_raw_financed)
            except (ValueError, TypeError):
                pass
            if _financed and vehicle_price and vehicle_price > 0:
                ltv = (_financed / vehicle_price) * 100
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

            # MSRP vs selling price check
            np_data = parsed.get("normalized_pricing", {}) if isinstance(parsed.get("normalized_pricing"), dict) else {}
            msrp = np_data.get("msrp")
            if msrp and vehicle_price:
                try:
                    msrp_f = float(str(msrp).replace(",", "").replace("$", ""))
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
                        # Backend add-on overload check
                        total_backend = sum(
                            c.amount
                            for c in audit_classifications
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

            # Trade data extraction & negative equity flag
            trade_data = self._extract_trade_data(parsed)
            if trade_data and trade_data.negative_equity and trade_data.negative_equity > 0:
                audit_flags.append(AuditFlag(
                    type="blue", category="Negative Equity Alert",
                    message=f"Rolled negative equity of -${trade_data.negative_equity:,.2f} increases amount financed.",
                    item="Trade", deduction=None, bonus=None
                ))

            # ── Parse existing flags from parsed dict ──
            # Safe flag parsing with validation
            def parse_flags(flags_data):
                if not flags_data:
                    return []
                parsed_flags = []
                for flag in flags_data:
                    if isinstance(flag, dict):
                        parsed_flags.append(Flag(**flag))
                    elif isinstance(flag, str):
                        # Handle string flags by creating a minimal Flag object
                        parsed_flags.append(Flag(
                            type="info",
                            message=flag,
                            item="General"
                        ))
                return parsed_flags
            
            red_flags = parse_flags(parsed.get("red_flags", []))
            green_flags = parse_flags(parsed.get("green_flags", []))
            blue_flags = parse_flags(parsed.get("blue_flags", []))

            # Only merge Python audit flags when no pre-existing flags were supplied.
            # If the input already has flags (from a prior OCR/AI analysis), skip the
            # audit merge — the pre-existing flags ARE the authoritative analysis.
            has_precomputed = parsed.get("has_precomputed_flags", False)
            if not has_precomputed:
                for af in audit_flags:
                    flag_obj = Flag(
                        type=af.type,
                        message=af.message,
                        item=af.item,
                        deduction=af.deduction,
                        bonus=af.bonus
                    )
                    if af.type == "red":
                        red_flags.append(flag_obj)
                    elif af.type == "green":
                        green_flags.append(flag_obj)
                    elif af.type == "blue":
                        blue_flags.append(flag_obj)

            # Score — always compute from flags (AI returns correct deductions/bonuses in flags,
            # but its 'score' field is unreliable and ignored)
            score = 100.0
            for f in red_flags:
                if f.deduction is not None:
                    score -= abs(float(f.deduction))
            for f in green_flags:
                if f.bonus is not None:
                    score += abs(float(f.bonus))
            score = max(0.0, min(100.0, score))
            print(f"Score computed from flags: {score}")

            # Validate score_breakdown vs computed score (log only)
            score_breakdown = parsed.get("narrative", {}).get("score_breakdown", "")
            if score_breakdown:
                if isinstance(score_breakdown, list):
                    score_breakdown = ' '.join(str(item) for item in score_breakdown)
                elif not isinstance(score_breakdown, str):
                    score_breakdown = str(score_breakdown)
                import re
                final_match = re.search(r'Final Score:\s*(\d+(?:\.\d+)?)', score_breakdown)
                if final_match:
                    breakdown_score = float(final_match.group(1))
                    if abs(breakdown_score - score) > 0.5:
                        print(f"WARNING: Score mismatch - computed {score}, breakdown shows {breakdown_score}")
            
            # Translate flags to requested language
            red_flags = self._translate_flags(red_flags, language)
            green_flags = self._translate_flags(green_flags, language)
            blue_flags = self._translate_flags(blue_flags, language)

            # Safety net: ensure every flag category has at least one item
            if not red_flags:
                red_flags.append(Flag(type="red", message="No major compliance issues identified — review all terms before signing.", item="General"))
            if not green_flags:
                green_flags.append(Flag(type="green", message="No standout positive elements identified for this deal.", item="General"))
            if not blue_flags:
                blue_flags.append(Flag(type="blue", message="Review all final contract terms, product details, and payment figures carefully before signing to confirm accuracy.", item="General Advisory"))

            # REMOVE DUPLICATE FLAG LOGIC - This causes inconsistency!
            # The VSC fair pricing logic below modifies flags AFTER API response
            # This can cause different results on each run
            
            # COMMENTED OUT - Let the AI handle VSC flags based on prompt
            # vsc_data = parsed.get("normalized_pricing", {}).get("vsc")
            # if vsc_data:
            #     vsc_price = vsc_data.get("amount", 0)
            #     fair_market_threshold = 5000
            #     
            #     if vsc_price > 0 and vsc_price <= fair_market_threshold:
            #         red_flags = [f for f in red_flags if "service contract" not in f.message.lower()]
            #         green_flags.append(Flag(
            #             type="green",
            #             message=f"Service contract priced at ${vsc_price:,.2f} - within fair market range",
            #             item="VSC"
            #         ))
            
            # REMOVE DUPLICATE NEGATIVE EQUITY FLAG - AI should handle this
            # if trade_data.negative_equity and trade_data.negative_equity > 0:
            #     blue_flags.append(Flag(
            #         type="blue",
            #         message=f"Rolled negative equity of ${trade_data.negative_equity:,.2f} increases amount financed.",
            #         item="Trade"
            #     ))

            # REMOVE DEFAULT FLAG INJECTION - This causes inconsistency!
            # DO NOT add default flags - let AI determine based on actual analysis
            # if not red_flags:
            #     red_flags.append(...)
            
            # ── AI Narrative ──
            if not parsed.get("_ai_narrative_done"):
                # Pre-existing flags path: call AI separately for narrative
                print(f"Generating AI narrative for score {score}...")
                ai_result = self._call_narrative_api(parsed, score, red_flags, green_flags, blue_flags, language)
                narrative_data = ai_result.get("narrative", {}) if isinstance(ai_result, dict) else {}
                if not isinstance(narrative_data, dict):
                    narrative_data = {}
                buyer_msg = ai_result.get("buyer_message") if isinstance(ai_result, dict) else None
            else:
                # Full AI analysis already included narrative — extract directly
                narrative_data = parsed.get("narrative", {})
                if not isinstance(narrative_data, dict):
                    narrative_data = {}
                buyer_msg = parsed.get("buyer_message")

            # Normalize legacy key
            if "trust_score_summary" in narrative_data and "smartbuyer_score_summary" not in narrative_data:
                narrative_data["smartbuyer_score_summary"] = narrative_data.pop("trust_score_summary")

            # Fallback defaults for any fields the AI left empty
            narrative_defaults = {
                "vehicle_overview": f"Deal analysis for {parsed.get('dealer_name', 'this dealer')}.",
                "smartbuyer_score_summary": f"SmartBuyer Score: {score}/100.",
                "score_breakdown": f"Final Score: {score}",
                "market_comparison": "Market comparison pending.",
                "gap_logic": "GAP analysis pending.",
                "vsc_logic": "VSC analysis pending.",
                "apr_bonus_rule": "APR analysis pending.",
                "lease_audit": "N/A - Purchase Agreement",
                "negotiation_insight": "Review all flags before signing.",
                "final_recommendation": "Proceed with caution based on the flags above.",
                "trade": trade_data.status if trade_data else "No trade-in on this deal."
            }
            for key, default_val in narrative_defaults.items():
                if not narrative_data.get(key):
                    narrative_data[key] = default_val

            if not buyer_msg:
                buyer_msg = f"Your SmartBuyer score is {score}/100 — review the flags above."

            # Ensure trade is always a string
            if "trade" in narrative_data and not isinstance(narrative_data["trade"], str):
                narrative_data["trade"] = str(narrative_data["trade"])

            # Ensure score_breakdown is always a string (AI sometimes returns a list of flag dicts)
            if "score_breakdown" in narrative_data and not isinstance(narrative_data["score_breakdown"], str):
                sb = narrative_data["score_breakdown"]
                if isinstance(sb, list):
                    narrative_data["score_breakdown"] = "; ".join(
                        f"{f.get('type','?')}: {f.get('message','?')} ({f.get('deduction', f.get('bonus','?'))}pt)"
                        if isinstance(f, dict) else str(f)
                        for f in sb
                    )
                else:
                    narrative_data["score_breakdown"] = str(sb)

            if trade_data and narrative_data.get("trade"):
                trade_data.status = narrative_data["trade"]

            return MultiImageAnalysisResponse(
                score=score,
                buyer_name=parsed.get("buyer_name"),
                dealer_name=parsed.get("dealer_name"),
                logo_text=parsed.get("logo_text"),
                email=parsed.get("email"),
                phone_number=parsed.get("phone_number"),
                address=parsed.get("address"),
                state=parsed.get("state"),
                region=parsed.get("region") or "Outside US",
                badge=self._assign_badge(score),
                selling_price=parsed.get("selling_price"),
                vin_number=parsed.get("vin_number"),
                date=parsed.get("date"),
                buyer_message=buyer_msg,
                red_flags=red_flags,
                green_flags=green_flags,
                blue_flags=blue_flags,
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
                normalized_pricing=parsed.get("normalized_pricing") or {},
                apr=parsed.get("apr") or {},
                term=parsed.get("term") or {},
                narrative=narrative_data,
                line_items=parsed.get("line_items", [])
            )
        except Exception as e:
            raise RuntimeError(f"Contract analysis failed: {str(e)}")

