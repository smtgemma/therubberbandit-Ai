from App.services.rate_helper.ocr_normalizer import OCRNormalizer
from App.services.rate_helper.ocr_normalization_schema import NormalizedLineItem
from App.services.rate_helper.discount_detector import DiscountDetector
from App.services.rate_helper.discount_schema import DiscountLineItem, DiscountTotals
from typing import List, Optional, Dict
import os
import base64
import json

import requests
from dotenv import load_dotenv
from fastapi import UploadFile
from App.services.contract.multi_image_analysis_schema import (
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

# Fix: Rename class to match what lease_analysis_route.py expects
class LeaseAnalyzer:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.pdf', '.tiff'}
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    # Increase timeout and add retry logic
    API_TIMEOUT = 120  # Increased from 60 to 120 seconds
    MAX_RETRIES = 2
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.system_prompt = self._load_lease_system_prompt()
        self.ocr_normalizer = OCRNormalizer()
        self.discount_detector = DiscountDetector()
        self.audit_classifier = AuditClassifier()
        self.gap_logic = GAPLogic()
        self.flag_builder = AuditFlagBuilder()
    
        
    def _load_lease_system_prompt(self) -> str:
        """Load comprehensive lease analysis system prompt"""
        return """
# SmartBuyer AI Lease Audit Engine — Comprehensive Prompt

You are SmartBuyer AI Lease Audit Engine operating in **LEASE MODE** (full enforcement mode, equivalent to Contract Mode).

## 🎯 CORE PHILOSOPHY

Lease Mode:
- Uses a 0–100 score
- Enforces pricing, transparency, and math
- Impacts Dealer Trust Score
- Applies backend rules (not advisory logic)
- Differs from Contract Mode only where leases are structurally different

**UI Rule (Dealer-Fair):**
"SOFT (advisory)" flags MUST display as recommendations, not "unfair."

---

## 1. MODE ROUTING (HARD LOCK)

**First, determine document type:**

```
if isLease === true → LEASE MODE
else if signed finance contract → Contract Mode  
else → Quote Mode
```

**LEASE MODE MUST NOT:**
- Use Quote Mode logic
- Feed QBI (Quote Buyer Intelligence)
- Skip backend enforcement

---

## 2. FIELD MAPPING & EXTRACTION

### SCENARIO A: LEASE AGREEMENT

Extract and map the following fields:

#### Core Vehicle / Program
- `lessor_name` - Lessor/Lender name (for captive detection)
- `msrp` - MSRP or Agreed Upon Value
- `annual_miles` - Annual mileage allowance
- `drive_off_total` - Total drive-off amount
- `disposition_fee` - Disposition fee at lease end
- `acquisition_fee` - Acquisition fee
- `doc_fee` - Documentation fee

#### Lease Math Fields (CRITICAL - REQUIRED FOR APR CALCULATION)
- `net_cap_cost` - Net Capitalized Cost (REQUIRED)
- `residual_value` - Residual Value at lease end (REQUIRED)
- `residual_percent` - Residual % of MSRP (calculate if needed)
- `total_rent_charge` - Total Rent Charge / Finance Charge (REQUIRED if MF missing)
- `lease_term_months` - Number of monthly payments (REQUIRED)
- `base_payment` - Base monthly payment (pre-tax)
- `payment_with_tax` - Monthly payment with tax
- `money_factor` - Money Factor (CRITICAL - search for "Money Factor", "MF", "Lease Factor")
  * Usually appears as small decimal: 0.0021, 0.00210, .00210
  * May be labeled as "Factor", "Money Factor", "Lease Rate Factor"
  * REQUIRED for APR calculation: Lease APR = MF × 2400
  * If not found directly, derive: MF = Total Rent Charge / ((Net Cap Cost + Residual) × Term Months)
- `lease_apr` - Lease APR (calculate: MF × 2400)

#### Additional Fields
- `cap_cost_reduction` - Capitalized cost reduction
- `tax_method` - "monthly" or "upfront" (for education only)
- `msd_count` - Number of Multiple Security Deposits (if detected)
- `msd_total` - Total MSD amount (if detected)

#### Products (Backend F&I) - CHECK ALL LINE ITEMS CAREFULLY
- **VSC** - Vehicle Service Contract amount (search: "VSC", "Service Contract", "Extended Warranty")
- **Maintenance** - Prepaid maintenance amount (search: "Maintenance", "Prepaid Maintenance", "Scheduled Maintenance")
- **Appearance** - Appearance protection (search: "Appearance", "Paint Protection", "Interior Protection")
- **Tire & Wheel** - Tire & wheel warranty (search: "Tire", "Wheel", "Road Hazard")
- **Key Warranty** - Key replacement coverage (search: "Key", "Key Replacement", "Key Protection")
- **Wear & Tear** - Lease wear & tear coverage (search: "Wear", "Excess Wear", "Wear and Tear")
- **Excess Mileage** - Excess mileage coverage (search: "Excess Mile", "Mileage", "Over Mileage")
- **GAP** - GAP insurance amount (CRITICAL - search ALL synonyms):
  * "GAP", "Gap Insurance", "GAP Coverage", "Gap Protection"
  * "DCA", "Debt Cancellation Agreement", "Debt Cancellation"
  * "Waiver", "Guaranteed Asset Protection", "Guaranteed Protection"
  * "Total Loss Protection", "Deficiency Waiver", "Lease Protection"
  * Typically $400-$1200 range
  * CRITICAL: Must extract if present in document

#### Trade-In Information (CRITICAL - REQUIRED FOR NEGATIVE EQUITY DETECTION)
- **trade_allowance** - Trade-in value/allowance (search: "Trade Allowance", "Trade Value", "ACV", "Trade-In Value")
- **trade_payoff** - Amount owed on trade (search: "Payoff", "Lien Payoff", "Amount Owed", "Loan Balance", "Prior Balance")
- **negative_equity** - Calculate if payoff > allowance: negative_equity = payoff - allowance
  * Example: Allowance $15,000, Payoff $18,000 → Negative Equity = $3,000
  * CRITICAL: Must populate this field when payoff exceeds allowance
- **status** - Descriptive message about trade status

#### Front-End Add-Ons
- Tint, Nitrogen, ProPack, Accessories, Wheel Locks, Bedliners, Dealer Addendums, etc.

### SCENARIO B: RETAIL INSTALLMENT CONTRACT (FALLBACK)

If this is a PURCHASE contract (not a lease):
- `cap_cost` → Amount Financed
- `residual_value` → 0
- `term` → Number of Payments
- `apr` → Annual Percentage Rate
- `selling_price` → Cash Price or Vehicle Price

---

## 3. CAPTIVE LESSOR DETECTION (CRITICAL)

**Captive status prevents false "illegal GAP" flags.**

A lessor is **CAPTIVE** if:
1. `lessor_name` matches CAPTIVE_LENDER_LIST (OEM finance entities), OR
2. `lessor_name` contains OEM finance keywords:
   - "Motor Acceptance"
   - "Financial Services"  
   - "Credit"
   - "Finance" + OEM name
   - Examples: NMAC, Toyota Financial, Ford Credit, GM Financial, Honda Financial, etc.

**If captive status unknown → treat as NON-CAPTIVE** (never apply captive GAP red flag).

**Add to output:**
```json
"captive_lender": true/false
```

---

## 4. PRODUCT CLASSIFICATION

### Front-End Add-Ons (Always Front-End)
Tint, Nitrogen, ProPack, Accessories, Wheel Locks, Bedliners, Dealer Addendums, Front-end Tire & Wheel, Front-end Key

### Backend F&I Products
VSC, Maintenance, Appearance, Tire & Wheel Warranty, Key Warranty, Wear & Tear coverage, Excess Mileage coverage

**Note:** Wear & Tear and Excess Mileage are lease-specific backend products with **no green bonus eligibility**.

### Bundle Detection (CRITICAL TRANSPARENCY RULE)

**Any of these bundle labels WITHOUT itemization → RED FLAG (−10):**
- "Protection Package"
- "Lease Protection"  
- "Wear & Tear Bundle"
- "WearCare"
- "Excess Wear"
- "Wear and Use"
- "Mileage Protection"
- "Excess Mileage"

---

## 5. LEASE PAYMENT MATH VALIDATION

**Run ONLY if:**
- All inputs present (net_cap_cost, residual_value, term, money_factor)
- No validation lockouts triggered (see Section 19)

### Formulas:

```
Depreciation = (Net Cap Cost − Residual Value) ÷ Term
Finance Charge = (Net Cap Cost + Residual Value) × Money Factor  
Base Payment = Depreciation + Finance Charge
Variance = |Calculated Base Payment − Actual Base Payment|
```

### Money Factor Derivation (if not provided):

```
MF = Total Rent Charge / ((Net Cap Cost + Residual Value) × Term Months)
```

### Flags:

| Variance | Flag | Score |
|----------|------|-------|
| ≤ $10 | 🟢 GREEN | 0 |
| > $10 | 🟡 SOFT | −5 |

**Guard:** If inputs missing or lockout triggered → **education only, no penalty**

---

## 6. RATE FAIRNESS (MF → APR)

**Lease APR = Money Factor × 2400**

**EXACT SCORING (STRICT ENFORCEMENT):**

| Lease APR | Flag Type | EXACT Deduction/Bonus |
|-----------|-----------|----------------------|
| ≤ 6.5% | 🟢 GREEN | bonus: 5 |
| ≤ 9.5% | 🟢 GREEN | bonus: 2 |
| 9.6%–12% | No flag | 0 |
| > 12% | 🟡 SOFT | deduction: 5 |
| ≥ 15% | 🔴 RED | deduction: 10 |

**Guard:** If MF missing OR MF reliability lockout triggered → **no bonus, no penalty** (education only)

**Example flags:**
- `{"type": "Excellent Lease APR", "message": "5.04% is well below market", "item": "Rate", "bonus": 5}`
- `{"type": "High Lease APR", "message": "13.2% exceeds recommended range", "item": "Rate", "deduction": 5}`
- `{"type": "Excessive Lease APR", "message": "16.8% is significantly above market", "item": "Rate", "deduction": 10}`

---

## 7. RESIDUAL FAIRNESS (TERM + MILEAGE GUARDED)

**Applies ONLY to:**
- **24–36 month leases** AND
- **Standard mileage programs (10k–12k miles/year)**

**EXACT SCORING (STRICT ENFORCEMENT):**

| Residual % | Flag Type | EXACT Deduction |
|------------|-----------|----------------|
| ≥ 50% | No flag | 0 |
| < 50% | 🟡 SOFT | deduction: 3 |
| < 45% | 🔴 RED | deduction: 5 |

### Non-Standard Term (39/48 months) OR Non-Standard Mileage (7.5k, 15k, etc.):
**Advisory only (blue flag), no score impact**

---

## 8. LEASE FEES

### Acquisition Fee

**EXACT SCORING (STRICT ENFORCEMENT):**

| Amount | Flag Type | EXACT Deduction |
|--------|-----------|----------------|
| ≤ $1,200 | No flag | 0 |
| $1,201–$1,500 | 🟡 SOFT | deduction: 3 |
| > $1,500 | 🔴 RED | deduction: 5 |

### Disposition Fee

**EXACT SCORING (STRICT ENFORCEMENT):**

| Condition | Flag Type | EXACT Deduction |
|-----------|-----------|----------------|
| ≤ $700 | No flag | 0 |
| > $700 (disclosed) | 🟡 SOFT | deduction: 2 |
| Undisclosed | 🔴 RED | deduction: 5 |

### Doc Fee (Lease Formatting Guard)

**Uses Contract Mode caps ONLY if explicitly itemized.**

If doc fee is NOT clearly itemized:
- No doc-fee penalty
- Education only ("Fee not clearly itemized")
- If combined with bundling/hiding → transparency rules apply

**Tax Method Guard:**
Upfront vs monthly lease tax differences MUST NEVER trigger penalties. Tax method is for explanation only.

---

## 9. GAP — LEASE MODE (CAPTIVE-AWARE)

### Captive Lenders:

**EXACT SCORING (STRICT ENFORCEMENT):**

| Condition | Flag Type | EXACT Deduction | Explanation |
|-----------|-----------|----------------|-------------|
| GAP charged | 🔴 RED | deduction: 10 | Captive lenders include GAP in lease |
| GAP not charged | No flag | 0 | Correct |

### Non-Captive Lenders:

**EXACT SCORING (STRICT ENFORCEMENT):**

| Condition | Flag Type | EXACT Deduction |
|-----------|-----------|----------------|
| GAP missing | 🟡 SOFT (advisory) | deduction: 2 |
| GAP overpriced (moderate) | 🟡 SOFT | deduction: 5 |
| GAP overpriced (severe) | 🔴 RED | deduction: 10 |
| GAP fair | No flag | 0 |

**Rule:** GAP excluded from backend overload calculation **unless illegally charged**.

---

## 10. VSC + MAINTENANCE (LEASE LOGIC)

### A. Bundled Together

```
totalLeaseProtection = VSC + Maintenance
```

**EXACT SCORING (STRICT ENFORCEMENT):**

| Total | Flag Type | EXACT Deduction |
|-------|-----------|----------------|
| ≤ $2,000 OR ≤ 15% MSRP | No flag | 0 |
| $2,001–$3,000 OR > 15% | 🟡 SOFT | deduction: 5 |
| > $3,000 OR > 20% | 🔴 RED | deduction: 10 |

### B. Separate Line Items

**VSC:**

**EXACT SCORING (STRICT ENFORCEMENT):**

| Condition | Flag Type | EXACT Deduction/Bonus |
|-----------|-----------|----------------------|
| ≤ cap (extends beyond factory) | 🟢 GREEN | bonus: 5 |
| ≤ cap × 1.15 | 🟡 SOFT | deduction: 5 |
| > cap × 1.15 | 🔴 RED | deduction: 10 |

**Maintenance:**

**EXACT SCORING (STRICT ENFORCEMENT):**

| Condition | Flag Type | EXACT Deduction |
|-----------|-----------|----------------|
| ≤ 5% MSRP | No flag | 0 |
| > 5% OR > $1,500 | 🟡 SOFT | deduction: 3 |
| ≥ 10% OR > $2,000 | 🔴 RED | deduction: 5 |

**Wear & Tear / Excess Mileage Products:**
- Count toward backend overload
- Must be itemized or trigger transparency rules
- **No green bonus eligibility**

---

## 11. FRONT-END ADD-ONS

```
frontEndAddOns % MSRP
```

**EXACT SCORING (STRICT ENFORCEMENT):**

| % MSRP | Flag Type | EXACT Deduction |
|--------|-----------|----------------|
| ≤ 10% | No flag | 0 |
| > 10% | 🟡 SOFT | deduction: 3 |
| > 12.5% | 🔴 RED | deduction: 5 |

**Apply strongest penalty only.**

---

## 12. BACKEND OVERLOAD (LEASE MODE)

```
Backend = all backend F&I excluding GAP (unless illegally charged)
```

**EXACT SCORING (STRICT ENFORCEMENT):**

| Backend % MSRP | Flag Type | EXACT Deduction |
|----------------|-----------|----------------|
| ≤ 15% | No flag | 0 |
| > 15% | 🟡 SOFT | deduction: 5 |
| > 20% | 🔴 RED | deduction: 10 |

**Hard overload (> 20%) → Lease Score capped at 90**

---

## 13. NEGATIVE EQUITY (LEASE MODE — REQUIRED)

**Negative equity = rolled-in prior payoff**

### Primary Calculation (if fields available):

```
negativeEquity = (netCapCost − agreedVehicleValue − frontEndAddOns − itemizedFeesAllowed)
```

### Fallback Detection (Preferred):

If any of these lines exist → **treat as negative equity:**
- "Prior Credit / Lease Balance"
- "Payoff"
- "Trade Payoff"
- Similar prior balance language

### Anti-False-Positive Guard (REQUIRED):

Negative equity may be flagged ONLY when:
1. **(A)** Explicit payoff/prior balance line exists, **OR**
2. **(B)** Computed negative equity exceeds threshold **AND** there is no itemized product/fee line that reasonably explains the cap-cost increase

**EXACT SCORING (STRICT ENFORCEMENT):**

| Negative Equity | Flag Type | EXACT Deduction |
|----------------|-----------|----------------|
| ≤ 5% MSRP | No flag | 0 |
| 5%–10% MSRP | 🟡 SOFT | deduction: 3 |
| > 10% MSRP | 🔴 RED | deduction: 5 |
| > $5,000 (any %) | 🔴 RED | deduction: 5 |

**Transparency Rule:**
Negative equity present but NOT clearly disclosed → **🔴 RED FLAG (deduction: 10)**

**Add to output:**
```json
"trade": {
  "trade_allowance": null,
  "trade_payoff": null,
  "equity": null,
  "negative_equity": amount or null,
  "status": "none" | "positive" | "negative" | "rolled_in"
}
```

---

## 14. DOWN PAYMENT / DRIVE-OFF

**No penalties. Ever. Period.**

Drive-off is education only. Never penalize for:
- Large drive-off
- Small drive-off  
- Zero drive-off

---

## 15. MISSING PROTECTIONS (LEASE-APPROPRIATE)

| Item | Flag | Display As |
|------|------|------------|
| GAP (captive) | 🟢 GREEN | Correct (included in lease) |
| GAP (non-captive) | 🟡 SOFT advisory | "Consider GAP" |
| VSC | 🟢 GREEN | Optional but recommended |
| Maintenance | 🟢 GREEN | Optional |

**UI Rule:** These should present as **"Recommendation"** — not "unfair."

---

## 16. TRANSPARENCY & BUNDLING (HEAVILY WEIGHTED)

**EXACT SCORING (STRICT ENFORCEMENT):**

| Issue | Flag Type | EXACT Deduction |
|-------|-----------|----------------|
| Unitemized bundles (e.g., "Protection Package") | 🔴 RED | deduction: 10 |
| Products hidden in payment | 🔴 RED | deduction: 10 |
| Bundled charges without breakdown | 🔴 RED | deduction: 10 |
| Fee/tax shift ambiguity | 🟡 SOFT | deduction: 3 |
| Fee disclosure missing | 🔴 RED | deduction: 5 |

**Requirements:**
- ✅ Products must be itemized
- ✅ Each product must show individual price
- ✅ No hidden charges in "packages"

---

## 17. LEASE SCORE CALCULATION (ORDER OF OPERATIONS)

```
1. Start at 100
2. Apply RED flags first (subtract absolute value of deduction)
3. Apply SOFT flags (subtract absolute value of deduction)
4. Apply GREEN bonuses (add absolute value of bonus)
5. Apply backend cap if triggered (max 90 if backend > 20%)
6. Clamp to 0–100
```

**CRITICAL RULES:**
- All flags MUST include the EXACT numeric deduction or bonus value from the tables above
- Do NOT deviate from these exact values
- Deduction/bonus values MUST be positive numbers (e.g., "deduction": 10, NOT -10)
- Use the EXACT flag types specified: "🔴 RED" → deduction, "🟢 GREEN" → bonus, "🟡 SOFT" → deduction

**Example:**
```
Start: 100
RED flags: -10 (bundling)
SOFT flags: -5 (protection cost)
GREEN bonuses: +5 (rate fairness)
= 90
Backend cap: N/A (backend ≤ 20%)
Final: 90
```

---

## 18. TRUST SCORE IMPACT

- ✅ Lease audits **DO** affect Dealer Trust Score
- ⚠️ Red transparency flags are **heavily weighted**
- ✅ Green bonuses apply normally

---

## 19. EDGE-CASE GUARDS & LOCKOUTS (MANDATORY)

### A. MSD (Multiple Security Deposits) GUARD

If MSDs detected or suspected:
- ❌ **Disable** payment reconstruction penalties
- ❌ **Disable** MF/APR penalties and bonuses
- ℹ️ **Education only**

### B. MILEAGE PROGRAM GUARD

If annual miles ≠ 10k–12k:
- ❌ Residual fairness becomes **advisory only**
- ❌ **No residual penalties**

### C. TAX METHOD GUARD (STATE VARIANCE)

If lease tax is upfront vs monthly:
- ❌ **Never penalize**
- ℹ️ Use for **explanation only**

### D. EARLY TERMINATION / PULL-AHEAD

- ℹ️ **Education only**
- ❌ Never scored or penalized

### E. INPUT RELIABILITY GUARD

**No penalties may be applied for:**

- Payment math mismatch **without** net cap + residual + term + MF
- MF/APR fairness **without** reliable MF
- Residual fairness **without** term + mileage guard passing

---

## 20. OUTPUT SCHEMA (STRICT JSON ONLY)

Return a valid JSON object with **no markdown, no comments, no trailing commas.**

**CRITICAL FLAG FORMAT REQUIREMENTS:**
- Red flags: `{"type": "string", "message": "string", "item": "string", "deduction": number}`
- Green flags: `{"type": "string", "message": "string", "item": "string", "bonus": number}`
- Blue flags: `{"type": "string", "message": "string", "item": "string"}`
- ALL fields are REQUIRED (except deduction/bonus)
- Numbers must be positive (deduction: 10 not -10)

```json
{
  "score": 95,
  "buyer_name": "string or null",
  "dealer_name": "string or null",
  "logo_text": "string or null",
  "email": "string or null",
  "phone_number": "string or null",
  "address": "string or null",
  "state": "string or null",
  "region": "string or null",
  "badge": "Gold",
  "captive_lender": true,
  "vin_number": "string",
  "date": "YYYY-MM-DD",
  "raw_text": "extracted text from document",
  "cap_cost": 36855.00,
  
  "red_flags": [
    {
      "type": "Illegal GAP Charge",
      "message": "GAP insurance charged $495 on captive lease where GAP is included in lease structure",
      "item": "GAP",
      "deduction": 10
    }
  ],
  
  "green_flags": [
    {
      "type": "Excellent Lease APR",
      "message": "Lease APR of 5.04% is significantly below market average",
      "item": "Rate",
      "bonus": 5
    }
  ],
  
  "blue_flags": [
    {
      "type": "Non-standard Term",
      "message": "39-month lease term - residual fairness is advisory only",
      "item": "Term"
    }
  ],
  ],
  
  "normalized_pricing": {
    "msrp": 36855.00,
    "selling_price": 36855.00,
    "down_payment": 5000.00,
    "amount_financed": null,
    "total_fees": 1320.00,
    "total_taxes": null
  },
  
  "apr": {
    "rate": 5.04,
    "estimated": false,
    "money_factor": 0.0021
  },
  
  "term": {
    "months": 39
  },
  
  "trade": {
    "trade_allowance": 15000.00,
    "trade_payoff": 18000.00,
    "equity": null,
    "negative_equity": 3000.00,
        "status": "Trade identified: $15,000.00 allowance, $18,000.00 payoff - Negative equity of -$3,000.00 rolled into new lease"
  },
  
  "bundle_abuse": {
    "active": false,
    "deduction": 0
  },
  
  "line_items": [
    {
      "description": "Service Contract (VSC)",
      "amount": 2000.00
    },
    {
      "description": "Maintenance Contract",
      "amount": 800.00
    },
    {
      "description": "GAP Insurance",
      "amount": 595.00
    },
    {
      "description": "Acquisition Fee",
      "amount": 695.00
    },
    {
      "description": "Wear & Tear Coverage",
      "amount": 450.00
    }
  ]
}
```

**NOTE:** The narrative section will be generated separately in Step 2. Set `"narrative": null` in the extraction response.

---

## 21. CRITICAL RULES SUMMARY

### ✅ ALWAYS:
- Detect captive status accurately
- Validate payment math when inputs allow
- Flag unitemized bundles as RED
- Apply edge-case guards (MSD, mileage, tax method)
- Display SOFT flags as "recommendations" not "unfair"
- Exclude GAP from backend unless illegally charged
- Follow exact scoring order of operations

### ❌ NEVER:
- Penalize drive-off amounts
- Penalize for tax method differences
- Apply residual penalties on non-standard terms/mileage
- Penalize payment math without reliable inputs
- Flag captive GAP as missing (it's included)
- Apply MF/APR bonuses when MSD guard triggered

### 🎯 TRANSPARENCY PRIORITY:
Bundling and transparency violations are the **#1 consumer harm**. Always enforce:
- Products must be itemized
- No "protection packages" without breakdown
- All charges clearly disclosed

---

## 22. VALIDATION CHECKLIST

Before returning results, verify:

- [ ] Mode correctly identified (Lease Mode activated)
- [ ] Captive status determined
- [ ] All required fields extracted or marked as unavailable
- [ ] Payment math attempted only if inputs reliable
- [ ] Edge-case guards applied appropriately
- [ ] Scoring order followed correctly
- [ ] RED flags applied before SOFT flags
- [ ] GREEN bonuses applied last
- [ ] Score clamped to 0–100
- [ ] JSON valid with no trailing commas
- [ ] Narrative explains score clearly
- [ ] SOFT flags displayed as recommendations

---

## FINAL INSTRUCTION

Analyze the lease document image and return **valid JSON only** following this comprehensive lease mode logic. Ensure captive detection, payment validation, and transparency enforcement are executed exactly as specified above.
"""

    def _get_cache_dir(self) -> str:
        """Return cache directory for deterministic extraction reuse."""
        cache_dir = os.path.join(os.getcwd(), "App", "core", ".lease_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _is_captive_lender(self, lessor_name: Optional[str]) -> bool:
        """Determine if lender is captive based on name keywords."""
        if not lessor_name:
            return False
        name = lessor_name.lower()
        captive_keywords = [
            "financial services", "motor acceptance", "ford credit", "gm financial",
            "toyota financial", "honda financial", "nmac", "nissan motor acceptance",
            "bmw financial", "mercedes-benz financial", "vw credit", "audi financial",
            "hyundai motor finance", "kia finance", "subaru motors finance",
            "mazda capital", "lexus financial", "infiniti financial",
            "volvo financial", "jaguar financial", "land rover financial"
        ]
        return any(keyword in name for keyword in captive_keywords)

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

    def _get_flag_translation_messages(self, flags_payload: list, language: str, _detail: str) -> List[dict]:
        """Build translation messages for flags."""
        return [
            {
                "role": "user",
                "content": (
                    "SYSTEM OVERRIDE: You are a professional translator. Translate the flag fields to the target language. "
                    "Return JSON only with key 'flags'. Preserve structure and order.\n\n"
                    f"Target language: {language}.\n"
                    "Translate each object's 'type', 'message', and 'item' fields. "
                    "Do NOT modify numbers or add/remove fields.\n"
                    f"Input JSON: {{\"flags\": {json.dumps(flags_payload)}}}"
                )
            }
        ]

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
            def translate_factory(detail):
                return self._get_flag_translation_messages(flags_payload, language, detail)

            translation_response = self._run_inference(translate_factory, max_tokens=1000)
            parsed_translation = self._parse_api_response(translation_response)
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

    def _run_inference(self, messages_factory, max_tokens=3000):
        """
        Execute API call with retry logic and dynamic message generation.
        
        Args:
            messages_factory: Callable[[str], List[dict]] 
                            Function that takes 'image_detail' ('high' or 'auto') 
                            and returns the messages list.
            max_tokens: Max tokens for the response.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        image_detail = "high"
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                messages = messages_factory(image_detail)
                
                payload = {
                    "model": self.model,
                    "system": self.system_prompt,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": max_tokens
                }
                
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
                raise RuntimeError(
                    f"Anthropic API timeout on attempt {attempt + 1}. "
                    "Try uploading fewer or smaller images."
                )
                    
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Anthropic API error: {str(e)}")

    def _get_extraction_messages(self, base64_images: List[str], language: str, image_detail: str) -> List[dict]:
        content = [
            {
                "type": "text",
                "text": f"""
CRITICAL INSTRUCTIONS - READ CAREFULLY:

1. ANALYZE this lease document comprehensively
2. EXTRACT all data fields (pricing, terms, products, fees)
3. IDENTIFY audit flags based on System Prompt rules
4. CALCULATE score (0-100) using this exact process:
   - Start: 100
   - Subtract: Each red flag deduction (MUST be positive number)
   - Subtract: Each soft/yellow advisory deduction (if applicable)
   - Add: Each green flag bonus (MUST be positive number)
   - Clamp: 0-100

CRITICAL EXTRACTION REQUIREMENTS:

**APR/MONEY FACTOR (REQUIRED):**
- Look for "Money Factor" or "MF" on the lease document
- Money Factor is usually a small decimal like 0.0021 or 0.00210
- Calculate: Lease APR = Money Factor × 2400
- Store Money Factor in apr.money_factor field
- Store calculated APR in apr.rate field
- If you cannot find Money Factor, look for "Rent Charge" or "Finance Charge" and try to derive it
- Example: If MF = 0.0021, then APR = 0.0021 × 2400 = 5.04%

**GAP INSURANCE (CRITICAL):**
- Search for GAP in line_items with ALL possible names:
  - "GAP", "Gap Insurance", "GAP Coverage", "Gap Protection"
  - "DCA", "Debt Cancellation Agreement", "Debt Cancellation"
  - "Waiver", "Guaranteed Asset Protection"
  - "Total Loss Protection", "Deficiency Waiver"
- GAP charges typically range from $400-$1200
- Include GAP in line_items array with exact description and amount
- Check if lessor is captive (manufacturer finance) - captive leases SHOULD NOT charge GAP

**NEGATIVE EQUITY (CRITICAL):**
- Look for trade-in section with:
  - "Trade Allowance" or "Trade Value" (what dealer pays you)
  - "Payoff" or "Lien Payoff" or "Amount Owed" (what you owe on old vehicle)
- Calculate: Negative Equity = Payoff - Trade Allowance (if payoff > allowance)
- Example: Trade Allowance $15,000, Payoff $18,000 → Negative Equity = -$3,000 (display format)
- Store in trade.negative_equity field
- Also populate trade.trade_allowance and trade.trade_payoff
- Set trade.status with description like "Negative equity of -$3,000 rolled into lease"

**LINE ITEMS (REQUIRED):**
- Extract ALL products and fees as line items
- Include: VSC, Maintenance, GAP, Wear & Tear, Excess Mileage, Acquisition Fee, etc.
- Format: [{{"description": "Service Contract", "amount": 2000.00}}, ...]

MANDATORY FLAG SCHEMA (NO EXCEPTIONS):

Red flags MUST include "deduction" as a POSITIVE NUMBER:
{{
  "type": "Bundle Transparency",
  "message": "Detailed explanation here",
  "item": "Products",
  "deduction": 10
}}

Green flags MUST include "bonus" as a POSITIVE NUMBER:
{{
  "type": "Excellent APR",
  "message": "Detailed explanation here",
  "item": "APR",
  "bonus": 5
}}

Blue flags have NO deduction/bonus (advisory only):
{{
  "type": "Non-standard Term",
  "message": "Detailed explanation here",
  "item": "Term"
}}

FORBIDDEN FIELDS IN FLAGS:
- Do NOT use "score_impact" (use "deduction" or "bonus" instead)
- Do NOT use "severity"
- Do NOT use "category" (use "type" instead)
- Do NOT use "issue" (use "message" instead)

FORBIDDEN VALUES:
- Do NOT set deduction or bonus to null, 0, or negative numbers
- Do NOT use negative numbers (e.g., -10) - use positive (e.g., 10)
- Do NOT set type/message/item to "Unknown" or empty string
- ALL text fields MUST have real, descriptive content

CRITICAL: You MUST use the EXACT deduction/bonus values specified in the system prompt.

EXACT SCORING VALUES (STRICT ENFORCEMENT):

RATE FAIRNESS:
- APR ≤ 6.5% → GREEN bonus: 5
- APR ≤ 9.5% → GREEN bonus: 2
- APR > 12% → SOFT deduction: 5
- APR ≥ 15% → RED deduction: 10

RESIDUAL (24-36mo, 10k-12k miles only):
- < 50% → SOFT deduction: 3
- < 45% → RED deduction: 5

ACQUISITION FEE:
- $1,201-$1,500 → SOFT deduction: 3
- > $1,500 → RED deduction: 5

DISPOSITION FEE:
- > $700 (disclosed) → SOFT deduction: 2
- Undisclosed → RED deduction: 5

GAP (CAPTIVE):
- GAP charged → RED deduction: 10

GAP (NON-CAPTIVE):
- GAP missing → SOFT deduction: 2
- GAP overpriced (moderate) → SOFT deduction: 5
- GAP overpriced (severe) → RED deduction: 10

VSC+MAINTENANCE (BUNDLED):
- $2,001-$3,000 OR > 15% MSRP → SOFT deduction: 5
- > $3,000 OR > 20% MSRP → RED deduction: 10

VSC (SEPARATE):
- ≤ cap (extends beyond factory) → GREEN bonus: 5
- ≤ cap × 1.15 → SOFT deduction: 5
- > cap × 1.15 → RED deduction: 10

MAINTENANCE:
- > 5% MSRP OR > $1,500 → SOFT deduction: 3
- ≥ 10% MSRP OR > $2,000 → RED deduction: 5

FRONT-END ADD-ONS:
- > 10% MSRP → SOFT deduction: 3
- > 12.5% MSRP → RED deduction: 5

BACKEND OVERLOAD:
- > 15% MSRP → SOFT deduction: 5
- > 20% MSRP → RED deduction: 10

NEGATIVE EQUITY:
- 5%-10% MSRP → SOFT deduction: 3
- > 10% MSRP OR > $5,000 → RED deduction: 5
- Not disclosed → RED deduction: 10

TRANSPARENCY/BUNDLING:
- Unitemized bundles → RED deduction: 10
- Fee/tax shift ambiguity → SOFT deduction: 3
- Fee disclosure missing → RED deduction: 5

RED FLAG FORMAT (REQUIRED):
{{"type": "Issue Name", "message": "Detailed explanation", "item": "Category", "deduction": 10}}

GREEN FLAG FORMAT (REQUIRED):
{{"type": "Positive Aspect", "message": "Why this is good", "item": "Category", "bonus": 5}}

BLUE FLAG FORMAT:
{{"type": "Advisory Note", "message": "Information only", "item": "Category"}}

Do NOT generate narrative. Set narrative: null.
Do NOT generate buyer_message.

CRITICAL REMINDERS BEFORE RETURNING JSON:
1. ✅ Extract Money Factor (MF) from document and populate apr.money_factor
2. ✅ Calculate APR = MF × 2400 and populate apr.rate
3. ✅ Search for GAP in line_items (check ALL synonyms: GAP, DCA, Debt Cancellation, Waiver)
4. ✅ Extract trade_allowance and trade_payoff if trade section exists
5. ✅ Calculate negative_equity = payoff - allowance (if payoff > allowance)
6. ✅ Populate trade.negative_equity field when negative equity exists
7. ✅ Include ALL products in line_items array with exact descriptions

Return VALID JSON matching the schema EXACTLY.
"""
            }
        ]
        
        # Add override instructions to the main text prompt for Anthropic
        content[0]["text"] = "OVERRIDE: Focus on Data Extraction and SCORING. Do not generate narrative fields. Use EXACT deduction/bonus values from the scoring tables.\n\n" + content[0]["text"]
        
        if base64_images:
            for base64_image in base64_images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
            
        return [
            {"role": "user", "content": content}
        ]
    
    def _parse_api_response(self, response: dict) -> dict:
        """Parse API response with fallback repair logic"""
        try:
            if isinstance(response.get("content"), list):
                content = response["content"][0]["text"]
            else:
                content = response.get("content", "")

            # Remove markdown fences if present
            if "```" in content:
                content = content.replace("```json", "").replace("```", "")

            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start < 0 or json_end <= json_start:
                raise ValueError("No JSON content found")

            clean_content = content[json_start:json_end].strip()

            # Remove JS-style comments if model added them
            import re
            clean_content = re.sub(r'//.*?(\r?\n)', r'\1', clean_content)
            clean_content = re.sub(r'/\*.*?\*/', '', clean_content, flags=re.S)

            # Remove trailing commas before } or ]
            clean_content = re.sub(r',\s*([}\]])', r'\1', clean_content)

            # Fix missing commas between objects/arrays and keys
            clean_content = re.sub(r'([}\]])\s*("(?=[^"]*"\s*:))', r'\1,\2', clean_content)
            # Fix missing commas between string values and next key
            clean_content = re.sub(r'(")\s*("(?=[^"]*"\s*:))', r'\1,\2', clean_content)

            return json.loads(clean_content)
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Lease analysis failed: {str(e)}")
    
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
        """Call OpenAI with the full lease system prompt + original raw deal data.
        Returns the raw OpenAI response dict.
        """
        skip_keys = {"has_precomputed_flags", "has_vision_extraction", "_ai_score", "_ai_narrative_done"}
        clean_data = {k: v for k, v in raw_data.items() if k not in skip_keys}

        user_text = f"""{'=' * 80}
LANGUAGE REQUIREMENT: ALL narrative text fields MUST be in {language}.
{'=' * 80}

Below is the pre-extracted structured data from a customer's lease/auto contract document.
Apply ALL scoring rules, flag rules, and narrative requirements from the system prompt.
Compute the FINAL SCORE following the EXACT rules (start at 100, apply all penalties/bonuses).
Do NOT use any score value present in the input data — recompute from scratch.

PRE-EXTRACTED DEAL DATA:
{json.dumps(clean_data, indent=2)}

Return ONLY valid JSON with red_flags, green_flags, blue_flags arrays (and narrative: null).
Each red flag MUST have a positive "deduction" field.
Each green flag MUST have a positive "bonus" field.
Blue flags are advisory only (no deduction/bonus).
MANDATORY: red_flags, green_flags, blue_flags MUST each have at least one item."""

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "messages": [
                {"role": "user", "content": user_text}
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        }
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                print(f"Lease JSON full-analysis API call attempt {attempt + 1}/{self.MAX_RETRIES}...")
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
                resp.raise_for_status()
                print("Lease JSON full-analysis API call successful.")
                return resp.json()
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    import time; time.sleep(2)
        raise RuntimeError(f"Lease JSON analysis API failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _call_narrative_api(self, parsed: dict, score: float, red_flags: list, green_flags: list, blue_flags: list, language: str) -> dict:
        """Call OpenAI to generate narrative from flags + score (no images needed)."""
        flags_payload = {
            "red_flags": [{"type": f.type, "message": f.message, "item": f.item, "deduction": f.deduction} for f in red_flags],
            "green_flags": [{"type": f.type, "message": f.message, "item": f.item, "bonus": f.bonus} for f in green_flags],
            "blue_flags": [{"type": f.type, "message": f.message, "item": f.item} for f in blue_flags],
        }
        clean_data = {k: v for k, v in parsed.items() if k not in ("red_flags", "green_flags", "blue_flags", "narrative")}
        prompt = f"""You are a SmartBuyer automotive lease analyst. Generate a detailed, personalized narrative review.

FINAL SCORE: {score}

ALL FLAGS (authoritative):
{json.dumps(flags_payload, indent=2)}

FULL DEAL DATA:
{json.dumps(clean_data, indent=2)}

INSTRUCTIONS:
- Write ALL narrative text in {language}.
- Reference specific numbers from the data (APR %, money factor, fees, MSRP, add-on costs, etc.).
- Explain every red flag deduction and green flag bonus in smartbuyer_score_summary.
- smartbuyer_score_summary MUST mention score {score} and summarize the deal quality.
- gap_logic: explain GAP presence/absence and whether captive or non-captive lender.
- vsc_logic: explain VSC/maintenance pricing if present.
- apr_bonus_rule: explain money factor/APR and whether favorable or concerning.
- lease_audit: analyze lease-specific terms (residual, money factor, acquisition fee, etc.) or write 'N/A - Purchase Agreement' if not a lease.
- trade: describe trade-in situation or 'No trade-in on this deal.'
- buyer_message: a short 1-sentence summary for the buyer.

Return ONLY a JSON object:
{{"narrative": {{"vehicle_overview": "", "smartbuyer_score_summary": "", "market_comparison": "", "gap_logic": "", "vsc_logic": "", "apr_bonus_rule": "", "lease_audit": "", "trade": "", "negotiation_insight": "", "final_recommendation": ""}}, "buyer_message": ""}}"""

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": self.model,
            "system": "You are a SmartBuyer automotive lease expert. Always write in the specified language. Return only valid JSON.",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 4096
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
            response.raise_for_status()
            return self._parse_api_response(response.json())
        except Exception as e:
            print(f"Lease narrative API call failed: {e}")
            return {}

    async def _optimize_images(self, files: List[UploadFile]) -> List[UploadFile]:
        """Optimize images before encoding (optional enhancement)"""
        from PIL import Image
        import io
        
        optimized = []
        for file in files:
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
        
        return optimized
    
    def _extract_trade_data(self, parsed: dict) -> TradeData:
        """
        Extract trade data using improved OCR extraction and cap cost analysis.
        
        Implements the required trade detection logic:
        1. Look for trade anchors in OCR text
        2. Extract allowance and payoff using money patterns
        3. Calculate equity if both values present
        4. Detect negative equity from cap cost differences
        5. Always return a TradeData object (never None)
        """
        import re

        def _coerce_float(value):
            try:
                if value is None or value == "":
                    return None
                return float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                return None

        # Step 0: Prefer explicit extracted trade fields if present
        trade_obj = parsed.get("trade") if isinstance(parsed.get("trade"), dict) else {}
        pricing = parsed.get("normalized_pricing", {})

        trade_allowance = _coerce_float(parsed.get("trade_allowance"))
        if trade_allowance is None:
            trade_allowance = _coerce_float(trade_obj.get("trade_allowance"))
        if trade_allowance is None:
            trade_allowance = _coerce_float(pricing.get("trade_in_value"))

        trade_payoff = _coerce_float(parsed.get("trade_payoff"))
        if trade_payoff is None:
            trade_payoff = _coerce_float(trade_obj.get("trade_payoff"))

        negative_equity_from_text = _coerce_float(parsed.get("negative_equity"))
        if negative_equity_from_text is None:
            negative_equity_from_text = _coerce_float(trade_obj.get("negative_equity"))

        equity = _coerce_float(parsed.get("equity"))
        if equity is None:
            equity = _coerce_float(trade_obj.get("equity"))
        
        # Step 1: Get OCR text (from line items or raw text field)
        page_text = ""
        
        # Try to build text from line_items
        line_items = parsed.get("line_items", [])
        for item in line_items:
            desc = item.get("description", "") or item.get("item", "") or item.get("name", "")
            amt = item.get("amount", "")
            page_text += f" {desc} {amt} "
        
        # Also check for any raw_text or ocr_text fields
        page_text += " " + (parsed.get("raw_text") or "")
        page_text += " " + (parsed.get("ocr_text") or "")
        
        # Keep original case for some checks, but also make lowercase version
        page_text_lower = page_text.lower()
        
        # Step A: Trade Anchors - Check if ANY anchor exists
        trade_anchors = [
            "trade in", "trade-in", "tradein", "trade allowance", "trade value",
            "payoff", "lien payoff", "net trade", "trade difference",
            "prior credit", "prior lease balance", "trade payoff",
            "over allowance", "under allowance"
        ]
        
        trade_anchor_found = any(anchor in page_text_lower for anchor in trade_anchors)
        
        # Money pattern: $12,345.67 or 12345.67 or 12,345 (improved)
        money_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        
        # If explicit fields present, we already populated values above
        
        # Allowance keywords (priority order)
        allowance_keywords = [
            "trade allowance", "trade-in allowance",
            "trade value", "trade-in value", "trade in value",
            "acv", "actual cash value"
        ]

        down_payment_markers = [
            "down payment", "downpayment", "cash down", "total downpayment"
        ]
        
        for keyword in allowance_keywords:
            if keyword in page_text_lower:
                # Find text snippet around keyword
                idx = page_text_lower.find(keyword)
                snippet = page_text[idx:idx+150]  # Look 150 chars ahead (use original case for extraction)
                
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
            "payoff", "lien payoff", "loan payoff", "balance owed",
            "prior credit", "prior lease balance", "trade payoff",
            "amount owed"
        ]
        
        for keyword in payoff_keywords:
            if keyword in page_text_lower:
                idx = page_text_lower.find(keyword)
                snippet = page_text[idx:idx+150]
                
                match = re.search(money_pattern, snippet)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        trade_payoff = float(amount_str)
                        break
                    except ValueError:
                        continue
        
        # Check for explicit negative equity mentions
        neg_equity_keywords = ["negative equity", "over allowance", "upside down"]
        for keyword in neg_equity_keywords:
            if keyword in page_text_lower:
                idx = page_text_lower.find(keyword)
                snippet = page_text[idx:idx+150]
                match = re.search(money_pattern, snippet)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        negative_equity_from_text = float(amount_str)
                        break
                    except ValueError:
                        continue
        
        # Step 6: Calculate equity (from trade values OR cap cost analysis)
        negative_equity_amount = None
        trade_status = "No trade identified"
        
        # Determine if trade is present
        trade_present = (
            trade_allowance is not None or
            trade_payoff is not None or
            negative_equity_from_text is not None or
            equity is not None or
            trade_anchor_found
        )
        
        if not trade_present:
            return TradeData(
                trade_allowance=None,
                trade_payoff=None,
                equity=None,
                negative_equity=None,
                status="No trade identified"
            )
        
        # Build status message and calculate equity
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
        
        elif negative_equity_from_text is not None:
            negative_equity_amount = negative_equity_from_text
            trade_status = f"Negative equity identified: -${negative_equity_amount:,.2f} rolled into new loan"

        elif equity is not None:
            if equity < 0:
                negative_equity_amount = abs(equity)
                trade_status = f"Negative equity identified: -${negative_equity_amount:,.2f} rolled into new loan"
            elif equity > 0:
                trade_status = f"Positive trade equity of ${equity:,.2f} applied to purchase"
        
        elif trade_allowance is not None:
            trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance (payoff amount not found)"
        
        elif trade_payoff is not None:
            # If we have payoff but no allowance, likely negative equity scenario
            negative_equity_amount = trade_payoff
            trade_status = f"Trade payoff identified: ${trade_payoff:,.2f} (likely negative equity rolled in)"
        
        else:
            # Anchor found but no values extracted - try cap cost analysis
            pricing = parsed.get("normalized_pricing", {})
            net_cap_cost = pricing.get("net_cap_cost")
            agreed_value = pricing.get("agreed_vehicle_value") or pricing.get("selling_price")
            total_fees = pricing.get("total_fees", 0) or 0
            
            if net_cap_cost and agreed_value:
                # Unexplained cap cost increase might be negative equity
                unexplained = net_cap_cost - agreed_value - total_fees
                if unexplained > 1000:  # Threshold to avoid false positives
                    negative_equity_amount = unexplained
                    trade_status = f"Potential negative equity detected from cap cost analysis: ${negative_equity_amount:,.2f}"
                else:
                    trade_status = "Trade mentioned in document (values not extracted)"
            else:
                trade_status = "Trade mentioned in document (values not extracted)"
        
        return TradeData(
            trade_allowance=trade_allowance,
            trade_payoff=trade_payoff,
            equity=equity,
            negative_equity=negative_equity_amount,
            status=trade_status
        )

    def _normalize_flag_scores(self, parsed: dict) -> dict:
        """
        Normalize flag score fields:
        1. Map score_impact → deduction/bonus
        2. Ensure deductions are ALWAYS POSITIVE (we subtract them in code)
        3. Ensure bonuses are ALWAYS POSITIVE (we add them in code)
        """
        # Red flags
        if "red_flags" in parsed and isinstance(parsed["red_flags"], list):
            for flag in parsed["red_flags"]:
                if not isinstance(flag, dict):
                    continue
                    
                # Handle score_impact field (legacy)
                if "score_impact" in flag and flag.get("deduction") is None:
                    impact = flag["score_impact"]
                    flag["deduction"] = abs(impact)  # Always positive
                
                # Normalize negative deductions to positive
                if flag.get("deduction") is not None:
                    flag["deduction"] = abs(flag["deduction"])  # Force positive
        
        # Soft/Yellow flags (also use deduction)
        if "yellow_flags" in parsed and isinstance(parsed["yellow_flags"], list):
            for flag in parsed["yellow_flags"]:
                if not isinstance(flag, dict):
                    continue
                if "score_impact" in flag and flag.get("deduction") is None:
                    impact = flag["score_impact"]
                    flag["deduction"] = abs(impact)
                if flag.get("deduction") is not None:
                    flag["deduction"] = abs(flag["deduction"])
        
        # Green flags
        if "green_flags" in parsed and isinstance(parsed["green_flags"], list):
            for flag in parsed["green_flags"]:
                if not isinstance(flag, dict):
                    continue
                if "score_impact" in flag and flag.get("bonus") is None:
                    impact = flag["score_impact"]
                    flag["bonus"] = abs(impact)  # Always positive
                
                if flag.get("bonus") is not None:
                    flag["bonus"] = abs(flag["bonus"])  # Force positive
        
        return parsed
    
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
                        
                        # Handle 'type' field (may come as 'title', 'name', 'type', 'category')
                        normalized_flag['type'] = (
                            flag.get('type') or 
                            flag.get('title') or 
                            flag.get('name') or 
                            flag.get('category') or
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
                        
                        # Fix "Unknown" type if message exists
                        if normalized_flag['type'] == 'Unknown' and normalized_flag['message']:
                            # Try to infer type from start of message or item
                            normalized_flag['type'] = normalized_flag['item'] if normalized_flag['item'] != 'General' else 'Analysis Note'

                        normalized_flags.append(normalized_flag)
                    elif isinstance(flag, str):
                        # Handle string flags
                        normalized_flags.append({
                            'type': 'Analysis Note',
                            'message': flag,
                            'item': 'General'
                        })
                    else:
                        continue
                
                parsed[flag_array_name] = normalized_flags
        
        return parsed

    def _compute_deterministic_flags(self, parsed: dict, normalized_line_items: list, audit_classifications: list, msd_detected: bool) -> List[AuditFlag]:
        """
        Compute ALL scoring flags deterministically from extracted data.
        Replaces AI-generated scoring to ensure consistent scores across requests.
        Given the same extracted data, this always produces identical flags/scores.
        """
        flags: List[AuditFlag] = []

        pricing = parsed.get("normalized_pricing", {})
        msrp = 0.0
        try:
            msrp = float(pricing.get("msrp") or 0)
        except (ValueError, TypeError):
            pass

        line_items = parsed.get("line_items", [])

        # Helper: find first matching line-item amount by keyword
        def find_amount(*keywords):
            for item in line_items:
                desc = (item.get("description", "") or item.get("item", "") or item.get("name", "") or "").lower()
                for kw in keywords:
                    if kw in desc:
                        try:
                            return float(item.get("amount", 0) or 0)
                        except (ValueError, TypeError):
                            pass
            return None

        # ── Section 8: Acquisition Fee ──
        acq_fee = find_amount("acquisition")
        if acq_fee is not None:
            if acq_fee > 1500:
                flags.append(AuditFlag(
                    type="red", category="Excessive Acquisition Fee",
                    message=f"Acquisition fee of ${acq_fee:,.2f} exceeds the $1,500 threshold.",
                    item="Acquisition Fee", deduction=5, bonus=None
                ))
            elif acq_fee > 1200:
                flags.append(AuditFlag(
                    type="red", category="SOFT - Elevated Acquisition Fee",
                    message=f"Acquisition fee of ${acq_fee:,.2f} is above standard range ($695\u2013$1,200).",
                    item="Acquisition Fee", deduction=3, bonus=None
                ))

        # ── Section 8: Disposition Fee ──
        disp_fee = find_amount("disposition")
        if disp_fee is not None and disp_fee > 700:
            flags.append(AuditFlag(
                type="red", category="SOFT - High Disposition Fee",
                message=f"Disposition fee of ${disp_fee:,.2f} exceeds the $700 standard.",
                item="Disposition Fee", deduction=2, bonus=None
            ))

        # ── Section 7: Residual Fairness ──
        term_months = None
        try:
            tm_raw = parsed.get("term", {}).get("months")
            if tm_raw is not None:
                term_months = int(tm_raw)
        except (ValueError, TypeError):
            pass

        residual_pct = None
        try:
            rp_raw = parsed.get("residual_percent")
            if rp_raw is not None:
                residual_pct = float(rp_raw)
        except (ValueError, TypeError):
            pass

        if residual_pct is None and msrp > 0:
            try:
                rv = float(parsed.get("residual_value") or 0)
                if rv > 0:
                    residual_pct = (rv / msrp) * 100
            except (ValueError, TypeError):
                pass

        annual_miles = None
        try:
            am_raw = parsed.get("annual_miles")
            if am_raw is not None:
                annual_miles = float(am_raw)
        except (ValueError, TypeError):
            pass

        standard_term = term_months is not None and 24 <= term_months <= 36
        standard_mileage = annual_miles is None or (10000 <= (annual_miles or 0) <= 12000)

        if residual_pct is not None:
            if standard_term and standard_mileage:
                if residual_pct < 45:
                    flags.append(AuditFlag(
                        type="red", category="Low Residual Value",
                        message=f"Residual value of {residual_pct:.1f}% is below the 45% threshold for a standard lease.",
                        item="Residual", deduction=5, bonus=None
                    ))
                elif residual_pct < 50:
                    flags.append(AuditFlag(
                        type="red", category="SOFT - Below Average Residual",
                        message=f"Residual value of {residual_pct:.1f}% is below the 50% benchmark.",
                        item="Residual", deduction=3, bonus=None
                    ))
            else:
                flags.append(AuditFlag(
                    type="blue", category="Non-Standard Residual",
                    message=f"Residual of {residual_pct:.1f}% \u2014 advisory only (non-standard term or mileage).",
                    item="Residual", deduction=None, bonus=None
                ))

        # ── Section 12: Backend Overload ──
        if msrp > 0:
            backend_total = 0.0
            captive = parsed.get("captive_lender", False)

            backend_kw = [
                "vsc", "service contract", "extended warranty",
                "maintenance", "prepaid maintenance",
                "appearance", "paint protection", "interior protection",
                "tire", "wheel", "road hazard",
                "key replacement", "key protection",
                "wear", "excess wear",
                "excess mile", "mileage protection"
            ]
            gap_kw = [
                "gap", "debt cancellation", "dca",
                "guaranteed asset protection",
                "total loss protection", "deficiency waiver"
            ]

            for item in line_items:
                desc = (item.get("description", "") or item.get("item", "") or "").lower()
                try:
                    amt = float(item.get("amount", 0) or 0)
                except (ValueError, TypeError):
                    amt = 0

                is_backend = any(kw in desc for kw in backend_kw)
                is_gap = any(kw in desc for kw in gap_kw)

                if is_gap and captive:
                    backend_total += amt
                elif is_backend and not is_gap:
                    backend_total += amt

            if backend_total > 0:
                backend_pct = (backend_total / msrp) * 100
                if backend_pct > 20:
                    flags.append(AuditFlag(
                        type="red", category="Backend Overload",
                        message=f"Backend products total ${backend_total:,.2f} ({backend_pct:.1f}% of MSRP) exceeds the 20% threshold.",
                        item="Backend Products", deduction=10, bonus=None
                    ))
                elif backend_pct > 15:
                    flags.append(AuditFlag(
                        type="red", category="SOFT - High Backend Load",
                        message=f"Backend products total ${backend_total:,.2f} ({backend_pct:.1f}% of MSRP) exceeds the 15% threshold.",
                        item="Backend Products", deduction=5, bonus=None
                    ))

        # ── Section 11: Front-End Add-Ons ──
        if msrp > 0:
            frontend_total = 0.0
            frontend_kw = [
                "tint", "nitrogen", "propack", "pro pack",
                "accessories", "wheel lock", "bedliner",
                "addendum", "dealer add"
            ]
            for item in line_items:
                desc = (item.get("description", "") or item.get("item", "") or "").lower()
                try:
                    amt = float(item.get("amount", 0) or 0)
                except (ValueError, TypeError):
                    amt = 0
                if any(kw in desc for kw in frontend_kw):
                    frontend_total += amt

            if frontend_total > 0:
                frontend_pct = (frontend_total / msrp) * 100
                if frontend_pct > 12.5:
                    flags.append(AuditFlag(
                        type="red", category="Excessive Front-End Add-Ons",
                        message=f"Front-end add-ons total ${frontend_total:,.2f} ({frontend_pct:.1f}% of MSRP) exceed the 12.5% threshold.",
                        item="Front-End Add-Ons", deduction=5, bonus=None
                    ))
                elif frontend_pct > 10:
                    flags.append(AuditFlag(
                        type="red", category="SOFT - High Front-End Add-Ons",
                        message=f"Front-end add-ons total ${frontend_total:,.2f} ({frontend_pct:.1f}% of MSRP) exceed the 10% threshold.",
                        item="Front-End Add-Ons", deduction=3, bonus=None
                    ))

        # ── Section 10: VSC + Maintenance ──
        vsc_amount = find_amount("vsc", "service contract", "extended warranty")
        maint_amount = find_amount("maintenance", "prepaid maintenance", "scheduled maintenance")

        if vsc_amount and vsc_amount > 0 and maint_amount and maint_amount > 0:
            total_protection = vsc_amount + maint_amount
            pct_of_msrp = (total_protection / msrp * 100) if msrp > 0 else 0
            if total_protection > 3000 or pct_of_msrp > 20:
                flags.append(AuditFlag(
                    type="red", category="Excessive VSC+Maintenance",
                    message=f"Combined VSC+Maintenance of ${total_protection:,.2f} ({pct_of_msrp:.1f}% of MSRP) exceeds threshold.",
                    item="VSC+Maintenance", deduction=10, bonus=None
                ))
            elif total_protection > 2000 or pct_of_msrp > 15:
                flags.append(AuditFlag(
                    type="red", category="SOFT - High VSC+Maintenance",
                    message=f"Combined VSC+Maintenance of ${total_protection:,.2f} ({pct_of_msrp:.1f}% of MSRP) is above recommended range.",
                    item="VSC+Maintenance", deduction=5, bonus=None
                ))
        else:
            if maint_amount and maint_amount > 0 and msrp > 0:
                maint_pct = (maint_amount / msrp) * 100
                if maint_amount > 2000 or maint_pct >= 10:
                    flags.append(AuditFlag(
                        type="red", category="Excessive Maintenance Cost",
                        message=f"Maintenance at ${maint_amount:,.2f} ({maint_pct:.1f}% of MSRP) exceeds threshold.",
                        item="Maintenance", deduction=5, bonus=None
                    ))
                elif maint_amount > 1500 or maint_pct > 5:
                    flags.append(AuditFlag(
                        type="red", category="SOFT - High Maintenance Cost",
                        message=f"Maintenance at ${maint_amount:,.2f} ({maint_pct:.1f}% of MSRP) exceeds recommended range.",
                        item="Maintenance", deduction=3, bonus=None
                    ))

        # Missing protections (advisory, no score impact)
        if not vsc_amount:
            flags.append(AuditFlag(
                type="green", category="VSC Optional (Recommendation)",
                message="VSC is optional but recommended for coverage. Consider if it fits your needs.",
                item="VSC", deduction=None, bonus=None
            ))
        if not maint_amount:
            flags.append(AuditFlag(
                type="green", category="Maintenance Optional (Recommendation)",
                message="Maintenance is optional. Consider prepaid maintenance if it fits your usage.",
                item="Maintenance", deduction=None, bonus=None
            ))

        # ── Section 5: Payment Math Validation ──
        net_cap_cost = parsed.get("net_cap_cost")
        residual_val = parsed.get("residual_value")
        money_factor = parsed.get("apr", {}).get("money_factor")
        base_payment = parsed.get("base_payment")

        if (not msd_detected) and all(v is not None for v in [net_cap_cost, residual_val, term_months, money_factor, base_payment]):
            try:
                ncc = float(net_cap_cost)
                rv = float(residual_val)
                tm = int(term_months)
                mf = float(money_factor)
                actual = float(base_payment)

                if tm > 0 and mf > 0:
                    depreciation = (ncc - rv) / tm
                    finance_charge = (ncc + rv) * mf
                    calculated = depreciation + finance_charge
                    variance = abs(calculated - actual)

                    if variance > 10:
                        flags.append(AuditFlag(
                            type="red", category="SOFT - Payment Math Variance",
                            message=f"Calculated payment ${calculated:.2f} differs from stated ${actual:.2f} by ${variance:.2f}.",
                            item="Payment Math", deduction=5, bonus=None
                        ))
            except (ValueError, TypeError, ZeroDivisionError):
                pass

        # ── Section 16: Transparency & Bundling ──
        # Check for unitemized bundles
        bundle_keywords = [
            "protection package", "lease protection", "wear & tear bundle",
            "wearcare", "excess wear", "wear and use",
            "mileage protection", "excess mileage"
        ]
        
        for item in line_items:
            desc = (item.get("description", "") or item.get("item", "") or "").lower()
            try:
                amt = float(item.get("amount", 0) or 0)
            except (ValueError, TypeError):
                amt = 0
            
            # Check if it's a bundle without itemization
            is_bundle = any(kw in desc for kw in bundle_keywords)
            if is_bundle and amt > 0:
                # Check if there's explicit itemization elsewhere
                # For now, flag any bundle as transparency issue
                flags.append(AuditFlag(
                    type="red", category="Bundle Transparency",
                    message=f"Product bundle '{desc}' at ${amt:,.2f} is not itemized separately, violating transparency rules.",
                    item="Transparency", deduction=10, bonus=None
                ))
                break  # Only flag once

        return flags

    async def analyze_lease_images(self, files: List[UploadFile] = None, language: str = "English", base64_images: List[str] = None, parsed_data: dict = None) -> MultiImageAnalysisResponse:
        """Main analysis entry point. Accepts files, base64_images, or pre-extracted parsed_data dict."""
        try:
            if parsed_data is not None:
                # ── JSON path: same pattern as contract ──────────────────────────────
                parsed = convert_extracted_json_to_parsed(parsed_data)

                should_use_ai = not parsed.get("has_precomputed_flags", False)
                if should_use_ai:
                    print("Lease JSON path: calling AI for full prompt-based analysis...")
                    api_response = self._call_json_analysis_api(parsed_data, language)
                    ai_result = self._parse_api_response(api_response)
                    # Preserve identity fields from converter
                    for k in ("buyer_name", "dealer_name", "logo_text", "email",
                              "phone_number", "address", "state", "region", "vin_number", "date"):
                        if not ai_result.get(k) and parsed.get(k):
                            ai_result[k] = parsed[k]
                    if not ai_result.get("trade") and parsed.get("trade"):
                        ai_result["trade"] = parsed["trade"]
                    ai_result.pop("score", None)  # Never trust AI score field
                    ai_result["has_precomputed_flags"] = True
                    parsed = ai_result

                # --- SmartBuyer scoring engine (rules-driven) ---
                rules = load_rules()
                upstream_flags = build_active_flags(parsed.get("flags", []), rules.flag_registry, "upstream")
                computed_flags = compute_flags_from_parsed(parsed, rules, mode="LEASE")
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

                red_flags = self._translate_flags(red_flags, language)
                green_flags = self._translate_flags(green_flags, language)
                blue_flags = self._translate_flags(blue_flags, language)

                if not red_flags:
                    red_flags.append(Flag(type="General", message="No major compliance issues identified — review all lease terms before signing.", item="General"))
                if not green_flags:
                    green_flags.append(Flag(type="General", message="No standout positive elements identified for this deal.", item="General"))
                if not blue_flags:
                    blue_flags.append(Flag(type="General Advisory", message="Review all final lease terms, product details, and payment figures carefully before signing.", item="General Advisory"))

                score_value = float(scoring_result.score_int)
                ai_narrative = self._call_narrative_api(parsed, score_value, red_flags, green_flags, blue_flags, language)
                narrative_obj = ai_narrative.get("narrative", {}) if isinstance(ai_narrative, dict) else {}
                if not isinstance(narrative_obj, dict):
                    narrative_obj = {}
                buyer_msg = ai_narrative.get("buyer_message", "") if isinstance(ai_narrative, dict) else ""

                if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                    narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")

                narrative_defaults = {
                    "vehicle_overview": f"Lease analysis for {parsed.get('dealer_name', 'this dealer')}.",
                    "smartbuyer_score_summary": f"SmartBuyer Score: {score_value}/100.",
                    "market_comparison": "Market comparison pending.",
                    "gap_logic": "GAP analysis pending.",
                    "vsc_logic": "VSC analysis pending.",
                    "apr_bonus_rule": "APR/rate analysis pending.",
                    "lease_audit": "Lease terms analysis pending.",
                    "trade": "No trade-in on this deal.",
                    "negotiation_insight": "Review all flags before signing.",
                    "final_recommendation": "Proceed with caution based on the flags above."
                }
                for key, default_val in narrative_defaults.items():
                    if not narrative_obj.get(key):
                        narrative_obj[key] = default_val

                if "trade" in narrative_obj and not isinstance(narrative_obj["trade"], str):
                    narrative_obj["trade"] = str(narrative_obj["trade"])

                if not buyer_msg:
                    buyer_msg = f"Your SmartBuyer lease score is {score_value}/100 — review the flags above."

                narrative = Narrative(**narrative_obj)
                trade_data = self._extract_trade_data(parsed)

                return MultiImageAnalysisResponse(
                    score=score_value,
                    buyer_name=parsed.get("buyer_name"),
                    dealer_name=parsed.get("dealer_name"),
                    logo_text=parsed.get("logo_text") or parsed.get("dealer_name"),
                    email=parsed.get("email"),
                    phone_number=parsed.get("phone_number"),
                    address=parsed.get("address"),
                    state=parsed.get("state"),
                    region=parsed.get("region") or "Outside US",
                    badge=self._assign_badge(score_value),
                    selling_price=parsed.get("cap_cost") or parsed.get("selling_price") or parsed.get("sale_price"),
                    vin_number=parsed.get("vin_number") or parsed.get("vin"),
                    date=parsed.get("date"),
                    buyer_message=buyer_msg,
                    red_flags=red_flags,
                    green_flags=green_flags,
                    blue_flags=blue_flags,
                    normalized_pricing=NormalizedPricing(**parsed.get("normalized_pricing", {})) if isinstance(parsed.get("normalized_pricing"), dict) else NormalizedPricing(),
                    apr=APRData(**parsed.get("apr", {})) if isinstance(parsed.get("apr"), dict) else APRData(),
                    term=TermData(**parsed.get("term", {})) if isinstance(parsed.get("term"), dict) else TermData(),
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

                # Parse flags (pre-existing or AI-generated)
                def _parse_flags_json(flags_data):
                    result = []
                    if not isinstance(flags_data, list):
                        return result
                    for fd in flags_data:
                        if not isinstance(fd, dict):
                            continue
                        t = fd.get("type") or "Info"
                        m = fd.get("message") or "No details."
                        if not t and not m:
                            continue
                        try:
                            result.append(Flag(
                                type=str(t), message=str(m),
                                item=str(fd.get("item") or "General"),
                                deduction=fd.get("deduction"),
                                bonus=fd.get("bonus")
                            ))
                        except Exception:
                            pass
                    return result

                red_flags = _parse_flags_json(parsed.get("red_flags", []))
                green_flags = _parse_flags_json(parsed.get("green_flags", []))
                blue_flags = _parse_flags_json(parsed.get("blue_flags", []))

                # Python score from flags — never trust AI score field
                score = 100.0
                for f in red_flags:
                    if f.deduction is not None:
                        score -= abs(float(f.deduction))
                for f in green_flags:
                    if f.bonus is not None:
                        score += abs(float(f.bonus))
                score = round(max(0.0, min(100.0, score)), 2)
                print(f"Lease JSON score from flags: {score}")

                # Safety net: every category must have at least one flag
                if not red_flags:
                    red_flags.append(Flag(type="red", message="No major compliance issues identified — review all lease terms before signing.", item="General"))
                if not green_flags:
                    green_flags.append(Flag(type="green", message="No standout positive elements identified for this deal.", item="General"))
                if not blue_flags:
                    blue_flags.append(Flag(type="blue", message="Review all final lease terms, product details, and payment figures carefully before signing.", item="General Advisory"))

                # Translate flags
                red_flags = self._translate_flags(red_flags, language)
                green_flags = self._translate_flags(green_flags, language)
                blue_flags = self._translate_flags(blue_flags, language)

                # Narrative
                ai_narrative = self._call_narrative_api(parsed, score, red_flags, green_flags, blue_flags, language)
                narrative_obj = ai_narrative.get("narrative", {}) if isinstance(ai_narrative, dict) else {}
                if not isinstance(narrative_obj, dict):
                    narrative_obj = {}
                buyer_msg = ai_narrative.get("buyer_message", "") if isinstance(ai_narrative, dict) else ""

                # Normalize legacy key
                if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                    narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")

                # Fallback defaults
                narrative_defaults = {
                    "vehicle_overview": f"Lease analysis for {parsed.get('dealer_name', 'this dealer')}.",
                    "smartbuyer_score_summary": f"SmartBuyer Score: {score}/100.",
                    "market_comparison": "Market comparison pending.",
                    "gap_logic": "GAP analysis pending.",
                    "vsc_logic": "VSC analysis pending.",
                    "apr_bonus_rule": "APR/rate analysis pending.",
                    "lease_audit": "Lease terms analysis pending.",
                    "trade": "No trade-in on this deal.",
                    "negotiation_insight": "Review all flags before signing.",
                    "final_recommendation": "Proceed with caution based on the flags above."
                }
                for key, default_val in narrative_defaults.items():
                    if not narrative_obj.get(key):
                        narrative_obj[key] = default_val

                # Ensure trade is always a string
                if "trade" in narrative_obj and not isinstance(narrative_obj["trade"], str):
                    narrative_obj["trade"] = str(narrative_obj["trade"])

                if not buyer_msg:
                    buyer_msg = f"Your SmartBuyer lease score is {score}/100 — review the flags above."

                narrative = Narrative(**narrative_obj)
                trade_data = self._extract_trade_data(parsed)

                return MultiImageAnalysisResponse(
                    score=score,
                    buyer_name=parsed.get("buyer_name"),
                    dealer_name=parsed.get("dealer_name"),
                    logo_text=parsed.get("logo_text") or parsed.get("dealer_name"),
                    email=parsed.get("email"),
                    phone_number=parsed.get("phone_number"),
                    address=parsed.get("address"),
                    state=parsed.get("state"),
                    region=parsed.get("region") or "Outside US",
                    badge=self._assign_badge(score),
                    selling_price=parsed.get("cap_cost") or parsed.get("selling_price") or parsed.get("sale_price"),
                    vin_number=parsed.get("vin_number") or parsed.get("vin"),
                    date=parsed.get("date"),
                    buyer_message=buyer_msg,
                    red_flags=red_flags,
                    green_flags=green_flags,
                    blue_flags=blue_flags,
                    normalized_pricing=NormalizedPricing(**parsed.get("normalized_pricing", {})) if isinstance(parsed.get("normalized_pricing"), dict) else NormalizedPricing(),
                    apr=APRData(**parsed.get("apr", {})) if isinstance(parsed.get("apr"), dict) else APRData(),
                    term=TermData(**parsed.get("term", {})) if isinstance(parsed.get("term"), dict) else TermData(),
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
                # ── end JSON path ─────────────────────────────────────────────────

            elif base64_images is None:
                validated_files = await self._validate_files(files)
                if not validated_files:
                    raise ValueError("No valid image files provided")
                
                # Optional: Optimize images before base64 encoding
                # optimized_files = await self._optimize_images(validated_files)
                # base64_images = await self._convert_files_to_base64(optimized_files)
                
                base64_images = await self._convert_files_to_base64(validated_files)
                # Deterministic extraction cache (same files -> same parsed extraction)
                cache_key = self._make_cache_key(base64_images)
                cached_parsed = self._load_cached_extraction(cache_key)
                if cached_parsed is not None:
                    print("Using cached extraction for deterministic scoring.")
                    parsed = cached_parsed
                else:
                    # Step 1: Data Extraction (No Narrative)
                    def extraction_factory(detail):
                        return self._get_extraction_messages(base64_images, language, detail)
                    
                    print("Starting Step 1: Data Extraction...")
                    extraction_response = self._run_inference(extraction_factory, max_tokens=3000)
                    parsed = self._parse_api_response(extraction_response)
                    self._save_cached_extraction(cache_key, parsed)
            else:
                base64_images = [img.split(",", 1)[1] if img.startswith("data:") and "," in img else img for img in base64_images]
                # Deterministic extraction cache (same files -> same parsed extraction)
                cache_key = self._make_cache_key(base64_images)
                cached_parsed = self._load_cached_extraction(cache_key)
                if cached_parsed is not None:
                    print("Using cached extraction for deterministic scoring.")
                    parsed = cached_parsed
                else:
                    # Step 1: Data Extraction (No Narrative)
                    def extraction_factory(detail):
                        return self._get_extraction_messages(base64_images, language, detail)
                    
                    print("Starting Step 1: Data Extraction...")
                    extraction_response = self._run_inference(extraction_factory, max_tokens=3000)
                    parsed = self._parse_api_response(extraction_response)
                    self._save_cached_extraction(cache_key, parsed)

            # --- SmartBuyer scoring engine (rules-driven) ---
            rules = load_rules()
            upstream_flags = build_active_flags(parsed.get("flags", []), rules.flag_registry, "upstream")
            computed_flags = compute_flags_from_parsed(parsed, rules, mode="LEASE")
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

            red_flags = self._translate_flags(red_flags, language)
            green_flags = self._translate_flags(green_flags, language)
            blue_flags = self._translate_flags(blue_flags, language)

            if not red_flags:
                red_flags.append(Flag(type="General", message="No major compliance issues identified — review all lease terms before signing.", item="General"))
            if not green_flags:
                green_flags.append(Flag(type="General", message="No standout positive elements identified for this deal.", item="General"))
            if not blue_flags:
                blue_flags.append(Flag(type="General Advisory", message="Review all final lease terms, product details, and payment figures carefully before signing.", item="General Advisory"))

            score_value = float(scoring_result.score_int)
            ai_narrative = self._call_narrative_api(parsed, score_value, red_flags, green_flags, blue_flags, language)
            narrative_obj = ai_narrative.get("narrative", {}) if isinstance(ai_narrative, dict) else {}
            if not isinstance(narrative_obj, dict):
                narrative_obj = {}
            buyer_msg = ai_narrative.get("buyer_message", "") if isinstance(ai_narrative, dict) else ""

            if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")

            narrative_defaults = {
                "vehicle_overview": f"Lease analysis for {parsed.get('dealer_name', 'this dealer')}.",
                "smartbuyer_score_summary": f"SmartBuyer Score: {score_value}/100.",
                "market_comparison": "Market comparison pending.",
                "gap_logic": "GAP analysis pending.",
                "vsc_logic": "VSC analysis pending.",
                "apr_bonus_rule": "APR/rate analysis pending.",
                "lease_audit": "Lease terms analysis pending.",
                "trade": "No trade-in on this deal.",
                "negotiation_insight": "Review all flags before signing.",
                "final_recommendation": "Proceed with caution based on the flags above."
            }
            for key, default_val in narrative_defaults.items():
                if not narrative_obj.get(key):
                    narrative_obj[key] = default_val

            if "trade" in narrative_obj and not isinstance(narrative_obj["trade"], str):
                narrative_obj["trade"] = str(narrative_obj["trade"])

            if not buyer_msg:
                buyer_msg = f"Your SmartBuyer lease score is {score_value}/100 — review the flags above."

            narrative = Narrative(**narrative_obj)
            trade_data = self._extract_trade_data(parsed)

            return MultiImageAnalysisResponse(
                score=score_value,
                buyer_name=parsed.get("buyer_name"),
                dealer_name=parsed.get("dealer_name"),
                logo_text=parsed.get("logo_text") or parsed.get("dealer_name"),
                email=parsed.get("email"),
                phone_number=parsed.get("phone_number"),
                address=parsed.get("address"),
                state=parsed.get("state"),
                region=parsed.get("region") or "Outside US",
                badge=self._assign_badge(score_value),
                selling_price=parsed.get("cap_cost") or parsed.get("selling_price") or parsed.get("sale_price"),
                vin_number=parsed.get("vin_number") or parsed.get("vin"),
                date=parsed.get("date"),
                buyer_message=buyer_msg,
                red_flags=red_flags,
                green_flags=green_flags,
                blue_flags=blue_flags,
                normalized_pricing=NormalizedPricing(**parsed.get("normalized_pricing", {})) if isinstance(parsed.get("normalized_pricing"), dict) else NormalizedPricing(),
                apr=APRData(**parsed.get("apr", {})) if isinstance(parsed.get("apr"), dict) else APRData(),
                term=TermData(**parsed.get("term", {})) if isinstance(parsed.get("term"), dict) else TermData(),
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
            
            # Step 0: Normalize flag keys and remove "Unknown" values
            parsed = self._normalize_flag_fields(parsed)

            # Captive lender detection
            lessor_name = parsed.get("lessor_name") or parsed.get("lender_name") or parsed.get("lessor")
            parsed["captive_lender"] = self._is_captive_lender(lessor_name)
            
            # Step 1: OCR Normalization
            raw_line_items = parsed.get("line_items", [])
            normalized_line_items = self._normalize_line_items(raw_line_items)
            
            # Step 2: Discount Detection and Normalization
            discounts, discount_totals = self.discount_detector.process_line_items(
                normalized_line_items,
                mode="QUOTE"
            )
            
            # Step 3: Audit Classification
            # Fix: Safe float conversion handling None/null values
            raw_cap_cost = parsed.get("cap_cost")
            vehicle_price = float(raw_cap_cost) if raw_cap_cost is not None else 0.0
            
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
            
            # Step 5: GAP Logic Evaluation (Captive-aware)
            term_months = parsed.get("term", {}).get("months")
            gap_items = [c for c in audit_classifications if c.classification == "GAP"]
            gap_present = len(gap_items) > 0
            is_captive = bool(parsed.get("captive_lender"))

            if is_captive:
                # Captive lenders include GAP in lease structure
                if gap_present:
                    audit_flags.append(AuditFlag(
                        type="red",
                        category="Illegal GAP Charge",
                        message="GAP insurance charged on a captive lease where GAP is included.",
                        item="GAP",
                        deduction=10,
                        bonus=None
                    ))
                else:
                    audit_flags.append(AuditFlag(
                        type="green",
                        category="GAP Correct (Captive)",
                        message="No GAP charge on captive lease (GAP is included in lease structure).",
                        item="GAP",
                        deduction=None,
                        bonus=None
                    ))
            else:
                if not gap_present:
                    audit_flags.append(AuditFlag(
                        type="red",
                        category="SOFT - GAP Missing (Advisory)",
                        message="GAP coverage is missing on a non-captive lease. Consider adding GAP protection.",
                        item="GAP",
                        deduction=2,
                        bonus=None
                    ))
            
            # Long-term loan risk flag
            if term_months and term_months >= 72:
                loan_risk_flag = self.flag_builder.build_long_term_loan_risk_flag(term_months)
                audit_flags.append(loan_risk_flag)
            
            # Step 5.5: GAP Pricing (Non-captive only)
            if not is_captive:
                for item in gap_items:
                    price = item.amount or 0.0
                    if price > 1200:
                        audit_flags.append(AuditFlag(
                            type="red",
                            category="GAP Overpriced (Severe)",
                            message=f"GAP insurance charged at ${price:.2f} is significantly overpriced (Market range: $400-$800).",
                            item="GAP",
                            deduction=10,
                            bonus=None
                        ))
                    elif price > 800:
                        audit_flags.append(AuditFlag(
                            type="red",
                            category="SOFT - GAP Overpriced (Moderate)",
                            message=f"GAP insurance charged at ${price:.2f} is above recommended market rate.",
                            item="GAP",
                            deduction=5,
                            bonus=None
                        ))

            # MSD guard (Multiple Security Deposits)
            msd_detected = bool(parsed.get("msd_count") or parsed.get("msd_total"))
            if not msd_detected:
                for item in normalized_line_items:
                    if "msd" in (item.raw_text or "").lower() or "security deposit" in (item.raw_text or "").lower():
                        msd_detected = True
                        break
            
            # Step 6: APR / Money Factor Analysis (Deterministic)
            # Section 6 Rules: ≤6.5% (+5), ≤9.5% (+2), >12% (-5), ≥15% (-10)
            apr_data = parsed.get("apr", {})
            money_factor = apr_data.get("money_factor") or parsed.get("money_factor")
            term_months = parsed.get("term", {}).get("months")

            # Derive MF if missing and rent charge is available
            if (not money_factor) and parsed.get("total_rent_charge") and parsed.get("net_cap_cost") and parsed.get("residual_value") and term_months:
                try:
                    trc = float(parsed.get("total_rent_charge"))
                    ncc = float(parsed.get("net_cap_cost"))
                    rv = float(parsed.get("residual_value"))
                    tm = float(term_months)
                    if tm > 0:
                        money_factor = trc / ((ncc + rv) * tm)
                except (ValueError, TypeError, ZeroDivisionError):
                    money_factor = None

            if money_factor is not None:
                if not isinstance(parsed.get("apr"), dict):
                    parsed["apr"] = {}
                parsed["apr"]["money_factor"] = money_factor

            if money_factor and float(money_factor) > 0 and not msd_detected:
                effective_apr = float(money_factor) * 2400
                if effective_apr <= 6.5:
                    audit_flags.append(AuditFlag(
                        type="green",
                        category="Excellent APR",
                        message=f"Excellent Money Factor/APR of {effective_apr:.2f}% (Market benchmark: <6.5%).",
                        item="Money Factor",
                        deduction=None,
                        bonus=5
                    ))
                elif effective_apr <= 9.5:
                    audit_flags.append(AuditFlag(
                        type="green",
                        category="Good APR",
                        message=f"Good Money Factor/APR of {effective_apr:.2f}% (Market benchmark: <9.5%).",
                        item="Money Factor",
                        deduction=None,
                        bonus=2
                    ))
                elif effective_apr >= 15.0:
                    audit_flags.append(AuditFlag(
                        type="red",
                        category="Predatory APR",
                        message=f"Predatory Money Factor/APR of {effective_apr:.2f}% significantly exceeds market rates.",
                        item="Money Factor",
                        deduction=10,
                        bonus=None
                    ))
                elif effective_apr > 12.0:
                    audit_flags.append(AuditFlag(
                        type="red",
                        category="SOFT - High APR",
                        message=f"High Money Factor/APR of {effective_apr:.2f}% exceeds standard rates.",
                        item="Money Factor",
                        deduction=5,
                        bonus=None
                    ))
            
            # Step 6.5: Process Trade Data (UPDATED - Use OCR-based extraction)
            trade_data = self._extract_trade_data(parsed)
            
            # CRITICAL: Add negative equity red flag if applicable (per Section 13)
            if trade_data.negative_equity and trade_data.negative_equity > 0:
                msrp = parsed.get("normalized_pricing", {}).get("msrp") or 0
                disclosure_present = bool(
                    trade_data.trade_allowance or
                    trade_data.trade_payoff or
                    parsed.get("negative_equity") or
                    (isinstance(parsed.get("trade"), dict) and parsed.get("trade", {}).get("negative_equity") is not None)
                )
                
                if msrp > 0:
                    neg_equity_pct = (trade_data.negative_equity / msrp) * 100
                    
                    # Per Section 13: > 10% MSRP or > $5000 = RED -5
                    if neg_equity_pct > 10 or trade_data.negative_equity > 5000:
                        neg_equity_flag = AuditFlag(
                            type="red",
                            category="High Negative Equity",
                            message=f"${trade_data.negative_equity:,.2f} negative equity ({neg_equity_pct:.1f}% of MSRP) rolled into lease significantly increases total cost and risk. This exceeds recommended thresholds.",
                            item="Trade",
                            deduction=5,  # RED -5
                            bonus=None
                        )
                        audit_flags.append(neg_equity_flag)
                    elif neg_equity_pct > 5:  # 5-10% = SOFT -3
                        neg_equity_flag = AuditFlag(
                            type="red",
                            category="SOFT - Moderate Negative Equity",
                            message=f"${trade_data.negative_equity:,.2f} negative equity ({neg_equity_pct:.1f}% of MSRP) increases lease exposure. Consider impact on monthly payment and total cost.",
                            item="Trade",
                            deduction=3,  # SOFT -3
                            bonus=None
                        )
                        audit_flags.append(neg_equity_flag)
                else:
                    # MSRP not available, use dollar threshold only
                    if trade_data.negative_equity > 5000:
                        neg_equity_flag = AuditFlag(
                            type="red",
                            category="High Negative Equity",
                            message=f"${trade_data.negative_equity:,.2f} negative equity rolled into lease significantly increases total cost and risk.",
                            item="Trade",
                            deduction=5,  # RED -5
                            bonus=None
                        )
                        audit_flags.append(neg_equity_flag)

                # Disclosure rule: negative equity present but not clearly disclosed
                if not disclosure_present:
                    audit_flags.append(AuditFlag(
                        type="red",
                        category="Negative Equity Not Disclosed",
                        message="Negative equity appears rolled into the lease without clear disclosure.",
                        item="Trade",
                        deduction=10,
                        bonus=None
                    ))
            
            # Step 7: Suppress "missing incentive" warnings if Finance Certificate detected
            if finance_certs:
                # Remove any "missing incentive" flags from parsed data
                # (This would be in the original API response processing)
                pass
            
            # Step 7.5: Normalize flag scores (ensure deductions/bonuses are positive)
            parsed = self._normalize_flag_scores(parsed)
            
            # Step 7.6: Compute deterministic scoring flags (Python-only, consistent)
            deterministic_flags = self._compute_deterministic_flags(
                parsed, normalized_line_items, audit_classifications, msd_detected
            )
            for df in deterministic_flags:
                audit_flags.append(df)
            
            # Step 8: Merge audit flags with existing flags
            def parse_flags(flags_data):
                flags_list = []
                if not isinstance(flags_data, list):
                    return []
                
                for item_dict in flags_data:
                    if not item_dict or not isinstance(item_dict, dict):
                        continue
                    
                    # ONLY skip if BOTH type AND message are empty
                    type_val = item_dict.get("type", "")
                    msg_val = item_dict.get("message", "")
                    
                    if not type_val and not msg_val:  # <-- CHANGED: Only skip if BOTH empty
                        continue
                    
                    # Set defaults for missing values
                    if not type_val:
                        type_val = "Analysis Note"
                    if not msg_val:
                        msg_val = "No details provided"
                        
                    flag_kwargs = {
                        "type": type_val,
                        "message": str(msg_val),
                        "item": str(item_dict.get("item", "General")),
                        "deduction": item_dict.get("deduction"),
                        "bonus": item_dict.get("bonus")
                    }
                    try:
                        flags_list.append(Flag(**flag_kwargs))
                    except Exception:
                        pass
                return flags_list

            # Parse original AI flags (DISCARDED - display only, not used in output)
            # All flags are now generated deterministically by Python for consistency
            ai_red = parse_flags(parsed.get("red_flags", []))
            ai_green = parse_flags(parsed.get("green_flags", []))
            ai_blue = parse_flags(parsed.get("blue_flags", []))
            
            # Debug: Log AI-generated flags (informational only, not used)
            print(f"AI Generated (discarded) - Red: {len(ai_red)}, Green: {len(ai_green)}, Blue: {len(ai_blue)}")
            
            # Initialize flags: red/green from Python only; blue includes AI advisory + Python
            red_flags = []
            green_flags = []
            blue_flags = list(ai_blue)
            
            # Add ONLY Python-generated audit flags
            for audit_flag in audit_flags:
                # Use category as the main title/type
                flag_type = audit_flag.category if audit_flag.category else audit_flag.type.title() + " Flag"
                
                flag_obj = Flag(
                    type=flag_type,
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
            
            print(f"Python flags - Red: {len(red_flags)}, Green: {len(green_flags)}, Blue: {len(blue_flags)}")
                    
            # Step 9: Recalculate Score from Flags
            # This ensures consistency even if AI scoring fails or we add python flags
            calculated_score = 100.0
            total_deductions = 0.0
            total_bonuses = 0.0
            
            # CRITICAL: Only score red and green flags, NOT blue flags (advisory only)
            all_scoring_flags = red_flags + green_flags  # Removed blue_flags from scoring
            
            for flag in all_scoring_flags:
                # Handle deductions (should be positive numbers like 10, 5, 3)
                if flag.deduction is not None:
                    val = abs(float(flag.deduction))  # Always use absolute value
                    calculated_score -= val
                    total_deductions += val
                
                # Handle bonuses (should be positive numbers like 5, 3, 2)
                if flag.bonus is not None:
                    val = abs(float(flag.bonus))
                    calculated_score += val
                    total_bonuses += val
            
            print(f"Score Calculation: Start=100, Deductions={total_deductions}, Bonuses={total_bonuses}, Calculated={calculated_score}")
            
            # Apply backend overload cap (Section 12: hard overload > 20% → max 90)
            backend_cap_active = any(
                f.type == "Backend Overload" and f.deduction is not None and abs(float(f.deduction)) >= 10
                for f in red_flags
            )
            if backend_cap_active and calculated_score > 90:
                print(f"Backend overload cap applied: {calculated_score} → 90")
                calculated_score = 90.0
            
            # Clamp to 0-100 — no AI fallback, score is fully deterministic
            final_score = round(max(0.0, min(100.0, calculated_score)), 2)
            print(f"Final Score (deterministic): {final_score}")

            # Translate flags to requested language (no scoring changes)
            red_flags = self._translate_flags(red_flags, language)
            green_flags = self._translate_flags(green_flags, language)
            blue_flags = self._translate_flags(blue_flags, language)
            
            # Prepare flag summary for narrative generation
            flags_for_narrative = {
                "red_flags": [{
                    "type": f.type,
                    "message": f.message,
                    "item": f.item,
                    "deduction": f.deduction
                } for f in red_flags],
                "green_flags": [{
                    "type": f.type,
                    "message": f.message,
                    "item": f.item,
                    "bonus": f.bonus
                } for f in green_flags],
                "blue_flags": [{
                    "type": f.type,
                    "message": f.message,
                    "item": f.item
                } for f in blue_flags]
            }
            
            # NOW generate narrative with final score and all flags
            def narrative_factory(detail):
                return self._get_narrative_messages(base64_images, parsed, final_score, flags_for_narrative, language, detail)
                
            print(f"Starting Step 2: Narrative Generation with final score {final_score}...")
            narrative_response = self._run_inference(narrative_factory, max_tokens=2000)
            parsed_narrative = self._parse_api_response(narrative_response)
            
            # Parse narrative - accept either smartbuyer_score_summary or legacy trust_score_summary
            narrative_obj = parsed_narrative.get("narrative", {})
                
            if isinstance(narrative_obj, str):
                narrative_obj = json.loads(narrative_obj)
            # Normalize legacy key
            if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")
            
            # Ensure required narrative fields exist to prevent validation errors
            # effectively removing "default outputs" (filler text)
            required_narrative_fields = [
                "vehicle_overview", "smartbuyer_score_summary", "market_comparison",
                "gap_logic", "vsc_logic", "apr_bonus_rule", "lease_audit",
                "trade", "negotiation_insight", "final_recommendation"
            ]
            
            for field in required_narrative_fields:
                if field not in narrative_obj or narrative_obj[field] is None:
                    narrative_obj[field] = ""
            
            # Use calculated trade status only if AI completely failed to provide trade narrative
            if not narrative_obj.get("trade"):
                 narrative_obj["trade"] = trade_data.status
            
            # CRITICAL: Final validation to ensure trade is a string, not dict
            if "trade" in narrative_obj and not isinstance(narrative_obj["trade"], str):
                if isinstance(narrative_obj["trade"], dict):
                    trade_obj = narrative_obj["trade"]
                    narrative_obj["trade"] = trade_obj.get("status") or str(trade_obj)
                else:
                    narrative_obj["trade"] = str(narrative_obj["trade"])

            narrative = Narrative(**narrative_obj)
            
            # Fix: Handle None values for required string fields  
            # Use buyer_message from narrative generation step
            buyer_msg = parsed_narrative.get("buyer_message")
            if not buyer_msg:
                buyer_msg = ""

            return MultiImageAnalysisResponse(
                score=final_score,
                buyer_name=parsed.get("buyer_name"),
                dealer_name=parsed.get("dealer_name"),
                logo_text=parsed.get("logo_text"),
                email=parsed.get("email"),
                phone_number=parsed.get("phone_number"),
                address=parsed.get("address"),
                state=parsed.get("state"),
                region=parsed.get("region") or "Outside US",
                badge=self._assign_badge(final_score),
                selling_price=parsed.get("cap_cost"),
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
            raise RuntimeError(f"Lease analysis failed: {str(e)}")
