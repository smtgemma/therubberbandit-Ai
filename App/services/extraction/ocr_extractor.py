import os
import json
import base64
import io
from typing import List, Optional, Dict
from dotenv import load_dotenv
import requests
import fitz  # PyMuPDF
from fastapi import UploadFile

load_dotenv()

class OCRExtractor:
    """Extract text and structured data from all types of automotive documents using ChatGPT Vision"""
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = "claude-3-5-sonnet-latest"
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    async def extract_quote_data(self, files: List[UploadFile]) -> Dict:
        """Extract all quote/contract data using ChatGPT Vision"""
        base64_images = await self._convert_to_base64(files)

        # Extract per-page raw text using PyMuPDF for PDFs (images return empty strings — that's OK)
        page_texts = await self._extract_raw_text_pymupdf(files)
        full_raw_text = "\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t.strip())

        # Extract structured data using Vision API
        system_prompt = self._get_quote_extraction_prompt()
        response = await self._call_anthropic_vision(base64_images, system_prompt)
        parsed = self._parse_response(response)

        # Merge raw text into parsed response
        if "extracted_text" in parsed:
            parsed["extracted_text"]["raw_text"] = full_raw_text
            parsed["extracted_text"]["page_texts"] = page_texts
        else:
            parsed["extracted_text"] = {
                "raw_text": full_raw_text,
                "page_texts": page_texts,
                "sections": {}
            }

        # Strip fields that are no longer part of the response contract
        _remove_keys = {"co_buyer_info", "signatures", "quality_assessment", "extracted_text"}
        for k in _remove_keys:
            parsed.pop(k, None)
        _vd = parsed.get("vehicle_details")
        if isinstance(_vd, dict):
            for k in ("trim", "stock_number", "color", "odometer", "mpg_city", "mpg_highway", "transmission"):
                _vd.pop(k, None)

        return parsed
    
    async def _extract_raw_text_pymupdf(self, files: List[UploadFile]) -> List[str]:
        """Extract per-page raw text from PDF files using PyMuPDF"""
        page_texts = []

        for file in files:
            contents = await file.read()

            try:
                pdf_document = fitz.open(stream=contents, filetype="pdf")
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    page_texts.append(text)
                pdf_document.close()
            except Exception as e:
                print(f"Warning: Could not extract text from {file.filename}: {str(e)}")

            await file.seek(0)

        return page_texts
    
    def _get_quote_extraction_prompt(self) -> str:
        """Comprehensive system prompt — extract every visible field from any auto document."""
        return """
You are an elite automotive document OCR and data extraction specialist. Extract EVERY structured data field visible on the document into a precise JSON response.

═══════════════════════════════════════════
CARDINAL RULES
═══════════════════════════════════════════
1. NEVER invent or guess values. Extract only what is explicitly visible.
2. NEVER omit a field from the JSON schema — use null if not found.
3. NEVER copy a value from one field to fill a different field.
4. Extract every line item, every fee, every charge, every named clause.
5. Preserve exact spelling, capitalization, and formatting of names/addresses.
6. Numeric fields: strip "$" and commas → number. "$67,158.60" → 67158.60.
7. Return ONLY valid JSON — no markdown, no code fences, no explanations.
8. Do NOT reproduce verbatim legal text in any field — use short descriptive values only.

═══════════════════════════════════════════
DOCUMENT TYPE
═══════════════════════════════════════════
- "RETAIL PURCHASE AGREEMENT" → Purchase Agreement
- "MOTOR VEHICLE RETAIL INSTALLMENT SALES CONTRACT" → Contract
- Multi-column financing options → Quote
- "LEASE" / "Lease Agreement" → Lease

═══════════════════════════════════════════
CRITICAL PRICE FIELD RULES
═══════════════════════════════════════════
vehicle_details.sale_price:
  ✓ LAW 553: "Cash Price" on line 1 of the Itemization of Amount Financed section
       → This is itemization.cash_price_including_accessories. Copy the SAME number to sale_price.
  ✓ Purchase Agreements: "TOTAL SELLING PRICE" or "CASH PRICE OF VEHICLE"
  ✓ Quotes: "Selling Price" next to vehicle description
  ✗ NEVER use "Total Sale Price" from the TILA box (→ tila_disclosures.total_sale_price only)
  ✗ NEVER use "Amount Financed" as sale_price

  ⚠ SELF-CHECK (mandatory before returning JSON):
     If vehicle_details.sale_price == tila_disclosures.amount_financed → YOU ARE WRONG.
     A deal with trade-in, fees, and backend products will ALWAYS have:
       amount_financed > cash_price (sale_price)
     Re-read line 1 of the Itemization section. That dollar amount is sale_price.
     sale_price must also equal itemization_of_amount_financed.cash_price_including_accessories.

vehicle_details.msrp:
  ✓ Only if document explicitly labels "MSRP", "MSRP/Retail", "Sticker Price"
  ✗ If no MSRP label found → null

═══════════════════════════════════════════
VEHICLE YEAR/MAKE/MODEL — PURCHASED vs TRADE-IN
═══════════════════════════════════════════
vehicle_details (year/make/model/vin/condition) = the VEHICLE BEING PURCHASED:
  ✓ Read from the "VEHICLE IDENTIFICATION" table on page 1 (YEAR / MAKE / MODEL / VIN columns)
  ✗ NEVER use the trade-in vehicle's year/make/model for vehicle_details

trade_in (year/make/model/vin) = the vehicle being traded in:
  ✓ Read from the "Trade-in: Make / Model / Year / VIN" line below the vehicle table
  These will often be different vehicles (e.g., purchased = 2025, trade-in = 2023)
  ✗ NEVER map "Down Payment" / "Total Downpayment" values into any trade_in fields

═══════════════════════════════════════════
FINANCIAL_TERMS CROSS-POPULATION (REQUIRED)
═══════════════════════════════════════════
Do NOT leave financial_terms fields null when the value exists elsewhere in the document.
Apply these mandatory mappings:
  financial_terms.cash_price         = itemization_of_amount_financed.cash_price_including_accessories
                                       (= vehicle_details.sale_price)
  financial_terms.unpaid_balance_of_cash_price = itemization_of_amount_financed.unpaid_balance_of_cash_price
  financial_terms.total_downpayment  = itemization_of_amount_financed.total_downpayment
  financial_terms.loan_amount        = tila_disclosures.amount_financed
                                       (= itemization_of_amount_financed.amount_financed)
  financial_terms.amount_financed_line5 = itemization_of_amount_financed.amount_financed
  financial_terms.total_other_charges = itemization_of_amount_financed.total_other_charges_and_paid_to_others
  financial_terms.apr                = tila_disclosures.annual_percentage_rate
  financial_terms.finance_charge     = tila_disclosures.finance_charge
  financial_terms.term_months        = payment_schedule.number_of_payments
  financial_terms.monthly_payment    = payment_schedule.payment_amount
  financial_terms.manufacturers_rebate = itemization_of_amount_financed.manufacturers_rebate
  financial_terms.doc_fee            = fees_breakdown.documentary_fee
  financial_terms.sales_tax          = extract from line 1 sub-field "SALES TAX $___" in the Cash Price line
  financial_terms.trade_in_value     = trade_in.gross_trade_in

═══════════════════════════════════════════
DOCUMENT-SPECIFIC FIELD MAPPING
═══════════════════════════════════════════
LAW 553 Contract:
  "Cash Price" line 1 → sale_price + itemization.cash_price_including_accessories
  "Gross Trade-In" → trade_in.gross_trade_in
  "Pay Off Made By Seller to [lender]" → trade_in.trade_payoff + trade_in.payoff_to_lender
  "Cash Paid to Buyer for Trade-In" → trade_in.cash_paid_to_buyer
  "Net Trade Allowance" → trade_in.net_trade_allowance
  "Manufacturers Rebate" → financial_terms.manufacturers_rebate
  "Total Downpayment" line 2 → financial_terms.total_downpayment
  "Unpaid Balance of Cash Price" line 3 → itemization.unpaid_balance_of_cash_price
  Section 4 rows A-O → itemization.other_charges[] each with label/description/amount
  Line 4 total → itemization.total_other_charges_and_paid_to_others
  Line 5 "Amount Financed" → itemization.amount_financed + financial_terms.loan_amount
  TILA box 5 fields → tila_disclosures.*
  Payment rows → payment_schedule.*
  "Late Charge" clause → financial_terms.late_charge_days + financial_terms.late_charge_rate
  "Returned Payment Charge $__" → legal_clauses.returned_payment_charge (number only)
  Arbitration checkbox/provision → legal_clauses.arbitration_provision (true/false)
  Vehicle checkboxes NEW/USED/DEMO → vehicle_details.condition
  Use checkboxes PERSONAL/BUSINESS → vehicle_details.use_purpose
  Form number e.g. "LAW 553-TX-ARB-e 3/25" → document_metadata.form_number
  OCCC notice lender name + phone → lender_info.lender_name + lender_info.lender_phone
  OCCC address → lender_info.lender_address

Retail Purchase Agreement:
  "CASH PRICE OF VEHICLE" or "TOTAL SELLING PRICE" → sale_price
  "LESS: TRADE-IN ALLOWANCE" → trade_in.gross_trade_in
  "PAY OFF AMOUNT" → trade_in.trade_payoff
  Accessory lines → addons_and_packages[]

Sales Quote (multi-column):
  "MSRP/Retail" → msrp
  "Selling Price" → sale_price
  "Trade Allowance" → trade_in.gross_trade_in
  "Trade Payoff" / "Payoff" / "Amount Owed" → trade_in.trade_payoff
  "Trade Difference" (if explicitly shown) → map by sign:
    - If labeled as negative/owed/over-allowance, treat as negative equity amount
    - If labeled as credit/equity/under-allowance, treat as positive equity amount
    - If sign context is unclear, do not infer sign from the number alone
  "Amount Financed" → financial_terms.loan_amount
  Selected term → payment_schedule.*

QUOTE TRADE EXTRACTION SAFETY RULES (MANDATORY):
  - Trade Allowance = dealership value offered for the trade vehicle.
  - Trade Payoff = amount owed on the trade vehicle loan/lien.
  - Trade equity relationship:
      equity = trade_allowance - trade_payoff
      negative_equity = max(0, trade_payoff - trade_allowance)
  - NEVER use "Cash Down", "Down Payment", or payment-option column headers/values
    as trade allowance, trade payoff, or trade difference.
  - If a quote table has multiple "Cash Down" options (e.g., 8000 / 12000 / 15000),
    those values belong only to down payment context, not trade context.

═══════════════════════════════════════════
JSON SCHEMA
═══════════════════════════════════════════
{
  "document_metadata": {
    "form_number": null,
    "document_title": null,
    "document_date": null,
    "quote_type": null
  },
  "buyer_info": {
    "name": null,
    "address": null,
    "city": null,
    "state": null,
    "zip": null,
    "phone": null,
    "email": null
  },
  "dealer_info": {
    "name": null,
    "address": null,
    "city": null,
    "state": null,
    "zip": null,
    "phone": null,
    "email": null
  },
  "vehicle_details": {
    "year": null,
    "make": null,
    "model": null,
    "vin": null,
    "condition": null,
    "use_purpose": null,
    "msrp": null,
    "sale_price": null
  },
  "trade_in": {
    "year": null,
    "make": null,
    "model": null,
    "vin": null,
    "license_number": null,
    "odometer": null,
    "gross_trade_in": null,
    "payoff_to_lender": null,
    "trade_payoff": null,
    "cash_paid_to_buyer": null,
    "net_trade_allowance": null
  },
  "tila_disclosures": {
    "annual_percentage_rate": null,
    "finance_charge": null,
    "amount_financed": null,
    "total_of_payments": null,
    "total_sale_price": null,
    "down_payment_tila": null
  },
  "payment_schedule": {
    "number_of_payments": null,
    "payment_amount": null,
    "first_payment_date": null,
    "final_payment_amount": null,
    "final_payment_date": null,
    "payment_frequency": null
  },
  "financial_terms": {
    "cash_price": null,
    "unpaid_balance_of_cash_price": null,
    "total_downpayment": null,
    "down_payment": null,
    "manufacturers_rebate": null,
    "trade_in_value": null,
    "loan_amount": null,
    "amount_financed_line5": null,
    "total_other_charges": null,
    "apr": null,
    "term_months": null,
    "monthly_payment": null,
    "total_interest": null,
    "finance_charge": null,
    "doc_fee": null,
    "title_fee": null,
    "registration_fee": null,
    "sales_tax": null,
    "sales_tax_rate": null,
    "late_charge_days": null,
    "late_charge_rate": null,
    "prepayment_penalty": null
  },
  "itemization_of_amount_financed": {
    "cash_price_including_accessories": null,
    "sales_tax_on_cash_price": null,
    "gross_trade_in": null,
    "payoff_by_seller": null,
    "cash_paid_to_buyer_for_trade": null,
    "net_trade_allowance": null,
    "manufacturers_rebate": null,
    "other_credits": [],
    "total_downpayment": null,
    "unpaid_balance_of_cash_price": null,
    "other_charges": [],
    "total_other_charges_and_paid_to_others": null,
    "amount_financed": null
  },
  "fees_breakdown": {
    "road_and_bridge_fee": null,
    "temporary_tag_fee": null,
    "state_inspection_fee": null,
    "vehicle_emissions_fee": null,
    "deputy_service_fee": null,
    "dealer_inventory_tax": null,
    "sales_tax": null,
    "government_license_registration": null,
    "government_certificate_of_title": null,
    "vehicle_inspection_program_fee": null,
    "documentary_fee": null,
    "debt_cancellation_fee": null,
    "other_fees": []
  },
  "addons_and_packages": [],
  "lease_specific": {
    "money_factor": null,
    "residual_value": null,
    "residual_percent": null,
    "annual_miles": null,
    "excess_mile_fee": null,
    "acquisition_fee": null,
    "disposition_fee": null,
    "cap_cost": null,
    "cap_cost_reduction": null,
    "drive_off_total": null
  },
  "lender_info": {
    "lender_name": null,
    "lender_phone": null,
    "lender_address": null
  },
  "legal_clauses": {
    "arbitration_provision": null,
    "returned_payment_charge": null,
    "liability_insurance_present": null,
    "security_interest_present": null,
    "prepayment_penalty_present": null,
    "late_charge_present": null
  },
}

═══════════════════════════════════════════
ARRAY ITEM STRUCTURES
═══════════════════════════════════════════
itemization_of_amount_financed.other_charges:
  { "label": "A", "description": "Net trade-in payoff to NMAC", "amount": 12039.44 }

itemization_of_amount_financed.other_credits:
  { "description": "Manufacturer Rebate", "amount": 1208.90 }

fees_breakdown.other_fees:
  { "description": "Plate Transfer Fee", "paid_to": "State", "amount": null }

addons_and_packages:
  { "name": "SERVICE CONTRACT", "price": 2000.00, "paid_to": "NESNA", "category": "GAP|VSC|Service Contract|Warranty|Maintenance|Appearance|Paint|Interior|Tint|Wheel|Other" }

legal_clauses boolean fields: use true if clause is present on document, false if explicitly absent, null if not determinable.
legal_clauses.returned_payment_charge: extract as number (e.g. 30), not text.
"""
    
    async def _convert_to_base64(self, files: List[UploadFile]) -> List[str]:
        """Convert uploaded files to base64"""
        base64_images = []
        for file in files:
            contents = await file.read()
            base64_content = base64.b64encode(contents).decode('utf-8')
            base64_images.append(base64_content)
            await file.seek(0)
        return base64_images
    
    async def _call_anthropic_vision(self, base64_images: List[str], system_prompt: str) -> dict:
        """Call Anthropic Vision API"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Build user content with all images
        content = []
        for base64_image in base64_images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            })
        content.append({
            "type": "text",
            "text": "Extract every field and all text from this document following the instructions exactly. Return only valid JSON."
        })

        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 4096
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            raw = response.json()
            # Warn if the model hit the token limit (response is truncated)
            stop_reason = raw.get("stop_reason", "")
            if stop_reason == "max_tokens":
                raise RuntimeError(
                    "Extraction response was truncated (max_tokens reached). "
                    "The document may be too large. Try splitting into fewer pages per request."
                )
            return raw
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Anthropic Vision API error: {str(e)}")
    
    def _parse_response(self, response: dict) -> dict:
        """Parse Anthropic Vision response"""
        try:
            if isinstance(response.get("content"), list):
                content = response["content"][0]["text"]
            else:
                content = response.get("content", "")
            
            # Remove markdown JSON fences if model adds them
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            elif "```" in content:
                content = content.replace("```", "").strip()
                
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected API response structure: {str(e)}")

        # Primary parse — response_format:json_object should always be valid JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError as primary_err:
            pass

        # Fallback: strip any accidental markdown fences and retry
        try:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError as fallback_err:
            raise RuntimeError(
                f"Could not parse model response as JSON. "
                f"The response may have been truncated or malformed. "
                f"Parse error: {fallback_err}. "
                f"Response preview: {content[:300]!r}"
            )

        raise RuntimeError(
            f"No valid JSON found in model response. "
            f"Response preview: {content[:300]!r}"
        )