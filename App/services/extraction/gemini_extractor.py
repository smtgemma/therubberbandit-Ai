import os
import io
import fitz # PyMuPDF
from typing import List, Dict, Optional
from fastapi import UploadFile
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import traceback

class DocumentMetadata(BaseModel):
    form_number: Optional[str] = None
    document_title: Optional[str] = None
    document_date: Optional[str] = None
    quote_type: Optional[str] = None

class BuyerInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class DealerInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

class VehicleDetails(BaseModel):
    year: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    vin: Optional[str] = None
    condition: Optional[str] = None
    use_purpose: Optional[str] = None
    msrp: Optional[float] = None
    sale_price: Optional[float] = None

class TradeIn(BaseModel):
    year: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    vin: Optional[str] = None
    license_number: Optional[str] = None
    odometer: Optional[str] = None
    gross_trade_in: Optional[float] = None
    payoff_to_lender: Optional[float] = None
    trade_payoff: Optional[float] = None
    cash_paid_to_buyer: Optional[float] = None
    net_trade_allowance: Optional[float] = None
    equity: Optional[float] = Field(None, description="Trade allowance minus trade payoff (positive if allowance > payoff)")
    negative_equity: Optional[float] = Field(None, description="Trade payoff minus trade allowance, returned as a negative number for visual representation (e.g. -13248.34)")

class TilaDisclosures(BaseModel):
    annual_percentage_rate: Optional[float] = None
    finance_charge: Optional[float] = None
    amount_financed: Optional[float] = None
    total_of_payments: Optional[float] = None
    total_sale_price: Optional[float] = None
    down_payment_tila: Optional[float] = None

class PaymentSchedule(BaseModel):
    number_of_payments: Optional[int] = None
    payment_amount: Optional[float] = None
    first_payment_date: Optional[str] = None
    final_payment_amount: Optional[float] = None
    final_payment_date: Optional[str] = None
    payment_frequency: Optional[str] = None

class FinancialTerms(BaseModel):
    cash_price: Optional[float] = None
    unpaid_balance_of_cash_price: Optional[float] = None
    total_downpayment: Optional[float] = None
    down_payment: Optional[float] = None
    manufacturers_rebate: Optional[float] = None
    trade_in_value: Optional[float] = None
    loan_amount: Optional[float] = None
    amount_financed_line5: Optional[float] = None
    total_other_charges: Optional[float] = None
    apr: Optional[float] = None
    term_months: Optional[int] = None
    monthly_payment: Optional[float] = None
    total_interest: Optional[float] = None
    finance_charge: Optional[float] = None
    doc_fee: Optional[float] = None
    title_fee: Optional[float] = None
    registration_fee: Optional[float] = None
    sales_tax: Optional[float] = None
    sales_tax_rate: Optional[float] = None
    late_charge_days: Optional[int] = None
    late_charge_rate: Optional[float] = None
    prepayment_penalty: Optional[float] = None

class OtherCharge(BaseModel):
    label: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[float] = None

class OtherCredit(BaseModel):
    description: Optional[str] = None
    amount: Optional[float] = None

class ItemizationOfAmountFinanced(BaseModel):
    cash_price_including_accessories: Optional[float] = None
    sales_tax_on_cash_price: Optional[float] = None
    gross_trade_in: Optional[float] = None
    payoff_by_seller: Optional[float] = None
    cash_paid_to_buyer_for_trade: Optional[float] = None
    net_trade_allowance: Optional[float] = None
    manufacturers_rebate: Optional[float] = None
    other_credits: List[OtherCredit] = Field(default_factory=list)
    total_downpayment: Optional[float] = None
    unpaid_balance_of_cash_price: Optional[float] = None
    other_charges: List[OtherCharge] = Field(default_factory=list)
    total_other_charges_and_paid_to_others: Optional[float] = None
    amount_financed: Optional[float] = None

class OtherFee(BaseModel):
    description: Optional[str] = None
    paid_to: Optional[str] = None
    amount: Optional[float] = None

class FeesBreakdown(BaseModel):
    road_and_bridge_fee: Optional[float] = None
    temporary_tag_fee: Optional[float] = None
    state_inspection_fee: Optional[float] = None
    vehicle_emissions_fee: Optional[float] = None
    deputy_service_fee: Optional[float] = None
    dealer_inventory_tax: Optional[float] = None
    sales_tax: Optional[float] = None
    government_license_registration: Optional[float] = None
    government_certificate_of_title: Optional[float] = None
    vehicle_inspection_program_fee: Optional[float] = None
    documentary_fee: Optional[float] = None
    debt_cancellation_fee: Optional[float] = None
    other_fees: List[OtherFee] = Field(default_factory=list)

class AddonAndPackage(BaseModel):
    name: Optional[str] = Field(None, description="The name of the addon/package (e.g. GAP, VSC, Service Contract)")
    price: Optional[float] = Field(None, description="Total price of the addon/package")
    paid_to: Optional[str] = Field(None, description="Entity the payment is made to")
    category: Optional[str] = Field(None, description="The category must be one of: GAP|VSC|Service Contract|Warranty|Maintenance|Appearance|Paint|Interior|Tint|Wheel|Other")

class LeaseSpecific(BaseModel):
    money_factor: Optional[float] = None
    residual_value: Optional[float] = None
    residual_percent: Optional[float] = None
    annual_miles: Optional[int] = None
    excess_mile_fee: Optional[float] = None
    acquisition_fee: Optional[float] = None
    disposition_fee: Optional[float] = None
    cap_cost: Optional[float] = None
    cap_cost_reduction: Optional[float] = None
    drive_off_total: Optional[float] = None

class LenderInfo(BaseModel):
    lender_name: Optional[str] = None
    lender_phone: Optional[str] = None
    lender_address: Optional[str] = None

class LegalClauses(BaseModel):
    arbitration_provision: Optional[bool] = None
    returned_payment_charge: Optional[float] = None
    liability_insurance_present: Optional[bool] = None
    security_interest_present: Optional[bool] = None
    prepayment_penalty_present: Optional[bool] = None
    late_charge_present: Optional[bool] = None

class Confidence(BaseModel):
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    reason: Optional[str] = None

class QuoteExtraction(BaseModel):
    confidence: Confidence
    document_metadata: DocumentMetadata
    buyer_info: BuyerInfo
    dealer_info: DealerInfo
    vehicle_details: VehicleDetails
    trade_in: TradeIn
    tila_disclosures: TilaDisclosures
    payment_schedule: PaymentSchedule
    financial_terms: FinancialTerms
    itemization_of_amount_financed: ItemizationOfAmountFinanced
    fees_breakdown: FeesBreakdown
    addons_and_packages: List[AddonAndPackage] = Field(default_factory=list)
    lease_specific: LeaseSpecific
    lender_info: LenderInfo
    legal_clauses: LegalClauses


class GeminiExtractor:
    """Extract text and structured data from all types of automotive documents using Gemini"""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
    
    async def extract_quote_data(self, files: List[UploadFile]) -> Dict:
        """Extract all quote/contract data using Gemini API"""
        
        # Extract per-page raw text using PyMuPDF for PDFs
        page_texts = await self._extract_raw_text_pymupdf(files)
        full_raw_text = "\n\n--- PAGE BREAK ---\n\n".join(t for t in page_texts if t.strip())

        # Collect files for Gemini prompt
        contents = []
        for file in files:
            contents.append(
                await self._file_to_genai_part(file)
            )

        system_prompt = self._get_quote_extraction_prompt()
        contents.append(system_prompt)

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": QuoteExtraction.model_json_schema(),
                "temperature": 0.0,
            },
        )
        
        parsed = QuoteExtraction.model_validate_json(response.text).model_dump()

        # Merge raw text into parsed response
        if "extracted_text" not in parsed:
            parsed["extracted_text"] = {}
            
        parsed["extracted_text"]["raw_text"] = full_raw_text
        parsed["extracted_text"]["page_texts"] = page_texts

        return parsed

    async def _file_to_genai_part(self, file: UploadFile):
        file_bytes = await file.read()
        await file.seek(0)
        
        # Determine mime type
        mime_type = file.content_type
        if not mime_type or mime_type == "application/octet-stream":
            filename = file.filename.lower()
            if filename.endswith(".pdf"):
                mime_type = "application/pdf"
            elif filename.endswith(".png"):
                mime_type = "image/png"
            elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/jpeg" # fallback
                
        return types.Part.from_bytes(data=file_bytes, mime_type=mime_type)

    async def _extract_raw_text_pymupdf(self, files: List[UploadFile]) -> List[str]:
        """Extract per-page raw text from PDF files using PyMuPDF"""
        page_texts = []

        for file in files:
            contents = await file.read()

            try:
                if file.filename.lower().endswith('.pdf') or (file.content_type and 'pdf' in file.content_type):
                    pdf_document = fitz.open(stream=contents, filetype="pdf")
                    for page_num in range(len(pdf_document)):
                        page = pdf_document[page_num]
                        text = page.get_text()
                        page_texts.append(text)
                    pdf_document.close()
                else:
                    # It's an image, PyMuPDF text extraction might not yield text directly
                    page_texts.append("")
            except Exception as e:
                print(f"Warning: Could not extract text from {file.filename}: {str(e)}")

            await file.seek(0)

        return page_texts
    
    def _get_quote_extraction_prompt(self) -> str:
        """Comprehensive system prompt — extract every visible field from any auto document."""
        return """
You are an elite automotive document OCR and data extraction specialist. Extract EVERY structured data field visible on the document into a precise JSON response.

Keep all existing response objects and object names unchanged. Add exactly one new top-level object named confidence with score and reason. confidence.score must be 0.0 to 1.0. confidence.reason should briefly mention handwriting legibility, image quality, blur/cropping, or uncertainty. Use lower scores for handwritten letters, blurry scans, cropped fields, ambiguous numbers, or unreadable text.

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
  "Pay Off Made By Seller to [library/lender]" → trade_in.trade_payoff + trade_in.payoff_to_lender
  "Net Trade-In" (line 2 total) → trade_in.net_trade_allowance + trade_in.negative_equity (as negative)
  "Debt Cancellation Agreement (GAP)" (line 4 row E) → addons_and_packages[] with category='GAP'
  "SERVIC CONTRACT" or "VSC" (line 4 row O) → addons_and_packages[] with category='VSC'
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
  "Fees" / "Dealer Fees" / "Doc Fee" / "Total Fees" → fees_breakdown.documentary_fee
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


