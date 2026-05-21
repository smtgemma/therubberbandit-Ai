from pydantic import BaseModel, Field
from typing import List, Optional

class Flag(BaseModel):
    type: str = Field(..., description="Flag type in 10 words or less")
    message: str = Field(..., description="Detailed explanation")
    deduction: Optional[float] = Field(default=None, description="Points deducted (red flags)")
    bonus: Optional[float] = Field(default=None, description="Points added (green flags)")
    item: str = Field(..., description="Item name")

class NormalizedPricing(BaseModel):
    msrp: Optional[float] = None
    selling_price: Optional[float] = None
    discount: Optional[float] = None
    rebate: Optional[float] = None
    down_payment: Optional[float] = None
    trade_in_value: Optional[float] = None
    amount_financed: Optional[float] = None
    total_fees: Optional[float] = None
    total_taxes: Optional[float] = None

class APRData(BaseModel):
    rate: Optional[float] = None
    estimated: bool = False

class TermData(BaseModel):
    months: Optional[int] = None

class TradeData(BaseModel):
    trade_allowance: Optional[float] = None
    trade_payoff: Optional[float] = None
    equity: Optional[float] = None
    negative_equity: Optional[float] = None
    status: str = "No trade identified"

class Narrative(BaseModel):
    vehicle_overview: str
    smartbuyer_score_summary: str
    score_breakdown: Optional[str] = None
    market_comparison: str
    gap_logic: str
    vsc_logic: str
    apr_bonus_rule: str
    lease_audit: str
    trade: str = "No trade identified"  # REQUIRED
    negotiation_insight: str
    final_recommendation: str

class ContractJsonRequest(BaseModel):
    data: dict = Field(..., description="Pre-extracted JSON data for contract analysis")
    language: str = Field(default="English", description="Language for narrative parts")

class MultiImageAnalysisResponse(BaseModel):
    score: float = Field(..., description="Overall score 0-95")
    buyer_name: Optional[str] = None
    dealer_name: Optional[str] = None
    logo_text: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    state: Optional[str] = None
    region: str = "Outside US"
    badge: str = Field(..., description="Gold|Silver|Bronze|Red")
    selling_price: Optional[float] = None
    vin_number: Optional[str] = None
    date: Optional[str] = None
    quote_type: str = Field(default="contract", description="Document type analyzed by AI: contract or lease")
    buyer_message: str
    red_flags: List[Flag] = Field(default_factory=list)
    green_flags: List[Flag] = Field(default_factory=list)
    blue_flags: List[Flag] = Field(default_factory=list)
    normalized_pricing: NormalizedPricing
    apr: APRData
    term: TermData
    trade: TradeData
    bundle_abuse: dict = Field(default_factory=dict)
    narrative: Narrative
