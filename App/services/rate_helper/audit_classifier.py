from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field
from .ocr_normalization_schema import NormalizedLineItem
from .pricing_caps_loader import load_pricing_caps, get_pricing_cap

# Product classification types
ProductClassification = Literal[
    "CONDITIONAL_FINANCE_INCENTIVE",
    "BUNDLED_ADDON_PACKAGE",
    "GAP",
    "VSC",
    "MAINTENANCE",
    "TIRE_WHEEL_PROTECTION",
    "DEALER_FEE",
    "GOV_FEE",
    "TAX",
    "MARKET_ADJUSTMENT",
    "UNKNOWN"
]

class AuditClassification(BaseModel):
    """Classification result for audit purposes"""
    classification: ProductClassification
    label: str
    amount: float
    is_transparency_issue: bool = Field(False, description="Soft flag for transparency")
    is_overpriced: bool = Field(False, description="Hard flag for pricing")
    penalty_points: int = Field(0, description="Penalty points for this item")
    flag_message: str = Field("", description="User-facing message")
    matched_keyword: Optional[str] = None

class AuditClassifier:
    """
    Product classifier for audit logic with specific rules for:
    1. Finance Certificate (Conditional Finance Incentive)
    2. Propack/Bundles (Bundled Add-On Package)
    3. Backend products with transparency checks
    
    Runs AFTER OCR normalization and discount detection.
    """
    
    # Finance Certificate Detection
    FINANCE_CERT_KEYWORDS = [
        "finance certificate",
        "finance cert",
        "conditional finance",
        "dealer finance incentive",
        "finance incentive"
    ]
    
    # Bundled Package Detection
    BUNDLE_KEYWORDS = [
        "propack",
        "pro pack",
        "protection package",
        "plus package",
        "dealer package"
    ]
    
    def __init__(self):
        pricing_caps = load_pricing_caps()
        self.bundle_max_price = float(
            get_pricing_cap(pricing_caps, ("add_ons", "bundle_max_price"))
        )
        self.bundle_max_percent_of_vehicle = float(
            get_pricing_cap(pricing_caps, ("add_ons", "bundle_max_percent_of_vehicle"))
        )
    
    def _clean_text(self, text: str) -> str:
        """Normalize text for matching"""
        import re
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def classify_for_audit(
        self,
        normalized_item: NormalizedLineItem,
        vehicle_price: Optional[float] = None
    ) -> AuditClassification:
        """
        Classify a normalized line item for audit purposes.
        
        Args:
            normalized_item: Already normalized by OCR normalizer
            vehicle_price: Vehicle selling price for percentage calculations
        
        Returns:
            AuditClassification with penalty and messaging
        """
        cleaned_text = self._clean_text(normalized_item.raw_text)
        amount = abs(normalized_item.amount_normalized)
        
        # PRIORITY 1: Finance Certificate Detection (CRITICAL)
        if self._is_finance_certificate(cleaned_text):
            return self._classify_finance_certificate(normalized_item)
        
        # PRIORITY 2: Bundled Package Detection
        if self._is_bundled_package(cleaned_text):
            return self._classify_bundled_package(normalized_item, vehicle_price)
        
        # PRIORITY 3: Backend products (already classified by OCR normalizer)
        if normalized_item.normalized_category in ["GAP", "VSC", "MAINTENANCE"]:
            return self._classify_backend_product(normalized_item)
        
        # Default: Use OCR normalizer classification
        return AuditClassification(
            classification="UNKNOWN",
            label=normalized_item.normalized_label,
            amount=amount,
            matched_keyword=normalized_item.matched_keyword
        )
    
    def _is_finance_certificate(self, cleaned_text: str) -> bool:
        """Check if item is a Finance Certificate"""
        for keyword in self.FINANCE_CERT_KEYWORDS:
            if keyword in cleaned_text:
                return True
        return False
    
    def _classify_finance_certificate(
        self,
        normalized_item: NormalizedLineItem
    ) -> AuditClassification:
        """
        Classify Finance Certificate as Conditional Finance Incentive.
        
        Audit Rules:
        - Do NOT treat as VSC, add-on, rebate, or product
        - Soft transparency flag (-3 to -5 points)
        - Explain it's tied to dealer financing
        """
        amount = abs(normalized_item.amount_normalized)
        
        # Penalty: -3 to -5 based on amount
        penalty = -5 if amount > 2000 else -3
        
        return AuditClassification(
            classification="CONDITIONAL_FINANCE_INCENTIVE",
            label="Conditional Finance Incentive",
            amount=amount,
            is_transparency_issue=True,
            is_overpriced=False,
            penalty_points=penalty,
            flag_message=(
                "Conditional finance incentive detected. This is not a physical product—it's tied to "
                "dealer-arranged financing. You may lose this incentive if you use outside financing. "
                "Verify APR disclosure and final terms."
            ),
            matched_keyword="finance certificate"
        )
    
    def _is_bundled_package(self, cleaned_text: str) -> bool:
        """Check if item is a bundled package"""
        for keyword in self.BUNDLE_KEYWORDS:
            if keyword in cleaned_text:
                return True
        return False
    
    def _classify_bundled_package(
        self,
        normalized_item: NormalizedLineItem,
        vehicle_price: Optional[float]
    ) -> AuditClassification:
        """
        Classify bundled add-on package (Propack, etc).
        
        Audit Rules:
        - Flag as overpriced if: price > $1,500 OR > 10% of vehicle price
        - Require itemization disclosure
        - Penalty: -8 to -12 points depending on severity
        """
        amount = abs(normalized_item.amount_normalized)
        is_overpriced = False
        penalty = 0
        
        # Check pricing thresholds
        price_threshold_exceeded = amount > self.bundle_max_price
        
        percent_threshold_exceeded = False
        if vehicle_price and vehicle_price > 0:
            percent_of_vehicle = amount / vehicle_price
            percent_threshold_exceeded = percent_of_vehicle > self.bundle_max_percent_of_vehicle
        
        # Determine if overpriced
        if price_threshold_exceeded or percent_threshold_exceeded:
            is_overpriced = True
            # Penalty scaling based on severity
            if amount > 2500 or (vehicle_price and (amount / vehicle_price) > 0.15):
                penalty = -12
            elif amount > 2000 or (vehicle_price and (amount / vehicle_price) > 0.125):
                penalty = -10
            else:
                penalty = -8
        
        flag_message = (
            "This package includes multiple dealer-installed items that are not itemized. "
            "Request a breakdown of individual components or removal of unwanted items."
        )
        
        if is_overpriced:
            flag_message = (
                f"Overpriced bundled package detected (${amount:,.0f}). "
                "This package is not itemized and exceeds typical pricing thresholds. "
                "Request a detailed breakdown or consider removing this package."
            )
        
        return AuditClassification(
            classification="BUNDLED_ADDON_PACKAGE",
            label="Bundled Add-On Package",
            amount=amount,
            is_transparency_issue=True,  # Always a transparency issue
            is_overpriced=is_overpriced,
            penalty_points=penalty,
            flag_message=flag_message,
            matched_keyword=normalized_item.matched_keyword
        )
    
    def _classify_backend_product(
        self,
        normalized_item: NormalizedLineItem
    ) -> AuditClassification:
        """
        Classify backend products (GAP, VSC, Maintenance).
        
        Quote Mode Rules:
        - Pricing NOT enforced
        - Disclosure clarity enforced
        """
        amount = abs(normalized_item.amount_normalized)
        
        # Map OCR category to audit classification
        classification_map = {
            "GAP": "GAP",
            "VSC": "VSC",
            "MAINTENANCE": "MAINTENANCE"
        }
        
        classification = classification_map.get(
            normalized_item.normalized_category,
            "UNKNOWN"
        )
        
        return AuditClassification(
            classification=classification,
            label=normalized_item.normalized_label,
            amount=amount,
            is_transparency_issue=False,
            is_overpriced=False,
            penalty_points=0,  # No pricing enforcement in quote mode
            flag_message="",
            matched_keyword=normalized_item.matched_keyword
        )
