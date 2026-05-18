import re
from typing import List, Optional, Tuple
from .discount_schema import DiscountLineItem, DiscountTotals, DiscountType, SignSource, AnalysisMode
from .discount_keywords import DiscountKeywords
from .ocr_normalization_schema import NormalizedLineItem

class DiscountDetector:
    """
    Discount/Rebate/Incentive detection and normalization layer.
    
    CRITICAL RULES:
    1. All discounts MUST be negative (forced sign correction)
    2. Category lock: Once classified as discount, never reclassified
    3. Priority order: Conditional Finance > Conditional Lease > Unconditional > Dealer
    4. No scoring impact - only identification and normalization
    
    Runs AFTER OCR normalization, BEFORE QBI/scoring logic.
    """
    
    # Confidence threshold for AMBIGUOUS classification
    CONFIDENCE_THRESHOLD = 0.60
    
    def __init__(self):
        self.keywords = DiscountKeywords.get_discount_keywords()
        self.sign_indicators = DiscountKeywords.get_explicit_sign_indicators()
    
    def _clean_text(self, text: str) -> str:
        """Normalize text for matching (lowercase, no punctuation)"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _extract_amount(self, amount_raw: str) -> float:
        """Extract numeric amount from raw string"""
        cleaned = re.sub(r'[^\d.-]', '', str(amount_raw))
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def _detect_explicit_sign(self, raw_text: str, amount_raw: str) -> bool:
        """
        Detect if discount has explicit negative notation.
        
        Explicit signs:
        - Minus sign: "-750"
        - Parentheses: "(750)"
        - "credit" or "cr" in text
        """
        # Check for minus in amount
        if '-' in str(amount_raw):
            return True
        
        # Check for parentheses (accounting notation)
        if '(' in str(amount_raw) and ')' in str(amount_raw):
            return True
        
        # Check for credit keywords in text
        cleaned_text = self._clean_text(raw_text)
        for indicator in self.sign_indicators:
            if indicator in cleaned_text:
                return True
        
        return False
    
    def _classify_discount_type(self, cleaned_text: str) -> Tuple[DiscountType, Optional[str], float]:
        """
        Classify discount type based on keywords.
        
        Returns: (discount_type, matched_keyword, confidence)
        
        Priority order (CRITICAL):
        1. CONDITIONAL_FINANCE
        2. CONDITIONAL_LEASE
        3. UNCONDITIONAL
        4. DEALER_DISCOUNT
        """
        # Check in priority order
        priority_order = [
            "CONDITIONAL_FINANCE",
            "CONDITIONAL_LEASE",
            "UNCONDITIONAL",
            "DEALER_DISCOUNT"
        ]
        
        for discount_type in priority_order:
            keywords = self.keywords[discount_type]
            for keyword in keywords:
                if keyword.lower() in cleaned_text:
                    # Direct keyword match = high confidence
                    confidence = 1.0
                    return (discount_type, keyword, confidence)
        
        # No match - return AMBIGUOUS with low confidence
        return ("AMBIGUOUS", None, 0.0)
    
    def _force_negative_sign(self, amount: float) -> float:
        """
        CRITICAL RULE: All discounts MUST be negative.
        Force sign correction even if OCR shows positive.
        """
        return -abs(amount)
    
    def detect_discount(
        self,
        normalized_item: NormalizedLineItem,
        mode: AnalysisMode = "QUOTE"
    ) -> Optional[DiscountLineItem]:
        """
        Detect if a normalized line item is a discount and extract details.
        
        Args:
            normalized_item: Line item from OCR normalizer
            mode: Analysis mode (QUOTE/CONTRACT/LEASE)
        
        Returns:
            DiscountLineItem if item is a discount, None otherwise
        """
        # Only process DISCOUNT_INCENTIVE category from OCR normalizer
        if normalized_item.normalized_category != "DISCOUNT_INCENTIVE":
            return None
        
        cleaned_text = self._clean_text(normalized_item.raw_text)
        amount = self._extract_amount(normalized_item.amount_raw)
        
        # Classify discount type
        discount_type, matched_keyword, confidence = self._classify_discount_type(cleaned_text)
        
        # Force AMBIGUOUS if confidence below threshold
        if confidence < self.CONFIDENCE_THRESHOLD:
            discount_type = "AMBIGUOUS"
            confidence = 0.0
        
        # Detect explicit sign
        has_explicit_sign = self._detect_explicit_sign(
            normalized_item.raw_text,
            normalized_item.amount_raw
        )
        sign_source: SignSource = "explicit" if has_explicit_sign else "inferred"
        
        # CRITICAL: Force negative sign (even if OCR shows positive)
        amount_normalized = self._force_negative_sign(amount)
        
        return DiscountLineItem(
            label_text=normalized_item.raw_text,
            amount_raw=normalized_item.amount_raw,
            amount_normalized=amount_normalized,
            sign_source=sign_source,
            discount_type=discount_type,
            confidence=confidence,
            mode=mode,
            matched_keyword=matched_keyword
        )
    
    def process_line_items(
        self,
        normalized_items: List[NormalizedLineItem],
        mode: AnalysisMode = "QUOTE"
    ) -> Tuple[List[DiscountLineItem], DiscountTotals]:
        """
        Process all normalized line items and extract discounts.
        
        Args:
            normalized_items: All normalized line items from OCR
            mode: Analysis mode
        
        Returns:
            Tuple of (discount_items, discount_totals)
        """
        discounts = []
        
        for item in normalized_items:
            discount = self.detect_discount(item, mode)
            if discount:
                discounts.append(discount)
        
        # Calculate totals
        totals = self._calculate_totals(discounts)
        
        return discounts, totals
    
    def _calculate_totals(self, discounts: List[DiscountLineItem]) -> DiscountTotals:
        """Calculate aggregated discount totals"""
        totals = DiscountTotals()
        
        for discount in discounts:
            # Add to category-specific total
            if discount.discount_type == "UNCONDITIONAL":
                totals.total_unconditional += discount.amount_normalized
            elif discount.discount_type == "CONDITIONAL_FINANCE":
                totals.total_conditional_finance += discount.amount_normalized
            elif discount.discount_type == "CONDITIONAL_LEASE":
                totals.total_conditional_lease += discount.amount_normalized
            elif discount.discount_type == "DEALER_DISCOUNT":
                totals.total_dealer_discount += discount.amount_normalized
        
        # Calculate overall total (all negative)
        totals.total_all_discounts = (
            totals.total_unconditional +
            totals.total_conditional_finance +
            totals.total_conditional_lease +
            totals.total_dealer_discount
        )
        
        totals.count = len(discounts)
        
        return totals
    
    def validate_totals(
        self,
        discounts: List[DiscountLineItem],
        subtotal: float
    ) -> dict:
        """
        Validate that discounts reduce total correctly.
        
        Returns:
            {
                "valid": bool,
                "expected_reduction": float,
                "issues": List[str]
            }
        """
        issues = []
        
        # Check: All discounts must be negative
        for discount in discounts:
            if discount.amount_normalized > 0:
                issues.append(
                    f"Discount '{discount.label_text}' has positive amount: "
                    f"{discount.amount_normalized} (auto-corrected internally)"
                )
        
        # Calculate expected reduction
        totals = self._calculate_totals(discounts)
        expected_reduction = totals.total_all_discounts  # Negative value
        
        return {
            "valid": len(issues) == 0,
            "expected_reduction": expected_reduction,
            "issues": issues
        }