"""
Converts structured extracted JSON data (from OCR or external sources)
into the internal 'parsed' dict format that the scoring pipelines expect.

Supports:
  A) Nested structure: buyer_info, dealer_info, vehicle_details, financial_terms, etc.
  B) Flat structure: fields at root level with various naming conventions
  C) Mixed structure: any combination

The scoring pipeline expects flat keys like buyer_name, dealer_name,
selling_price, line_items, normalized_pricing, apr, term, trade, etc.
"""
from typing import Dict, Any, Optional, List


def _safe_float(val) -> Optional[float]:
    """Safely convert a value to float, return None on failure."""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            val = val.replace(",", "").replace("$", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    """Safely convert a value to int, return None on failure."""
    if val is None:
        return None
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError):
        return None


def _pick_first(data: dict, *keys):
    """Return the first non-None value from the given keys."""
    for k in keys:
        val = data.get(k)
        if val is not None:
            return val
    return None


def _flatten_nested(data: dict) -> dict:
    """Flatten common nested structures into a flat dict for easy lookup."""
    flat = dict(data)  # shallow copy of root

    nested_sections = [
        "buyer_info", "dealer_info", "vehicle_details", "vehicle",
        "financial_terms", "financials", "finance", "financial",
        "lease_specific", "lease_details", "lease",
        "trade_info", "trade_data",
        "extracted_text",
    ]

    def _merge_section(section: dict, allow_overwrite_dicts: bool = False) -> None:
        """Merge a section's scalar values into flat.
        If allow_overwrite_dicts=True, replace root-level dict collisions (e.g. flat['apr'] is
        a schema dict like {'listed':null} — the real numeric value from the OCR pipeline wins).
        """
        for k, v in section.items():
            existing = flat.get(k)
            if existing is None:
                flat[k] = v
            elif allow_overwrite_dicts and isinstance(existing, dict) and not isinstance(v, dict):
                # Root key is a schema object but OCR section has the actual scalar — prefer scalar
                flat[k] = v

    # Pass 1: root-level nested sections (no overwrite)
    for section_key in nested_sections:
        section = data.get(section_key)
        if isinstance(section, dict):
            _merge_section(section, allow_overwrite_dicts=False)

    # Pass 2: visionApiExtraction (and similar OCR wrapper keys).
    # Its sub-sections contain the real extracted scalars and should override
    # root-level dict collisions (e.g. root 'apr' = {"listed":null} vs
    # visionApiExtraction.financial_terms.apr = 0.0).
    for wrapper_key in ("visionApiExtraction", "vision_api_extraction",
                        "ocr_extraction", "extraction_result"):
        wrapper = data.get(wrapper_key)
        if isinstance(wrapper, dict):
            for section_key in nested_sections:
                section = wrapper.get(section_key)
                if isinstance(section, dict):
                    _merge_section(section, allow_overwrite_dicts=True)
            break  # only process the first wrapper found

    return flat


# Keys that are expected at the top level of a valid input payload.
# If NONE of these appear, the dict is likely a single-key wrapper (e.g.
# {"additionalProp1": { ...actual data... }}) and should be unwrapped.
_KNOWN_TOP_LEVEL_KEYS = {
    "buyer_name", "dealer_name", "buyer_info", "dealer_info",
    "vehicle_details", "vehicle", "financial_terms", "financials",
    "visionApiExtraction", "vision_api_extraction", "ocr_extraction",
    "red_flags", "green_flags", "blue_flags", "narrative",
    "score", "selling_price", "apr", "term", "trade",
    "vin_number", "vin", "email", "phone_number", "address",
    "state", "region", "badge", "quote_type", "normalized_pricing",
    "bundle_abuse", "addons_and_packages", "lease_specific",
    "extracted_text", "quality_assessment", "discount_incentive",
    "logo_text", "date", "buyer_message", "name",
}


def _unwrap_if_needed(data: dict) -> dict:
    """Detect and unwrap arbitrary single-key wrapper dicts.

    Handles inputs like {"additionalProp1": { ...real data... }} that are
    sent when the caller wraps the payload under a dynamic property name.
    If none of the known top-level keys are present but there are sub-dicts,
    merge all sub-dict values into a single flat dict.
    """
    if not data:
        return data
    # If any recognised key is present we're already at the right level
    if any(k in _KNOWN_TOP_LEVEL_KEYS for k in data):
        return data
    # No known keys — collect all values that are dicts (the wrapped payloads)
    dict_values = [v for v in data.values() if isinstance(v, dict)]
    if not dict_values:
        return data
    # Merge all wrapped dicts (most common case: exactly one wrapper key)
    merged: dict = {}
    for v in dict_values:
        merged.update(v)
    return merged


def convert_extracted_json_to_parsed(data: dict) -> dict:
    """
    Convert user-provided JSON data into the 'parsed' dict format
    that the rating/contract/lease scoring pipelines consume.

    Handles both nested and flat structures with multiple naming conventions.
    Also handles single-key wrapper payloads like {"additionalProp1": {...}}.
    """
    # Step 0: Unwrap arbitrary top-level wrappers (e.g. additionalProp1)
    data = _unwrap_if_needed(data)

    # Step 1: Flatten all nested sections for uniform access
    flat = _flatten_nested(data)

    parsed: Dict[str, Any] = {}

    # ─── Buyer Info ───
    # Access buyer_info section directly to avoid collision with dealer "name"
    buyer_section = data.get("buyer_info") or {}
    parsed["buyer_name"] = (
        _pick_first(flat, "buyer_name", "customer_name")
        or buyer_section.get("name")
        or buyer_section.get("buyer_name")
        or _pick_first(flat, "name", "buyer")
    )
    parsed["email"] = (
        _pick_first(flat, "email", "buyer_email", "customer_email")
        or buyer_section.get("email")
    )
    parsed["phone_number"] = (
        _pick_first(flat, "phone_number", "phone", "buyer_phone", "contact_phone")
        or buyer_section.get("phone")
        or buyer_section.get("phone_number")
    )
    parsed["address"] = (
        _pick_first(flat, "address", "buyer_address", "customer_address")
        or buyer_section.get("address")
    )

    # ─── Dealer Info ───
    dealer_section = data.get("dealer_info") or {}
    parsed["dealer_name"] = (
        _pick_first(flat, "dealer_name", "dealership", "dealership_name")
        or dealer_section.get("name")
        or dealer_section.get("dealer_name")
    )
    parsed["state"] = (
        _pick_first(flat, "state", "dealer_state")
        or dealer_section.get("state")
    )
    parsed["logo_text"] = parsed["dealer_name"]

    # Determine region from state
    us_states = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
        "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
        "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
    }
    state = (str(parsed.get("state") or "")).upper().strip()
    parsed["region"] = state if state in us_states else "Outside US"

    # ─── Vehicle Details ───
    vehicle_section = data.get("vehicle_details") or data.get("vehicle") or {}
    parsed["vin_number"] = (
        _pick_first(flat, "vin_number", "vin", "VIN")
        or vehicle_section.get("vin")
    )
    parsed["date"] = _pick_first(flat, "date", "purchase_date", "contract_date", "sale_date")

    msrp = _safe_float(
        _pick_first(flat, "msrp", "MSRP", "sticker_price")
        or vehicle_section.get("msrp")
    )
    selling_price = _safe_float(
        _pick_first(flat, "selling_price", "sale_price", "purchase_price",
                    "negotiated_price", "agreed_price", "total_price")
        or vehicle_section.get("sale_price")
        or vehicle_section.get("selling_price")
    )

    # Guard: if sale_price == amount_financed it was mis-extracted from the TILA box.
    # Fall back to itemization cash_price_including_accessories which is the true vehicle price.
    _amount_financed_check = _safe_float(
        _pick_first(flat, "amount_financed", "loan_amount")
        or (data.get("tila_disclosures") or {}).get("amount_financed")
        or (data.get("itemization_of_amount_financed") or {}).get("amount_financed")
    )
    _cash_price_fallback = _safe_float(
        _pick_first(flat, "cash_price", "cash_price_including_accessories")
        or (data.get("itemization_of_amount_financed") or {}).get("cash_price_including_accessories")
        or (data.get("financial_terms") or {}).get("cash_price")
    )
    if (
        selling_price is not None
        and _amount_financed_check is not None
        and abs(selling_price - _amount_financed_check) < 0.02
        and _cash_price_fallback is not None
        and _cash_price_fallback < _amount_financed_check
    ):
        # sale_price was incorrectly set to amount_financed — use the itemization cash price
        selling_price = _cash_price_fallback

    parsed["selling_price"] = selling_price
    parsed["sale_price"] = selling_price

    # ─── Financial Terms ───
    down_payment = _safe_float(_pick_first(flat, "down_payment", "cash_down", "down"))
    trade_in_value = _safe_float(_pick_first(flat, "trade_in_value", "trade_value", "trade_allowance"))
    amount_financed = _safe_float(_pick_first(flat, "amount_financed", "loan_amount", "finance_amount", "total_financed"))
    total_fees = _safe_float(_pick_first(flat, "total_fees", "fees"))
    total_taxes = _safe_float(_pick_first(flat, "total_taxes", "sales_tax", "tax", "taxes"))

    # Sum individual fees if total_fees not provided
    if total_fees is None:
        total_fees = _sum_fees(flat)

    # Supplement amount_financed from tila/itemization if still missing
    if amount_financed is None:
        _tila_af = data.get("tila_disclosures") or {}
        _item_af = data.get("itemization_of_amount_financed") or {}
        _ft_af = data.get("financial_terms") or {}
        amount_financed = _safe_float(
            _tila_af.get("amount_financed")
            or _item_af.get("amount_financed")
            or _ft_af.get("loan_amount")
        )

    # Supplement trade_in_value from trade_in section if missing
    if trade_in_value is None:
        _ti = data.get("trade_in") or {}
        trade_in_value = _safe_float(
            _ti.get("gross_trade_in")
            or _ti.get("net_trade_allowance")
        )

    # Supplement sales tax from financial_terms or fees_breakdown
    if total_taxes is None:
        _fb = data.get("fees_breakdown") or {}
        _ft2 = data.get("financial_terms") or {}
        total_taxes = _safe_float(
            _ft2.get("sales_tax")
            or _fb.get("sales_tax")
        )

    # Supplement doc_fee from fees_breakdown
    _doc_fee = _safe_float(
        _pick_first(flat, "doc_fee", "documentary_fee")
        or (data.get("fees_breakdown") or {}).get("documentary_fee")
        or (data.get("financial_terms") or {}).get("doc_fee")
    )

    parsed["normalized_pricing"] = {
        "msrp": msrp,
        "selling_price": selling_price,
        "discount": _safe_float(_pick_first(flat, "discount", "total_discount")),
        "rebate": _safe_float(_pick_first(flat, "rebate", "total_rebate",
                                         "manufacturers_rebate")
                   or (data.get("itemization_of_amount_financed") or {}).get("manufacturers_rebate")
                   or (data.get("financial_terms") or {}).get("manufacturers_rebate")),
        "down_payment": down_payment,
        "trade_in_value": trade_in_value,
        "amount_financed": amount_financed,
        "total_fees": total_fees,
        "total_taxes": total_taxes,
        "doc_fee": _doc_fee,
    }

    # ─── APR Data ───
    # Check tila_disclosures and financial_terms as additional sources
    _tila = data.get("tila_disclosures") or {}
    _ft = data.get("financial_terms") or {}
    _ps = data.get("payment_schedule") or {}
    apr_rate = _safe_float(
        _pick_first(flat, "apr", "interest_rate", "rate", "APR")
        or _tila.get("annual_percentage_rate")
        or _ft.get("apr")
    )
    money_factor = _safe_float(_pick_first(flat, "money_factor", "mf", "lease_rate"))
    parsed["apr"] = {
        "rate": apr_rate,
        "estimated": apr_rate is None,
        "money_factor": money_factor,
        "validation_status": "unvalidated",
        "payment_variance": None,
    }
    if money_factor is not None:
        parsed["money_factor"] = money_factor

    # ─── Term Data ───
    term_months = _safe_int(
        _pick_first(flat, "term_months", "term", "months", "loan_term", "lease_term")
        or _ps.get("number_of_payments")
        or _ft.get("term_months")
    )
    parsed["term"] = {
        "months": term_months,
    }

    # ─── Monthly Payment ───
    parsed["monthly_payment"] = _safe_float(
        _pick_first(flat, "monthly_payment", "payment_amount", "payment")
        or _ps.get("payment_amount")
        or _ft.get("monthly_payment")
    )

    # ─── Lease Specific ───
    cap_cost = _safe_float(_pick_first(flat, "cap_cost", "capitalized_cost", "gross_cap_cost"))
    residual_value = _safe_float(_pick_first(flat, "residual_value", "residual", "lease_end_value"))
    residual_percent = _safe_float(_pick_first(flat, "residual_percent", "residual_pct"))
    annual_miles = _safe_float(_pick_first(flat, "annual_miles", "miles_per_year", "mileage_allowance"))

    parsed["cap_cost"] = cap_cost
    parsed["residual_value"] = residual_value
    parsed["residual_percent"] = residual_percent
    parsed["annual_miles"] = annual_miles

    # Lender info (for captive lender detection)
    _lender_section = data.get("lender_info") or {}
    parsed["lessor_name"] = (
        _pick_first(flat, "lessor_name", "lessor", "lender_name", "lender", "finance_company")
        or _lender_section.get("lender_name")
    )
    parsed["lender_name"] = parsed["lessor_name"]

    # Rent charge for MF derivation
    parsed["total_rent_charge"] = _safe_float(_pick_first(flat, "total_rent_charge", "rent_charge"))
    parsed["net_cap_cost"] = _safe_float(_pick_first(flat, "net_cap_cost", "adjusted_cap_cost"))

    # MSD (Multiple Security Deposits)
    parsed["msd_count"] = _safe_int(_pick_first(flat, "msd_count"))
    parsed["msd_total"] = _safe_float(_pick_first(flat, "msd_total"))

    # ─── Trade Data ───
    trade_payoff = _safe_float(_pick_first(flat, "trade_payoff", "payoff_amount", "loan_payoff"))
    neg_equity = None

    if trade_in_value and trade_payoff and trade_payoff > trade_in_value:
        neg_equity = trade_payoff - trade_in_value

    if trade_in_value:
        equity = None
        if trade_payoff and trade_in_value > trade_payoff:
            equity = trade_in_value - trade_payoff
        parsed["trade"] = {
            "trade_allowance": trade_in_value,
            "trade_payoff": trade_payoff,
            "equity": equity,
            "negative_equity": neg_equity,
            "status": f"Negative equity of -${neg_equity:,.2f} detected" if neg_equity else f"Trade-in value: ${trade_in_value:,.2f}",
        }
        parsed["trade_allowance"] = trade_in_value
        parsed["trade_payoff"] = trade_payoff
        parsed["negative_equity"] = neg_equity
    else:
        # Check if trade data is passed as a dict at root level
        trade_obj = data.get("trade") or data.get("trade_data") or data.get("trade_info")
        if isinstance(trade_obj, dict) and any(trade_obj.get(k) for k in ("trade_allowance", "trade_in_value", "trade_value")):
            # IMPORTANT: do not use generic "allowance" key here.
            # Some payloads use "allowance" for down payment or unrelated incentives.
            ta = _safe_float(_pick_first(trade_obj, "trade_allowance", "trade_in_value", "trade_value"))
            tp = _safe_float(_pick_first(trade_obj, "trade_payoff", "payoff", "payoff_amount"))
            eq = _safe_float(trade_obj.get("equity"))
            ne = _safe_float(trade_obj.get("negative_equity"))
            if ta and tp and not ne and tp > ta:
                ne = tp - ta
            parsed["trade"] = {
                "trade_allowance": ta,
                "trade_payoff": tp,
                "equity": eq,
                "negative_equity": ne,
                "status": trade_obj.get("status", "Trade identified"),
            }
            parsed["trade_allowance"] = ta
            parsed["trade_payoff"] = tp
            parsed["negative_equity"] = ne
        else:
            parsed["trade"] = {}

    # ─── Line Items / Addons ───
    line_items = _extract_line_items(data, flat)
    parsed["line_items"] = line_items

    # ─── Extracted Text (for trade detection heuristics) ───
    extracted = data.get("extracted_text") or {}
    if isinstance(extracted, dict):
        parsed["raw_text"] = extracted.get("raw_text", "")
        sections = extracted.get("sections") or {}
        parsed["ocr_text"] = " ".join(str(v) for v in sections.values() if v)
    elif isinstance(extracted, str):
        parsed["raw_text"] = extracted
        parsed["ocr_text"] = extracted
    else:
        parsed["raw_text"] = ""
        parsed["ocr_text"] = ""

    # ─── Metadata ───
    parsed["quote_type"] = _pick_first(flat, "quote_type", "document_type") or "Audit"

    # ─── Audit Status + Replay Identifiers ───
    parsed["audit_status"] = (
        _pick_first(flat, "audit_status", "auditStatus", "status")
        or "COMPLETE"
    )
    replay_keys = [
        "ocr_pipeline_version",
        "rules_hash",
        "prompt_hash",
        "schema_version",
        "scoring_engine_version",
        "model_version",
        "ocr_provider",
    ]
    parsed["replay_context"] = {
        k: _pick_first(flat, k)
        for k in replay_keys
        if _pick_first(flat, k) is not None
    }

    # ─── Flags ───
    # If the input already contains pre-computed flags (from a prior OCR/AI analysis),
    # carry them through so the scoring pipeline uses them directly.
    # Otherwise start empty — the Python audit pipeline will generate them.
    def _has_real_flags(raw) -> bool:
        return isinstance(raw, list) and len(raw) > 0 and any(
            isinstance(f, dict) and (f.get("type") or f.get("message")) for f in raw
        )

    in_red   = data.get("red_flags",   [])
    in_green = data.get("green_flags", [])
    in_blue  = data.get("blue_flags",  [])

    parsed["red_flags"]   = in_red   if _has_real_flags(in_red)   else []
    parsed["green_flags"] = in_green if _has_real_flags(in_green) else []
    parsed["blue_flags"]  = in_blue  if _has_real_flags(in_blue)  else []

    # Signal to the scoring pipeline: pre-existing flags are present, skip Python audit merge
    parsed["has_precomputed_flags"] = (
        _has_real_flags(in_red) or _has_real_flags(in_green) or _has_real_flags(in_blue)
    )

    # Schema D / upstream flags (flag_id-driven)
    upstream_flags = data.get("flags") or data.get("active_flags") or data.get("audit_flags")
    parsed["flags"] = upstream_flags if isinstance(upstream_flags, list) else []

    # Signal: raw OCR extraction wrapper was present — AI should always re-analyze regardless of flags
    parsed["has_vision_extraction"] = bool(
        data.get("visionApiExtraction") or data.get("vision_api_extraction")
        or data.get("ocr_extraction") or data.get("extraction_result")
    )

    # ─── Score placeholder (scoring pipeline recalculates) ───
    parsed["score"] = 75.0

    # ─── Empty narrative (scoring pipeline or AI will generate) ───
    parsed["narrative"] = {}

    # ─── Buyer message ───
    parsed["buyer_message"] = "Contract data analysis completed"

    # ─── Bundle abuse ───
    parsed["bundle_abuse"] = {"active": False, "deduction": 0}

    return parsed


def _extract_line_items(data: dict, flat: dict) -> List[dict]:
    """
    Extract line items from various possible locations and formats.
    """
    items = []

    # Source 1: addons_and_packages (nested format)
    addons = data.get("addons_and_packages") or data.get("addons") or data.get("packages") or []
    if isinstance(addons, list):
        for addon in addons:
            if isinstance(addon, dict):
                desc = addon.get("name") or addon.get("description") or addon.get("item") or ""
                amt = str(addon.get("price") or addon.get("amount") or addon.get("cost") or "0")
                if desc:
                    items.append({
                        "description": desc,
                        "amount": amt,
                        "item": desc,
                        "category": addon.get("category", "Other"),
                    })
            elif isinstance(addon, str):
                items.append({"description": addon, "amount": "0", "item": addon})

    # Source 2: line_items at root level (already formatted)
    raw_items = data.get("line_items") or data.get("items") or []
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                desc = item.get("description") or item.get("name") or item.get("item") or ""
                amt = str(item.get("amount") or item.get("price") or item.get("cost") or "0")
                if desc:
                    items.append({
                        "description": desc,
                        "amount": amt,
                        "item": desc,
                        "category": item.get("category", "Other"),
                    })

    # Source 3: Individual fee fields from financial terms
    fee_fields = {
        "doc_fee": "Documentary Fee",
        "documentation_fee": "Documentary Fee",
        "title_fee": "Title Fee",
        "registration_fee": "Registration Fee",
        "license_fee": "License Fee",
        "acquisition_fee": "Acquisition Fee",
        "disposition_fee": "Disposition Fee",
        "bank_fee": "Bank Fee",
    }
    for field_key, desc in fee_fields.items():
        val = flat.get(field_key)
        if val is not None:
            # Don't duplicate if already in items
            if not any(desc.lower() in (i.get("description", "").lower()) for i in items):
                items.append({"description": desc, "amount": str(val)})

    # Source 4: Product fields (GAP, VSC, etc.)
    product_fields = {
        "gap_price": "GAP Insurance",
        "gap_amount": "GAP Insurance",
        "gap_cost": "GAP Insurance",
        "vsc_price": "Vehicle Service Contract (VSC)",
        "vsc_amount": "Vehicle Service Contract (VSC)",
        "vsc_cost": "Vehicle Service Contract (VSC)",
        "maintenance_price": "Prepaid Maintenance",
        "maintenance_amount": "Prepaid Maintenance",
        "maintenance_cost": "Prepaid Maintenance",
        "extended_warranty": "Extended Warranty",
        "extended_warranty_price": "Extended Warranty",
        "paint_protection": "Paint Protection",
        "tire_wheel": "Tire & Wheel Protection",
        "key_replacement": "Key Replacement",
        "appearance_package": "Appearance Package",
    }
    for field_key, desc in product_fields.items():
        val = flat.get(field_key)
        if val is not None:
            if not any(desc.lower() in (i.get("description", "").lower()) for i in items):
                items.append({"description": desc, "amount": str(val)})

    return items


def _sum_fees(flat: dict) -> Optional[float]:
    """Sum up individual fee fields into total_fees."""
    fees = []
    for key in ["doc_fee", "documentation_fee", "title_fee", "registration_fee",
                 "license_fee", "acquisition_fee"]:
        val = flat.get(key)
        if val is not None:
            try:
                fees.append(float(str(val).replace(",", "").replace("$", "")))
            except (ValueError, TypeError):
                pass
    return sum(fees) if fees else None
