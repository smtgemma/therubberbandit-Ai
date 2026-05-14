import json
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .ocr_normalizer import OCRNormalizer

RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")


@dataclass
class FlagDefinition:
    flag_id: str
    group: str
    points: int
    severity: str
    modes: List[str]
    description: str
    scoring_eligible: bool


@dataclass
class ActiveFlag:
    flag_id: str
    group: str
    points: int
    severity: str
    modes: List[str]
    description: str
    scoring_eligible: bool
    confidence: float
    source: str
    message: str
    adjusted_points: float
    suppressed: bool = False


@dataclass
class ScoringResult:
    score: float
    score_int: int
    flags: List[ActiveFlag]
    suppressed_ids: List[str]
    trust_score_delta: float
    trust_score_skipped: bool
    rules_hash: str
    audit_status: str
    eligible_for_scoring: bool


class RuleSet:
    def __init__(self, flag_registry: Dict[str, FlagDefinition], suppression: dict, pricing_caps: dict, doc_fee_caps: dict, rules_hash: str):
        self.flag_registry = flag_registry
        self.suppression = suppression
        self.pricing_caps = pricing_caps
        self.doc_fee_caps = doc_fee_caps
        self.rules_hash = rules_hash


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rules() -> RuleSet:
    paths = {
        "flag_registry": os.path.join(RULES_DIR, "flag_registry.json"),
        "suppression": os.path.join(RULES_DIR, "suppression_pairs.json"),
        "pricing_caps": os.path.join(RULES_DIR, "pricing_caps.json"),
        "doc_fee_caps": os.path.join(RULES_DIR, "doc_fee_state_rules.json"),
    }

    contents = []
    for path in paths.values():
        with open(path, "rb") as f:
            contents.append(f.read())

    rules_hash = hashlib.sha256(b"".join(contents)).hexdigest()

    raw_flags = _read_json(paths["flag_registry"]).get("flags", [])
    registry: Dict[str, FlagDefinition] = {}
    for item in raw_flags:
        flag_id = str(item.get("id", "")).strip().upper()
        if not flag_id:
            continue
        registry[flag_id] = FlagDefinition(
            flag_id=flag_id,
            group=str(item.get("group", "")),
            points=int(item.get("points", 0)),
            severity=str(item.get("severity", "")),
            modes=list(item.get("modes", [])),
            description=str(item.get("description", "")),
            scoring_eligible=bool(item.get("scoring_eligible", False)),
        )

    suppression = _read_json(paths["suppression"])
    pricing_caps = _read_json(paths["pricing_caps"])
    doc_fee_caps = _read_json(paths["doc_fee_caps"])

    return RuleSet(registry, suppression, pricing_caps, doc_fee_caps, rules_hash)


def _confidence_modifier(confidence: float) -> float:
    if confidence >= 0.75:
        return 1.0
    if confidence >= 0.40:
        return 0.9
    return 0.8


def _clamp_confidence(value: Optional[float]) -> float:
    if value is None:
        return 1.0
    try:
        val = float(value)
    except (ValueError, TypeError):
        return 1.0
    return max(0.0, min(1.0, val))


def _extract_flag_id(raw: dict) -> Optional[str]:
    for key in ("flag_id", "id", "flag", "name"):
        value = raw.get(key)
        if value:
            return str(value).strip().upper()
    return None


def build_active_flags(flags_payload: List[dict], registry: Dict[str, FlagDefinition], source: str) -> List[ActiveFlag]:
    active_flags: List[ActiveFlag] = []
    for raw in flags_payload:
        if not isinstance(raw, dict):
            continue
        flag_id = _extract_flag_id(raw)
        if not flag_id or flag_id not in registry:
            continue
        definition = registry[flag_id]
        confidence = _clamp_confidence(raw.get("confidence") or raw.get("confidence_score"))
        message = raw.get("message") or definition.description
        points = definition.points
        adjusted = points
        if points < 0:
            adjusted = points * _confidence_modifier(confidence)
            if adjusted < -15:
                adjusted = -15.0
        active_flags.append(ActiveFlag(
            flag_id=flag_id,
            group=definition.group,
            points=points,
            severity=definition.severity,
            modes=definition.modes,
            description=definition.description,
            scoring_eligible=definition.scoring_eligible,
            confidence=confidence,
            source=source,
            message=str(message),
            adjusted_points=float(adjusted),
        ))
    return active_flags


def _make_flag(flag_id: str, rules: RuleSet, confidence: float = 1.0, message: Optional[str] = None, source: str = "computed") -> Optional[ActiveFlag]:
    definition = rules.flag_registry.get(flag_id)
    if not definition:
        return None
    points = definition.points
    adjusted = points
    if points < 0:
        adjusted = points * _confidence_modifier(confidence)
        if adjusted < -15:
            adjusted = -15.0
    return ActiveFlag(
        flag_id=definition.flag_id,
        group=definition.group,
        points=points,
        severity=definition.severity,
        modes=definition.modes,
        description=definition.description,
        scoring_eligible=definition.scoring_eligible,
        confidence=confidence,
        source=source,
        message=message or definition.description,
        adjusted_points=float(adjusted),
    )


def _safe_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return None


def _normalize_line_items(parsed: dict) -> List[dict]:
    line_items = parsed.get("line_items", [])
    if not isinstance(line_items, list):
        return []
    normalizer = OCRNormalizer()
    normalized = []
    for item in line_items:
        if not isinstance(item, dict):
            continue
        raw_text = item.get("description") or item.get("item") or item.get("name") or ""
        amount_raw = str(item.get("amount") or item.get("price") or item.get("cost") or "0")
        if not raw_text:
            continue
        normalized_item = normalizer.normalize_line_item(raw_text=raw_text, amount_raw=amount_raw)
        normalized.append(normalized_item)
    return normalized


def compute_flags_from_parsed(parsed: dict, rules: RuleSet, mode: str) -> List[ActiveFlag]:
    flags: List[ActiveFlag] = []
    mode = (mode or "").upper()

    msrp = _safe_float((parsed.get("normalized_pricing") or {}).get("msrp"))
    selling_price = _safe_float(parsed.get("selling_price"))
    if msrp is None and selling_price is not None:
        new_used_flag = _make_flag("NEW_USED_NOT_CONFIRMED", rules, source="system")
        if new_used_flag:
            flags.append(new_used_flag)
        msrp = selling_price

    doc_fee = _safe_float((parsed.get("normalized_pricing") or {}).get("doc_fee"))
    if doc_fee is not None:
        state = str(parsed.get("state") or "").upper().strip()
        caps = rules.doc_fee_caps.get("states", {})
        benchmark = _safe_float(rules.doc_fee_caps.get("benchmark_default"))
        cap = None
        if state in caps:
            cap = _safe_float(caps[state].get("cap"))
        else:
            no_cap_flag = _make_flag("NO_CAP", rules, source="system")
            if no_cap_flag:
                flags.append(no_cap_flag)
        if cap is not None and doc_fee > cap:
            excessive = _make_flag("DOC_FEE_EXCESSIVE", rules)
            if excessive:
                flags.append(excessive)
        else:
            if benchmark is not None and doc_fee > benchmark:
                elevated = _make_flag("DOC_FEE_ELEVATED", rules)
                if elevated:
                    flags.append(elevated)
            else:
                within = _make_flag("DOC_FEE_WITHIN_CAP", rules)
                if within:
                    flags.append(within)

    normalized_items = _normalize_line_items(parsed)
    gap_total = 0.0
    vsc_total = 0.0
    maintenance_total = 0.0
    addon_total = 0.0
    dca_present = False

    for item in normalized_items:
        amount = abs(float(item.amount_normalized))
        category = str(item.normalized_category or "")
        raw = str(item.raw_text or "").lower()
        if category == "GAP":
            gap_total += amount
            if "debt cancellation" in raw or "dca" in raw:
                dca_present = True
        elif category == "VSC":
            vsc_total += amount
        elif category == "MAINTENANCE":
            maintenance_total += amount
        elif category == "ADDON_PACKAGE":
            addon_total += amount

    # GAP / DCA caps
    if gap_total > 0 and msrp is not None:
        gap_rules = rules.pricing_caps.get("gap_dca", {})
        msrp_threshold = _safe_float(gap_rules.get("msrp_threshold")) or 60000.0
        cap_under = _safe_float(gap_rules.get("cap_under_threshold")) or 1200.0
        cap_over = _safe_float(gap_rules.get("cap_over_threshold")) or 1500.0
        percent_under = _safe_float(gap_rules.get("percent_under_threshold")) or 0.03

        if msrp < msrp_threshold:
            gap_cap = min(cap_under, msrp * percent_under)
        else:
            gap_cap = cap_over

        if gap_total > gap_cap:
            overpriced_id = "DEBT_CANCELLATION_OVERPRICED" if dca_present else "GAP_OVERPRICED"
            overpriced = _make_flag(overpriced_id, rules)
            if overpriced:
                flags.append(overpriced)
        else:
            within = _make_flag("GAP_WITHIN_CAP", rules)
            if within:
                flags.append(within)
        if dca_present:
            info = _make_flag("DEBT_CANCELLATION_DETECTED", rules, source="informational")
            if info:
                flags.append(info)

    # VSC caps
    if vsc_total > 0 and msrp is not None:
        vsc_rules = rules.pricing_caps.get("vsc", {})
        msrp_threshold = _safe_float(vsc_rules.get("msrp_threshold")) or 45000.0
        new_under = vsc_rules.get("new_under", {})
        used_under = vsc_rules.get("used_under", {})
        new_over = vsc_rules.get("new_over", {})
        used_over = vsc_rules.get("used_over", {})
        cap_new = None
        cap_used = None
        if msrp <= msrp_threshold:
            cap_new = min(msrp * (_safe_float(new_under.get("percent")) or 0.15), _safe_float(new_under.get("cap")) or 4000.0)
            cap_used = min(msrp * (_safe_float(used_under.get("percent")) or 0.16), _safe_float(used_under.get("cap")) or 6000.0)
        else:
            cap_new = msrp * (_safe_float(new_over.get("percent")) or 0.15)
            cap_used = msrp * (_safe_float(used_over.get("percent")) or 0.16)

        cap = min(cap_new, cap_used)

        mileage = _safe_float(parsed.get("mileage") or parsed.get("odometer") or parsed.get("vehicle_mileage"))
        high_mileage = vsc_rules.get("high_mileage", {})
        if mileage is not None and mileage >= (high_mileage.get("miles_min") or 100001):
            cap = msrp * (_safe_float(high_mileage.get("percent")) or 0.17)

        if vsc_total > cap:
            overpriced = _make_flag("VSC_OVERPRICED", rules)
            if overpriced:
                flags.append(overpriced)
        else:
            within = _make_flag("VSC_WITHIN_CAP", rules)
            if within:
                flags.append(within)

    # Maintenance caps
    if maintenance_total > 0 and msrp is not None:
        maint_rules = rules.pricing_caps.get("maintenance", {})
        percent = _safe_float(maint_rules.get("percent")) or 0.05
        cap = _safe_float(maint_rules.get("cap")) or 1500.0
        maintenance_cap = min(cap, msrp * percent)
        if maintenance_total > maintenance_cap:
            overpriced = _make_flag("MAINTENANCE_OVERPRICED", rules)
            if overpriced:
                flags.append(overpriced)
        else:
            within = _make_flag("MAINTENANCE_WITHIN_CAP", rules)
            if within:
                flags.append(within)

    # Add-on combined cap (front-end)
    add_on_rules = rules.pricing_caps.get("add_ons", {})
    add_on_cap = _safe_float(add_on_rules.get("combined_cap"))
    if add_on_cap is not None and addon_total > add_on_cap:
        cap_exceeded = _make_flag("ADDON_CAP_EXCEEDED", rules)
        if cap_exceeded:
            flags.append(cap_exceeded)

    # Term-based structural flags
    term_months = _safe_float((parsed.get("term") or {}).get("months"))
    if term_months is not None:
        if term_months >= 84:
            high_term = _make_flag("HIGH_RISK_TERM", rules)
            if high_term:
                flags.append(high_term)
        elif 73 <= term_months <= 83:
            extended = _make_flag("EXTENDED_TERM", rules)
            if extended:
                flags.append(extended)

    # Negative equity disclosed
    neg_equity = _safe_float((parsed.get("trade") or {}).get("negative_equity"))
    if neg_equity is not None and neg_equity > 0:
        neg_flag = _make_flag("NEGATIVE_EQUITY_DISCLOSED", rules)
        if neg_flag:
            flags.append(neg_flag)

    # Lease mileage program flags
    if mode == "LEASE":
        annual_miles = _safe_float(parsed.get("annual_miles"))
        if annual_miles is not None:
            if annual_miles < 10000:
                low_miles = _make_flag("LOW_MILEAGE_ALLOWANCE", rules)
                if low_miles:
                    flags.append(low_miles)
            elif 10000 <= annual_miles <= 12000:
                standard = _make_flag("STANDARD_MILEAGE_PROGRAM", rules)
                if standard:
                    flags.append(standard)

    return flags


def apply_suppression(flags: List[ActiveFlag], suppression_rules: dict) -> List[str]:
    by_id = {f.flag_id: f for f in flags}
    suppressed: List[str] = []

    for rule in suppression_rules.get("pairs", []):
        trigger = str(rule.get("if_active", "")).upper()
        if not trigger or trigger not in by_id:
            continue
        trigger_flag = by_id[trigger]
        if trigger_flag.suppressed:
            continue
        for target in rule.get("suppress", []):
            target_id = str(target).upper()
            if target_id in by_id:
                by_id[target_id].suppressed = True
                suppressed.append(target_id)

    backend_rule = suppression_rules.get("backend_overload") or {}
    backend_flag_id = str(backend_rule.get("flag", "")).upper()
    compare_ids = [str(fid).upper() for fid in backend_rule.get("compare_to_sum_of", [])]
    if backend_flag_id in by_id:
        backend_flag = by_id[backend_flag_id]
        if not backend_flag.suppressed:
            total = 0.0
            for fid in compare_ids:
                if fid in by_id and not by_id[fid].suppressed:
                    total += abs(by_id[fid].adjusted_points)
            if total > abs(backend_flag.adjusted_points):
                backend_flag.suppressed = True
                suppressed.append(backend_flag_id)
            else:
                for fid in compare_ids:
                    if fid in by_id and not by_id[fid].suppressed:
                        by_id[fid].suppressed = True
                        suppressed.append(fid)

    return suppressed


def compute_trust_score_delta(flags: List[ActiveFlag]) -> Tuple[float, bool]:
    if any(f.flag_id == "DUPLICATE_SUBMISSION_DETECTED" for f in flags):
        return 0.0, True

    weights = {
        "DEALER_CONDUCT": 1.0,
        "STRUCTURAL_RISK": 0.4,
        "POSITIVE": 1.0,
    }

    total = 0.0
    for flag in flags:
        if flag.suppressed or not flag.scoring_eligible:
            continue
        weight = weights.get(flag.group, 0.0)
        total += flag.adjusted_points * weight
    return total, False


def score_flags(
    flags: List[ActiveFlag],
    rules: RuleSet,
    audit_status: str
) -> ScoringResult:
    eligible = audit_status in ("COMPLETE", "HUMAN_REVIEW_RECOMMENDED")
    if not eligible:
        return ScoringResult(
            score=0.0,
            score_int=0,
            flags=flags,
            suppressed_ids=[],
            trust_score_delta=0.0,
            trust_score_skipped=True,
            rules_hash=rules.rules_hash,
            audit_status=audit_status,
            eligible_for_scoring=False,
        )

    suppressed_ids = apply_suppression(flags, rules.suppression)
    total = 100.0
    for flag in flags:
        if flag.suppressed or not flag.scoring_eligible:
            continue
        total += flag.adjusted_points

    total = max(0.0, min(100.0, total))
    score_int = int(round(total))

    trust_delta, trust_skipped = compute_trust_score_delta(flags)

    return ScoringResult(
        score=total,
        score_int=score_int,
        flags=flags,
        suppressed_ids=suppressed_ids,
        trust_score_delta=trust_delta,
        trust_score_skipped=trust_skipped,
        rules_hash=rules.rules_hash,
        audit_status=audit_status,
        eligible_for_scoring=True,
    )
