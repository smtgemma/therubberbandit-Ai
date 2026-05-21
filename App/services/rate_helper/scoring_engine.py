import json
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .ocr_normalizer import OCRNormalizer

RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")
BASE_SCORE = 100.0
SCORE_FLOOR = 0.0
SCORE_CEILING = 100.0
MAX_FLAG_DEDUCTION = -15.0
VALID_AUDIT_STATUSES = {
    "COMPLETE",
    "HUMAN_REVIEW_RECOMMENDED",
    "INCOMPLETE",
    "CONFLICT_HOLD",
    "RULE_LOAD_FAILURE",
    "INVALID_REPLAY_CONTEXT",
}
SCORING_ELIGIBLE_STATUSES = {"COMPLETE", "HUMAN_REVIEW_RECOMMENDED"}


class RuleLoadError(RuntimeError):
    """Raised when locked scoring rules are missing or malformed."""


class InvalidAuditStatusError(ValueError):
    """Raised when an audit status is outside the locked enum."""


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

    def require_pricing_cap(self, path: Iterable[str]) -> Any:
        return _require_path(self.pricing_caps, path, "pricing_caps")

    def require_doc_fee_rule(self, path: Iterable[str]) -> Any:
        return _require_path(self.doc_fee_caps, path, "doc_fee_state_rules")


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuleLoadError(f"RULE_LOAD_FAILURE: cannot load {path}") from exc
    if not isinstance(data, dict):
        raise RuleLoadError(f"RULE_LOAD_FAILURE: {path} must contain a JSON object")
    return data


def _require_path(data: dict, path: Iterable[str], source: str) -> Any:
    current: Any = data
    traversed: List[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(traversed)
            raise RuleLoadError(f"RULE_LOAD_FAILURE: missing {source}.{joined}")
        current = current[key]
    return current


def _require_float(value: Any, label: str) -> float:
    result = _safe_float(value)
    if result is None:
        raise RuleLoadError(f"RULE_LOAD_FAILURE: {label} must be numeric")
    return result


def load_rules() -> RuleSet:
    paths = {
        "flag_registry": os.path.join(RULES_DIR, "flag_registry.json"),
        "suppression": os.path.join(RULES_DIR, "suppression_pairs.json"),
        "pricing_caps": os.path.join(RULES_DIR, "pricing_caps.json"),
        "doc_fee_caps": os.path.join(RULES_DIR, "doc_fee_state_rules.json"),
    }

    contents = []
    for path in paths.values():
        try:
            with open(path, "rb") as f:
                contents.append(f.read())
        except OSError as exc:
            raise RuleLoadError(f"RULE_LOAD_FAILURE: cannot load {path}") from exc

    rules_hash = hashlib.sha256(b"".join(contents)).hexdigest()

    raw_flags = _read_json(paths["flag_registry"]).get("flags", [])
    if not isinstance(raw_flags, list):
        raise RuleLoadError("RULE_LOAD_FAILURE: flag_registry.flags must be a list")
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
    _validate_pricing_caps(pricing_caps)
    _validate_doc_fee_caps(doc_fee_caps)

    return RuleSet(registry, suppression, pricing_caps, doc_fee_caps, rules_hash)


def _validate_pricing_caps(pricing_caps: dict) -> None:
    required_paths = [
        ("gap_dca", "msrp_threshold"),
        ("gap_dca", "cap_under_threshold"),
        ("gap_dca", "cap_over_threshold"),
        ("gap_dca", "percent_under_threshold"),
        ("vsc", "msrp_threshold"),
        ("vsc", "new_under", "percent"),
        ("vsc", "new_under", "cap"),
        ("vsc", "used_under", "percent"),
        ("vsc", "used_under", "cap"),
        ("vsc", "new_over", "percent"),
        ("vsc", "used_over", "percent"),
        ("vsc", "high_mileage", "miles_min"),
        ("vsc", "high_mileage", "percent"),
        ("maintenance", "percent"),
        ("maintenance", "cap"),
        ("add_ons", "combined_cap"),
        ("backend_total", "max_percent_of_msrp"),
        ("term", "extended_min_months"),
        ("term", "extended_max_months"),
        ("term", "high_risk_min_months"),
        ("lease_mileage", "standard_min"),
        ("lease_mileage", "standard_max"),
    ]
    for path in required_paths:
        value = _require_path(pricing_caps, path, "pricing_caps")
        _require_float(value, f"pricing_caps.{'.'.join(path)}")


def _validate_doc_fee_caps(doc_fee_caps: dict) -> None:
    _require_float(_require_path(doc_fee_caps, ("benchmark_default",), "doc_fee_state_rules"), "doc_fee_state_rules.benchmark_default")
    states = _require_path(doc_fee_caps, ("states",), "doc_fee_state_rules")
    if not isinstance(states, dict):
        raise RuleLoadError("RULE_LOAD_FAILURE: doc_fee_state_rules.states must be an object")


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
            if adjusted < MAX_FLAG_DEDUCTION:
                adjusted = MAX_FLAG_DEDUCTION
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
        if adjusted < MAX_FLAG_DEDUCTION:
            adjusted = MAX_FLAG_DEDUCTION
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
        caps = rules.require_doc_fee_rule(("states",))
        benchmark = _require_float(
            rules.require_doc_fee_rule(("benchmark_default",)),
            "doc_fee_state_rules.benchmark_default",
        )
        cap = None
        if state in caps:
            cap = _require_float(
                _require_path(caps[state], ("cap",), f"doc_fee_state_rules.states.{state}"),
                f"doc_fee_state_rules.states.{state}.cap",
            )
        else:
            no_cap_flag = _make_flag("NO_CAP", rules, source="system")
            if no_cap_flag:
                flags.append(no_cap_flag)
        if cap is not None and doc_fee > cap:
            excessive = _make_flag("DOC_FEE_ABOVE_STATE_CAP", rules)
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
    tire_wheel_total = 0.0
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
        elif category == "TIRE_WHEEL_PROTECTION":
            tire_wheel_total += amount
        elif category == "ADDON_PACKAGE":
            addon_total += amount

    backend_total = gap_total + vsc_total + maintenance_total + tire_wheel_total

    # GAP / DCA caps
    if gap_total > 0 and msrp is not None:
        msrp_threshold = _require_float(rules.require_pricing_cap(("gap_dca", "msrp_threshold")), "pricing_caps.gap_dca.msrp_threshold")
        cap_under = _require_float(rules.require_pricing_cap(("gap_dca", "cap_under_threshold")), "pricing_caps.gap_dca.cap_under_threshold")
        cap_over = _require_float(rules.require_pricing_cap(("gap_dca", "cap_over_threshold")), "pricing_caps.gap_dca.cap_over_threshold")
        percent_under = _require_float(rules.require_pricing_cap(("gap_dca", "percent_under_threshold")), "pricing_caps.gap_dca.percent_under_threshold")

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
        msrp_threshold = _require_float(rules.require_pricing_cap(("vsc", "msrp_threshold")), "pricing_caps.vsc.msrp_threshold")
        vehicle_condition = _vehicle_condition(parsed)
        if msrp <= msrp_threshold:
            section = "used_under" if vehicle_condition == "USED" else "new_under"
            percent = _require_float(rules.require_pricing_cap(("vsc", section, "percent")), f"pricing_caps.vsc.{section}.percent")
            cap_limit = _require_float(rules.require_pricing_cap(("vsc", section, "cap")), f"pricing_caps.vsc.{section}.cap")
            cap = min(msrp * percent, cap_limit)
        else:
            section = "used_over" if vehicle_condition == "USED" else "new_over"
            percent = _require_float(rules.require_pricing_cap(("vsc", section, "percent")), f"pricing_caps.vsc.{section}.percent")
            cap = msrp * percent

        mileage = _safe_float(parsed.get("mileage") or parsed.get("odometer") or parsed.get("vehicle_mileage"))
        high_mileage_min = _require_float(rules.require_pricing_cap(("vsc", "high_mileage", "miles_min")), "pricing_caps.vsc.high_mileage.miles_min")
        if mileage is not None and mileage >= high_mileage_min:
            high_mileage_percent = _require_float(rules.require_pricing_cap(("vsc", "high_mileage", "percent")), "pricing_caps.vsc.high_mileage.percent")
            cap = msrp * high_mileage_percent

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
        percent = _require_float(rules.require_pricing_cap(("maintenance", "percent")), "pricing_caps.maintenance.percent")
        cap = _require_float(rules.require_pricing_cap(("maintenance", "cap")), "pricing_caps.maintenance.cap")
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
    add_on_cap = _require_float(rules.require_pricing_cap(("add_ons", "combined_cap")), "pricing_caps.add_ons.combined_cap")
    if addon_total > add_on_cap:
        cap_exceeded = _make_flag("ADDON_CAP_EXCEEDED", rules)
        if cap_exceeded:
            flags.append(cap_exceeded)

    if backend_total > 0 and msrp is not None:
        backend_percent = _require_float(
            rules.require_pricing_cap(("backend_total", "max_percent_of_msrp")),
            "pricing_caps.backend_total.max_percent_of_msrp",
        )
        if backend_total / msrp > backend_percent:
            overloaded = _make_flag("BACKEND_OVERLOAD_DETECTED", rules)
            if overloaded:
                flags.append(overloaded)
        else:
            within = _make_flag("BACKEND_WITHIN_THRESHOLD", rules)
            if within:
                flags.append(within)

    # Term-based structural flags
    term_months = _safe_float((parsed.get("term") or {}).get("months"))
    if term_months is not None:
        high_risk_min = _require_float(rules.require_pricing_cap(("term", "high_risk_min_months")), "pricing_caps.term.high_risk_min_months")
        extended_min = _require_float(rules.require_pricing_cap(("term", "extended_min_months")), "pricing_caps.term.extended_min_months")
        extended_max = _require_float(rules.require_pricing_cap(("term", "extended_max_months")), "pricing_caps.term.extended_max_months")
        if term_months >= high_risk_min:
            high_term = _make_flag("HIGH_RISK_TERM", rules)
            if high_term:
                flags.append(high_term)
        elif extended_min <= term_months <= extended_max:
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
            standard_min = _require_float(rules.require_pricing_cap(("lease_mileage", "standard_min")), "pricing_caps.lease_mileage.standard_min")
            standard_max = _require_float(rules.require_pricing_cap(("lease_mileage", "standard_max")), "pricing_caps.lease_mileage.standard_max")
            if annual_miles < standard_min:
                low_miles = _make_flag("MILEAGE_BELOW_STANDARD", rules)
                if low_miles:
                    flags.append(low_miles)
            elif standard_min <= annual_miles <= standard_max:
                standard = _make_flag("STANDARD_MILEAGE_PROGRAM", rules)
                if standard:
                    flags.append(standard)

    return flags


def _vehicle_condition(parsed: dict) -> str:
    for key in ("vehicle_condition", "condition", "new_used", "vehicle_status"):
        value = parsed.get(key)
        if value:
            text = str(value).upper()
            if "USED" in text or "PRE-OWNED" in text or "PREOWNED" in text:
                return "USED"
            if "NEW" in text:
                return "NEW"
    return "NEW"


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
    audit_status = str(audit_status or "").upper()
    if audit_status not in VALID_AUDIT_STATUSES:
        raise InvalidAuditStatusError(f"Invalid audit_status: {audit_status}")

    if audit_status == "HUMAN_REVIEW_RECOMMENDED":
        for flag in flags:
            if flag.points < 0:
                flag.confidence = min(flag.confidence, 0.39)
                flag.adjusted_points = max(flag.points * 0.8, MAX_FLAG_DEDUCTION)

    eligible = audit_status in SCORING_ELIGIBLE_STATUSES
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
    total = BASE_SCORE
    for flag in flags:
        if flag.suppressed or not flag.scoring_eligible:
            continue
        total += flag.adjusted_points

    total = max(SCORE_FLOOR, min(SCORE_CEILING, total))
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
