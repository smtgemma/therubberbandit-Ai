import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable

_RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")
_PRICING_CAPS_PATH = os.path.join(_RULES_DIR, "pricing_caps.json")


class PricingCapLoadError(RuntimeError):
    """Raised when pricing cap rules are missing or malformed."""


@lru_cache(maxsize=1)
def load_pricing_caps() -> Dict[str, Any]:
    try:
        with open(_PRICING_CAPS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PricingCapLoadError("RULE_LOAD_FAILURE: cannot load pricing_caps.json") from exc
    if not isinstance(data, dict):
        raise PricingCapLoadError("RULE_LOAD_FAILURE: pricing_caps.json must contain a JSON object")
    return data


def get_pricing_cap(data: Dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = data
    traversed = []
    for key in path:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            joined = ".".join(traversed)
            raise PricingCapLoadError(f"RULE_LOAD_FAILURE: missing pricing_caps.{joined}")
        current = current[key]
    return current
