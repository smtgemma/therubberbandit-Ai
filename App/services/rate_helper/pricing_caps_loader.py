import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable

_RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")
_PRICING_CAPS_PATH = os.path.join(_RULES_DIR, "pricing_caps.json")


@lru_cache(maxsize=1)
def load_pricing_caps() -> Dict[str, Any]:
    try:
        with open(_PRICING_CAPS_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_pricing_cap(data: Dict[str, Any], path: Iterable[str], default: Any) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
