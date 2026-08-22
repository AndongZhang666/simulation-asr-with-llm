"""Canonical configuration serialization and reproducible experiment identifiers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigurationError(ValueError):
    """Raised when a configuration cannot be represented deterministically."""


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError("configuration values must not contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConfigurationError("configuration mapping keys must be strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    raise ConfigurationError(f"unsupported configuration value type: {type(value).__name__}")


def canonical_json(config: Mapping[str, Any]) -> str:
    """Serialize a mapping with an unambiguous, stable JSON representation."""

    canonical_config = _canonicalize(config)
    return json.dumps(canonical_config, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def config_sha256(config: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of the canonical configuration bytes."""

    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def run_id(prefix: str, config: Mapping[str, Any], *, digest_length: int = 12) -> str:
    """Create a human-readable identifier coupled to the full resolved configuration."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", prefix):
        raise ConfigurationError("run prefix must use lowercase letters, numbers, and hyphens")
    if digest_length < 8 or digest_length > 64:
        raise ConfigurationError("digest_length must be between 8 and 64")
    return f"{prefix}-{config_sha256(config)[:digest_length]}"
