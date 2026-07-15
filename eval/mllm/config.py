"""Local, secret-free configuration loader compatible with the existing gateway style."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: Path, model_slugs: set[str] | None = None) -> dict[str, Any]:
    config = _expand(json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("config.models must contain at least one model")
    shared_provider = config.get("provider")
    if shared_provider is None:
        shared_provider = next((model.get("provider") for model in config["models"] if model.get("provider")), None)
    if model_slugs:
        config["models"] = [model for model in config["models"] if model.get("slug") in model_slugs]
        if not config["models"]:
            raise ValueError(f"no configured models match {sorted(model_slugs)}")
    for model in config["models"]:
        provider = dict(model.get("provider") or shared_provider or {})
        provider["extraBody"] = {
            **((shared_provider or {}).get("extraBody") or {}),
            **(provider.get("extraBody") or {}),
            **(model.get("extraBody") or {}),
        }
        for key in ("apiKey", "apiBase"):
            if not provider.get(key):
                raise ValueError(f"model {model.get('slug', model.get('id'))}: missing provider.{key}")
        for key in ("id", "slug"):
            if not model.get(key):
                raise ValueError(f"model is missing {key}")
        model["provider"] = provider
        model.setdefault("temperature", 0)
        model.setdefault("maxTokens", 600)
        model.setdefault("concurrency", 1)
    config.setdefault("api", {}).setdefault("timeout", 120)
    config.setdefault("retry", {}).setdefault("maxRetriesPerReplicate", 5)
    config["retry"].setdefault("baseBackoffSeconds", [2, 4, 8, 16, 32])
    config.setdefault("image", {}).setdefault("transport", "base64")
    return config
