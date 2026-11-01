"""Configuration loading.

All tunable values live in ``config/settings.yaml``. This module finds the
project root, loads the YAML once and exposes helpers for resolving paths.
Environment variables can override a few values without editing the file:

* ``RR_EMBEDDING_MODEL``  - Hugging Face id or local folder for the embedder
* ``RR_SENTIMENT_BACKEND`` - weak_supervised | vader | transformer
* ``RR_ARTIFACTS_DIR``    - alternative artifacts folder (used by tests)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    root: Path

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read a nested value with a dotted key, e.g. ``scoring.alpha``."""
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted_key: str) -> Path:
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"Missing path setting: {dotted_key}")
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    @property
    def processed_dir(self) -> Path:
        return self.path("paths.processed_dir")

    @property
    def interim_dir(self) -> Path:
        return self.path("paths.interim_dir")

    @property
    def artifacts_dir(self) -> Path:
        override = os.environ.get("RR_ARTIFACTS_DIR")
        return Path(override) if override else self.path("paths.artifacts_dir")

    @property
    def embedding_model(self) -> str:
        return os.environ.get("RR_EMBEDDING_MODEL") or self.get("embeddings.model_name")

    @property
    def sentiment_backend(self) -> str:
        return os.environ.get("RR_SENTIMENT_BACKEND") or self.get("sentiment.backend")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_file = Path(os.environ.get("RR_SETTINGS_FILE", PROJECT_ROOT / "config" / "settings.yaml"))
    return Settings(raw=_load_yaml(config_file), root=PROJECT_ROOT)


@lru_cache(maxsize=1)
def get_aspect_config() -> dict[str, Any]:
    return _load_yaml(get_settings().path("paths.aspects_file"))
