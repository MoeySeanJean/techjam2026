"""Load local credentials from a gitignored `.env`.

Deliberately dependency-free and deliberately non-magical: it reads `.env` into
`os.environ` without overwriting anything already set, so a real environment
variable always wins over the file. Nothing here ever prints or logs a secret.

The deliverable is a public repository, so `.env` is listed in `.gitignore` and
no credential appears in any tracked file. `.env.example` documents the shape.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT, ".env")

_loaded = False


def load(path: Optional[str] = None, override: bool = False) -> Dict[str, str]:
    """Read `.env` into the process environment. Idempotent."""
    global _loaded
    path = path or ENV_PATH
    found: Dict[str, str] = {}
    if not os.path.exists(path):
        _loaded = True
        return found
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if not key or not value:
                continue
            found[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    _loaded = True
    return found


def get(name: str, default: Optional[str] = None) -> Optional[str]:
    if not _loaded:
        load()
    return os.environ.get(name, default)


def redact(secret: Optional[str]) -> str:
    """Render a credential safe to print in logs and reports."""
    if not secret:
        return "(unset)"
    if len(secret) <= 10:
        return "***"
    return f"{secret[:9]}...{secret[-4:]}"
