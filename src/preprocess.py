"""Clean and normalize resume / job description text."""

from __future__ import annotations

import re
import unicodedata


def preprocess_text(raw: str) -> str:
    """
    Lowercase, normalize unicode, collapse whitespace, strip noise characters.
    """
    if not raw or not isinstance(raw, str):
        return ""

    try:
        text = unicodedata.normalize("NFKC", raw)
    except Exception:
        text = raw

    text = text.lower()
    # Keep letters, numbers, spaces, and common separators for skill phrases
    text = re.sub(r"[^\w\s\-\+/#\.]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
