"""Extract skill mentions from text using a keyword list."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimal fallback if data file is missing
_DEFAULT_SKILLS: tuple[str, ...] = (
    "python",
    "java",
    "javascript",
    "sql",
    "machine learning",
    "aws",
    "docker",
    "kubernetes",
    "react",
    "git",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_skills_keywords(path: Path | None = None) -> list[str]:
    """Load skills from data/skills_keywords.txt; fallback to defaults on error."""
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.append(_project_root() / "data" / "skills_keywords.txt")

    for p in candidates:
        try:
            if not p.exists():
                continue
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            skills = [ln.strip().lower() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
            if skills:
                return sorted(set(skills), key=len, reverse=True)
        except OSError as exc:
            logger.warning("Could not read skills file %s: %s", p, exc)
        except Exception as exc:
            logger.warning("Unexpected error loading skills from %s: %s", p, exc)

    logger.info("Using embedded default skills list")
    return sorted(set(_DEFAULT_SKILLS), key=len, reverse=True)


def extract_skills(text: str, keywords: list[str] | None = None) -> list[str]:
    """
    Find skills from `keywords` that appear in normalized `text` (substring match).
    Longer phrases are checked first to prefer 'machine learning' over 'learning'.
    """
    if not text:
        return []

    try:
        kw = keywords if keywords is not None else load_skills_keywords()
    except Exception as exc:
        logger.warning("load_skills_keywords failed: %s; using defaults", exc)
        kw = sorted(set(_DEFAULT_SKILLS), key=len, reverse=True)

    haystack = f" {text} "
    found: list[str] = []
    seen: set[str] = set()

    for phrase in kw:
        if not phrase:
            continue
        pattern = re.escape(phrase.lower())
        # Word-ish boundaries: space or non-word around phrase
        if re.search(rf"(?<!\w){pattern}(?!\w)", haystack, re.IGNORECASE):
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                found.append(phrase)

    return sorted(found, key=str.lower)
