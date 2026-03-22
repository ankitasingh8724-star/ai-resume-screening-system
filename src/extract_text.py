"""Extract plain text from PDF or text uploads."""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


def extract_text_from_bytes(data: bytes, filename: str = "") -> str:
    """
    Extract text from PDF (pdfminer, then PyPDF2 fallback) or decode as UTF-8 text.

    Returns empty string on failure; callers should validate.
    """
    if not data:
        logger.warning("Empty file bytes for %s", filename or "(unknown)")
        return ""

    lower = (filename or "").lower()
    is_pdf = lower.endswith(".pdf") or data[:4] == b"%PDF"

    if is_pdf:
        text = _extract_pdf_pdfminer(data, filename)
        if text.strip():
            return text
        text = _extract_pdf_pypdf2(data, filename)
        return text

    try:
        return data.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover — decode rarely fails with errors=replace
        logger.exception("UTF-8 decode failed for %s: %s", filename, exc)
        return ""


def _extract_pdf_pdfminer(data: bytes, filename: str) -> str:
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.pdfparser import PDFSyntaxError

        stream = io.BytesIO(data)
        return extract_text(stream) or ""
    except ImportError:
        logger.warning("pdfminer.six not installed; skipping for %s", filename)
        return ""
    except PDFSyntaxError as exc:
        logger.warning("PDF syntax error for %s: %s", filename, exc)
        return ""
    except Exception as exc:
        logger.warning("pdfminer failed for %s: %s", filename, exc)
        return ""


def _extract_pdf_pypdf2(data: bytes, filename: str) -> str:
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    parts.append(t)
            except Exception as exc:
                logger.debug("PyPDF2 page extract failed: %s", exc)
        return "\n".join(parts)
    except ImportError:
        logger.warning("PyPDF2 not installed; skipping for %s", filename)
        return ""
    except Exception as exc:
        logger.warning("PyPDF2 failed for %s: %s", filename, exc)
        return ""
