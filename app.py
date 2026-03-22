"""
Streamlit dashboard: upload resumes, enter job description, TF-IDF match scores and ranking.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable when running `streamlit run app.py`
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.extract_text import extract_text_from_bytes
from src.matcher import compute_match_scores, rank_by_score
from src.preprocess import preprocess_text
from src.skills import extract_skills, load_skills_keywords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_session() -> None:
    if "candidates" not in st.session_state:
        st.session_state.candidates = []  # list of dict: name, raw_text, preprocessed


def _screening_input_key(jd_text: str, candidates: list) -> tuple:
    """Fingerprint job description + queue so cached results invalidate when inputs change."""
    try:
        jd_c = preprocess_text(jd_text or "")
    except Exception:
        jd_c = ""
    sig = tuple((c.get("name", ""), c.get("preprocessed", "")) for c in candidates)
    return (jd_c, sig)


def _invalidate_screening_cache_if_stale(jd_text: str, candidates: list) -> None:
    ck = _screening_input_key(jd_text, candidates)
    b = st.session_state.get("screening_bundle")
    if b is not None and b.get("key") != ck:
        st.session_state.pop("screening_bundle", None)


def _render_screening_dashboard(
    ranking: list,
    scores: list[float],
    max_overlap: int,
) -> None:
    st.subheader("Results dashboard")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Resumes screened", len(scores))
    with m2:
        st.metric("Best match score", f"{max(scores) * 100:.1f}%" if scores else "—")
    with m3:
        st.metric("Max skill overlap with JD", max_overlap)

    rank_rows = []
    for rank_pos, (orig_idx, name, score) in enumerate(ranking, start=1):
        cand = st.session_state.candidates[orig_idx]
        rank_rows.append(
            {
                "Rank": rank_pos,
                "Candidate": name,
                "Match score %": round(score * 100, 2),
                "Skills (sample)": ", ".join(cand.get("skills", [])[:12])
                + ("…" if len(cand.get("skills", [])) > 12 else ""),
            }
        )

    try:
        df = pd.DataFrame(rank_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Could not display table: {exc}")
        st.json(rank_rows)

    st.subheader("Similarity score per resume")
    for orig_idx, name, score in ranking:
        st.progress(min(1.0, max(0.0, score)), text=f"{name}: {score * 100:.1f}%")

    st.subheader("Extracted skills by candidate")
    for orig_idx, name, score in ranking:
        cand = st.session_state.candidates[orig_idx]
        with st.expander(f"{name} — {score * 100:.1f}% match"):
            sk_list = cand.get("skills", []) or []
            st.write(", ".join(sk_list) or "(none detected)")


def _add_uploaded_files(uploaded_files) -> tuple[int, list[str]]:
    """Process Streamlit uploaded files; return (count_added, error_messages)."""
    errors: list[str] = []
    added = 0
    if not uploaded_files:
        return 0, errors

    try:
        keywords = load_skills_keywords()
    except Exception as exc:
        logger.warning("skills keywords load: %s", exc)
        keywords = None

    for uf in uploaded_files:
        name = getattr(uf, "name", None) or "upload"
        try:
            data = uf.read()
        except Exception as exc:
            errors.append(f"{name}: could not read file ({exc})")
            continue

        try:
            raw = extract_text_from_bytes(data, filename=name)
        except Exception as exc:
            errors.append(f"{name}: extraction error ({exc})")
            continue

        if not raw or not str(raw).strip():
            errors.append(f"{name}: no text extracted (empty or unsupported PDF)")
            continue

        try:
            clean = preprocess_text(raw)
        except Exception as exc:
            errors.append(f"{name}: preprocess failed ({exc})")
            continue

        try:
            sk = extract_skills(clean, keywords=keywords)
        except Exception as exc:
            logger.warning("skills for %s: %s", name, exc)
            sk = []

        st.session_state.candidates.append(
            {
                "name": name,
                "raw_text": raw,
                "preprocessed": clean,
                "skills": sk,
            }
        )
        added += 1

    return added, errors


def main() -> None:
    st.set_page_config(page_title="AI Resume Screening", layout="wide")
    _init_session()

    st.title("AI Resume Screening System")
    st.caption("NLP-style skill extraction, TF-IDF vectors, cosine similarity vs. job description.")

    col_left, col_right = st.columns((1, 1))

    with col_left:
        st.subheader("Job description")
        jd = st.text_area(
            "Paste the job description",
            height=220,
            placeholder="Required: role summary, requirements, tech stack…",
            key="jd",
        )

        st.subheader("Upload resumes")
        files = st.file_uploader(
            "PDF or plain text (multiple files supported)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Add uploads to queue", type="primary"):
                try:
                    added, errs = _add_uploaded_files(files or [])
                    if added:
                        st.success(f"Added {added} resume(s) to the queue.")
                    for e in errs:
                        st.warning(e)
                    if not files:
                        st.info("Choose one or more files first.")
                except Exception as exc:
                    st.error(f"Failed to process uploads: {exc}")
                    logger.exception("upload handler")

        with c2:
            if st.button("Clear queue"):
                try:
                    st.session_state.candidates = []
                    st.session_state.pop("screening_bundle", None)
                    st.success("Queue cleared.")
                except Exception as exc:
                    st.error(str(exc))

        with c3:
            if st.button("Run screening"):
                st.session_state["run_screening"] = True

    with col_right:
        st.subheader("Queue")
        n = len(st.session_state.candidates)
        if n == 0:
            st.info("No resumes in queue. Upload PDF or TXT files and click **Add uploads to queue**.")
        else:
            for i, c in enumerate(st.session_state.candidates):
                st.write(f"{i + 1}. **{c['name']}** — {len(c.get('skills', []))} skills detected")

    st.divider()

    _invalidate_screening_cache_if_stale(jd, st.session_state.candidates)
    run = st.session_state.pop("run_screening", False)

    if run:
        jd_clean = ""
        try:
            jd_clean = preprocess_text(jd or "")
        except Exception as exc:
            st.error(f"Could not preprocess job description: {exc}")
            jd_clean = ""

        if not (jd or "").strip():
            st.warning("Enter a job description before running screening.")
        elif not st.session_state.candidates:
            st.warning("Add at least one resume to the queue.")
        else:
            resumes_prep = [c["preprocessed"] for c in st.session_state.candidates]
            names = [c["name"] for c in st.session_state.candidates]

            try:
                scores = compute_match_scores(jd_clean, resumes_prep)
            except Exception as exc:
                st.error(f"Matching failed: {exc}")
                logger.exception("compute_match_scores")
                scores = [0.0] * len(names)

            ranking = rank_by_score(names, scores)
            max_overlap = 0
            try:
                jd_skills = set(extract_skills(jd_clean, keywords=load_skills_keywords()))
                for c in st.session_state.candidates:
                    max_overlap = max(
                        max_overlap,
                        len(jd_skills.intersection(set(c.get("skills", [])))),
                    )
            except Exception:
                pass

            st.session_state.screening_bundle = {
                "key": _screening_input_key(jd, st.session_state.candidates),
                "ranking": ranking,
                "scores": scores,
                "max_overlap": max_overlap,
            }

    bundle = st.session_state.get("screening_bundle")
    if bundle is not None:
        ck = _screening_input_key(jd, st.session_state.candidates)
        if bundle.get("key") == ck:
            try:
                _render_screening_dashboard(
                    bundle["ranking"],
                    bundle["scores"],
                    bundle.get("max_overlap", 0),
                )
            except Exception as exc:
                st.error(f"Could not render results: {exc}")
                logger.exception("render dashboard")
        else:
            st.session_state.pop("screening_bundle", None)

    st.divider()
    st.caption(
        "Uses pdfminer.six with PyPDF2 fallback, text cleaning, keyword-based skill extraction, "
        "and scikit-learn TF-IDF + cosine similarity."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("app main")
        st.error(f"Application error: {exc}")
