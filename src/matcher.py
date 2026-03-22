"""TF-IDF vectorization and cosine similarity between job description and resumes."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_match_scores(
    job_description: str,
    resume_texts: Sequence[str],
) -> list[float]:
    """
    Fit TF-IDF on [job_description] + resumes, return cosine similarity of each resume to JD.

    Scores are in [0, 1] when vectors are non-zero; empty inputs yield 0.0.
    """
    jd = (job_description or "").strip()
    resumes = [((r or "").strip()) for r in resume_texts]

    if not jd:
        return [0.0] * len(resumes)

    if not resumes:
        return []

    try:
        corpus = [jd] + list(resumes)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(corpus)
        jd_vec = matrix[0:1]
        resume_matrix = matrix[1:]
        sims = cosine_similarity(resume_matrix, jd_vec).ravel()
        # Replace NaN (e.g. zero vectors) with 0.0
        out = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)
        return [float(x) for x in out.tolist()]
    except ValueError as exc:
        logger.warning("TF-IDF / similarity failed (empty vocabulary?): %s", exc)
        return [0.0] * len(resumes)
    except Exception as exc:
        logger.exception("compute_match_scores error: %s", exc)
        return [0.0] * len(resumes)


def rank_by_score(
    names: Sequence[str],
    scores: Sequence[float],
) -> list[tuple[int, str, float]]:
    """Return list of (original_index, name, score) sorted by score descending."""
    if len(names) != len(scores):
        logger.warning("names/scores length mismatch")
        n = min(len(names), len(scores))
        names = names[:n]
        scores = scores[:n]

    indexed = [(i, names[i], float(scores[i])) for i in range(len(names))]
    indexed.sort(key=lambda x: x[2], reverse=True)
    return indexed
