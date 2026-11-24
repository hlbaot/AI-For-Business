"""TF-IDF keyword extraction module for Vietnamese segmented text."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.relevance import select_analysis_documents
from src.text_features import vietnamese_segmented_tokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "classified_data.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "tfidf_results.csv"

TARGET_CATEGORIES: tuple[str, ...] = ("Motivation", "Purpose", "Preparation")


def _extract_top_terms(texts: list[str], group_name: str, top_n: int) -> tuple[list[dict[str, Any]], int]:
    """Compute mean TF-IDF scores and return the top N terms for a text group."""
    model_texts = [" ".join(vietnamese_segmented_tokenizer(text)) for text in texts]
    non_empty = [text for text in model_texts if text]
    if not non_empty:
        return [], 0

    vectorizer = TfidfVectorizer(analyzer=str.split)
    matrix = vectorizer.fit_transform(non_empty)
    feature_names = vectorizer.get_feature_names_out()

    mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
    doc_freqs = np.asarray((matrix > 0).sum(axis=0)).ravel()
    top_indices = np.argsort(mean_scores)[::-1][:top_n]

    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        rows.append(
            {
                "group": group_name,
                "rank": rank,
                "keyword": str(feature_names[idx]),
                "tfidf_score": round(float(mean_scores[idx]), 6),
                "doc_freq": int(doc_freqs[idx]),
                "group_doc_count": len(non_empty),
            }
        )
    return rows, len(non_empty)


def run_tfidf_analysis(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    top_n: int = 15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run overall and per-category TF-IDF keyword extraction."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists() or in_path.stat().st_size == 0:
        raise ValueError(f"Input file for TF-IDF is missing or empty: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    if df.empty or "text_segmented" not in df.columns:
        raise ValueError("Input DataFrame must be non-empty and contain 'text_segmented'.")

    df = select_analysis_documents(df)
    if df.empty:
        raise ValueError("No relevant documents for TF-IDF.")

    all_rows, n_documents = _extract_top_terms(df["text_segmented"].tolist(), "ALL", top_n)

    groups_analyzed = ["ALL"]
    document_counts = {"ALL": n_documents}
    if "category" in df.columns:
        for category in sorted(TARGET_CATEGORIES):
            subset = df.loc[df["category"] == category, "text_segmented"].fillna("").tolist()
            cat_rows, count = _extract_top_terms(subset, str(category), top_n)
            document_counts[str(category)] = count
            if cat_rows:
                all_rows.extend(cat_rows)
                groups_analyzed.append(str(category))

    result_df = pd.DataFrame(
        all_rows, columns=["group", "rank", "keyword", "tfidf_score", "doc_freq", "group_doc_count"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    stats: dict[str, Any] = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "groups_analyzed": groups_analyzed,
        "n_candidate_documents": len(df),
        "n_documents": n_documents,
        "group_document_counts": document_counts,
        "total_keywords_exported": len(result_df),
    }
    logger.info(
        "TF-IDF complete: exported %d rows across groups %s to %s",
        len(result_df),
        groups_analyzed,
        out_path,
    )
    return result_df, stats
