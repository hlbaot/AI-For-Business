"""TF-IDF keyword extraction module for Vietnamese segmented text."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "classified_data.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "tfidf_results.csv"

VIETNAMESE_STOPWORDS: frozenset[str] = frozenset(
    {
        "và",
        "của",
        "là",
        "có",
        "được",
        "cho",
        "để",
        "khi",
        "này",
        "đó",
        "kia",
        "các",
        "những",
        "một",
        "với",
        "trong",
        "ngoài",
        "trên",
        "dưới",
        "từ",
        "đến",
        "tại",
        "ở",
        "về",
        "thì",
        "mà",
        "nên",
        "vì",
        "nếu",
        "nhưng",
        "hoặc",
        "hay",
        "cũng",
        "rất",
        "quá",
        "lắm",
        "đã",
        "đang",
        "sẽ",
        "bị",
        "phải",
        "lại",
        "ra",
        "vào",
        "lên",
        "xuống",
        "mình",
        "tôi",
        "bạn",
        "em",
        "anh",
        "chị",
        "họ",
        "ta",
        "ai",
        "gì",
        "nào",
        "sao",
        "như",
        "nhiều",
        "hơn",
        "nhất",
        "vậy",
        "thế",
        "đây",
        "ấy",
        "đi",
        "làm",
        "việc",
        "người",
        "năm",
        "tháng",
        "ngày",
    }
)

TARGET_CATEGORIES: tuple[str, ...] = ("Motivation", "Purpose", "Preparation")

PUNCT_ONLY_PATTERN = re.compile(r"^[_\W]+$", flags=re.UNICODE)


def vietnamese_segmented_tokenizer(text: str) -> list[str]:
    """Tokenize pre-segmented Vietnamese text (compound words joined by '_')."""
    if not isinstance(text, str):
        return []
    tokens: list[str] = []
    for raw_tok in text.lower().split():
        tok = raw_tok.strip(".,?!:;\"'()[]")
        if not tok or PUNCT_ONLY_PATTERN.match(tok):
            continue
        if tok in VIETNAMESE_STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        tokens.append(tok)
    return tokens


def _extract_top_terms(texts: list[str], group_name: str, top_n: int) -> list[dict[str, Any]]:
    """Compute mean TF-IDF scores and return the top N terms for a text group."""
    non_empty = [t for t in texts if isinstance(t, str) and t.strip()]
    if not non_empty:
        return []

    max_df = 0.95 if len(non_empty) >= 5 else 1.0
    vectorizer = TfidfVectorizer(
        tokenizer=vietnamese_segmented_tokenizer,
        token_pattern=None,
        lowercase=False,
        min_df=1,
        max_df=max_df,
    )

    try:
        matrix = vectorizer.fit_transform(non_empty)
    except ValueError:
        # Fallback if all terms were filtered by max_df on a tiny subset
        vectorizer = TfidfVectorizer(
            tokenizer=vietnamese_segmented_tokenizer,
            token_pattern=None,
            lowercase=False,
            min_df=1,
            max_df=1.0,
        )
        matrix = vectorizer.fit_transform(non_empty)

    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) == 0:
        return []

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
    return rows


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

    # Only the 3 target groups are analysed; "ALL" = Motivation + Purpose + Preparation combined.
    if "category" in df.columns:
        df = df[df["category"].isin(TARGET_CATEGORIES)].reset_index(drop=True)
        if df.empty:
            raise ValueError("No documents in Motivation/Purpose/Preparation for TF-IDF.")

    all_rows = _extract_top_terms(df["text_segmented"].fillna("").tolist(), "ALL", top_n)

    groups_analyzed = ["ALL"]
    if "category" in df.columns:
        for category in sorted(df["category"].dropna().unique()):
            subset = df.loc[df["category"] == category, "text_segmented"].fillna("").tolist()
            cat_rows = _extract_top_terms(subset, str(category), top_n)
            if cat_rows:
                all_rows.extend(cat_rows)
                groups_analyzed.append(str(category))

    result_df = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    stats: dict[str, Any] = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "groups_analyzed": groups_analyzed,
        "total_keywords_exported": len(result_df),
    }
    logger.info(
        "TF-IDF complete: exported %d rows across groups %s to %s",
        len(result_df),
        groups_analyzed,
        out_path,
    )
    return result_df, stats
