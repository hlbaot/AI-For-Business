"""LDA topic modeling module using word frequency matrices (CountVectorizer)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.tfidf import TARGET_CATEGORIES, vietnamese_segmented_tokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "classified_data.csv"
DEFAULT_TOPICS_PATH = PROJECT_ROOT / "results" / "lda_topics.csv"
DEFAULT_DOC_TOPICS_PATH = PROJECT_ROOT / "results" / "lda_doc_topics.csv"


def _assess_interpretability(n_docs: int, n_topics: int, vocab_size: int) -> str:
    """Provide an honest interpretability assessment based on corpus size."""
    if n_docs < 30:
        return (
            f"Small corpus warning ({n_docs} docs, {n_topics} topics, {vocab_size} terms): "
            "LDA topics are exploratory only and may overlap; validate manually."
        )
    if n_docs < n_topics * 10:
        return (
            f"Moderate corpus size ({n_docs} docs for {n_topics} topics): "
            "interpretability should be checked manually by inspecting top keywords."
        )
    return (
        f"Corpus size ({n_docs} docs, {vocab_size} terms) is suitable for exploratory "
        f"topic inspection across {n_topics} topics."
    )


def run_lda_analysis(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_TOPICS_PATH,
    doc_topics_path: Path | str | None = DEFAULT_DOC_TOPICS_PATH,
    n_topics: int = 3,
    top_n_words: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run LDA topic modeling on word-segmented Vietnamese text using term frequencies."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists() or in_path.stat().st_size == 0:
        raise ValueError(f"Input file for LDA is missing or empty: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    if df.empty or "text_segmented" not in df.columns:
        raise ValueError("Input DataFrame must be non-empty and contain 'text_segmented'.")

    # Only the 3 target groups (exclude "Other") are modelled.
    if "category" in df.columns:
        df = df[df["category"].isin(TARGET_CATEGORIES)].reset_index(drop=True)

    valid_mask = df["text_segmented"].fillna("").str.strip() != ""
    df_valid = df.loc[valid_mask].copy().reset_index(drop=True)
    texts = df_valid["text_segmented"].tolist()

    if len(texts) < 2:
        raise ValueError(f"Need at least 2 non-empty documents for LDA, got {len(texts)}.")

    effective_topics = min(max(1, n_topics), len(texts))
    if effective_topics != n_topics:
        logger.warning(
            "Requested n_topics=%d exceeds document count=%d; adjusted to %d.",
            n_topics,
            len(texts),
            effective_topics,
        )

    max_df = 0.95 if len(texts) >= 5 else 1.0
    vectorizer = CountVectorizer(
        tokenizer=vietnamese_segmented_tokenizer,
        token_pattern=None,
        lowercase=False,
        min_df=1,
        max_df=max_df,
    )

    try:
        dtm = vectorizer.fit_transform(texts)
    except ValueError:
        vectorizer = CountVectorizer(
            tokenizer=vietnamese_segmented_tokenizer,
            token_pattern=None,
            lowercase=False,
            min_df=1,
            max_df=1.0,
        )
        dtm = vectorizer.fit_transform(texts)

    feature_names = vectorizer.get_feature_names_out()
    vocab_size = len(feature_names)
    if vocab_size == 0:
        raise ValueError("Vocabulary is empty after tokenization; cannot fit LDA.")

    lda = LatentDirichletAllocation(
        n_components=effective_topics,
        max_iter=30,
        learning_method="batch",
        random_state=random_state,
    )
    doc_topic_matrix = lda.fit_transform(dtm)

    interpretability_note = _assess_interpretability(len(texts), effective_topics, vocab_size)

    topic_rows: list[dict[str, Any]] = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        # Normalize weights into topic-word probabilities for clear interpretation
        probs = topic_weights / topic_weights.sum()
        top_indices = np.argsort(probs)[::-1][:top_n_words]
        for rank, word_idx in enumerate(top_indices, start=1):
            topic_rows.append(
                {
                    "topic_id": topic_idx,
                    "topic_label": f"Topic_{topic_idx}",
                    "rank": rank,
                    "keyword": str(feature_names[word_idx]),
                    "weight": round(float(probs[word_idx]), 6),
                    "raw_pseudo_count": round(float(topic_weights[word_idx]), 4),
                    "interpretability_note": interpretability_note,
                }
            )

    topics_df = pd.DataFrame(topic_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    topics_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    if doc_topics_path is not None:
        doc_path = Path(doc_topics_path)
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_out = pd.DataFrame(
            {
                "text_id": df_valid["text_id"] if "text_id" in df_valid.columns else range(len(df_valid)),
                "dominant_topic": [f"Topic_{int(i)}" for i in np.argmax(doc_topic_matrix, axis=1)],
                "dominant_topic_prob": [round(float(p), 4) for p in np.max(doc_topic_matrix, axis=1)],
            }
        )
        for t_idx in range(effective_topics):
            doc_out[f"Topic_{t_idx}_prob"] = np.round(doc_topic_matrix[:, t_idx], 4)
        if "category" in df_valid.columns:
            doc_out["rule_category"] = df_valid["category"]
        doc_out.to_csv(doc_path, index=False, encoding="utf-8-sig")

    perplexity = float(lda.perplexity(dtm))
    log_likelihood = float(lda.score(dtm))
    stats: dict[str, Any] = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "doc_topics_output_path": str(doc_topics_path) if doc_topics_path else None,
        "n_documents": len(texts),
        "vocab_size": vocab_size,
        "n_topics": effective_topics,
        "perplexity": round(perplexity, 4),
        "log_likelihood": round(log_likelihood, 4),
        "interpretability_note": interpretability_note,
        "mapping_disclaimer": (
            "LDA topics (Topic_0, Topic_1, ...) are unsupervised statistical clusters "
            "and are NOT automatically equated with Motivation, Purpose, or Preparation."
        ),
    }
    logger.info(
        "LDA complete: %d topics, vocab=%d, perplexity=%.2f | %s",
        effective_topics,
        vocab_size,
        perplexity,
        interpretability_note,
    )
    return topics_df, stats
