"""LDA topic modeling module using word frequency matrices (CountVectorizer)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

from src.relevance import select_analysis_documents
from src.text_features import vietnamese_segmented_tokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "classified_data.csv"
DEFAULT_TOPICS_PATH = PROJECT_ROOT / "results" / "lda_topics.csv"
DEFAULT_DOC_TOPICS_PATH = PROJECT_ROOT / "results" / "lda_doc_topics.csv"
DEFAULT_N_TOPICS = 4
LDA_RANDOM_STATE = 42
LDA_MAX_ITER = 30
LDA_PRIOR = 1 / 3  # Keep the same prior across K and across experiments.


@dataclass
class LDACorpus:
    documents: pd.DataFrame
    tokens: list[list[str]]
    matrix: csr_matrix
    vocabulary: np.ndarray
    candidate_count: int
    excluded_no_tokens: int
    excluded_no_retained_terms: int
    min_df: int


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


def load_lda_corpus(input_path: Path | str) -> LDACorpus:
    """Prepare the same document rows and term counts for production and evaluation."""
    in_path = Path(input_path)
    if not in_path.exists() or in_path.stat().st_size == 0:
        raise ValueError(f"Input file for LDA is missing or empty: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    if df.empty or "text_segmented" not in df.columns:
        raise ValueError("Input DataFrame must be non-empty and contain 'text_segmented'.")

    df = select_analysis_documents(df)

    model_texts = df["text_segmented"].map(lambda text: " ".join(vietnamese_segmented_tokenizer(text)))
    has_tokens = model_texts.ne("")
    texts = model_texts.loc[has_tokens]
    if len(texts) < 2:
        raise ValueError(f"Need at least 2 documents with usable terms for LDA, got {len(texts)}.")

    # In the project corpus, single-document terms are often typos, names or isolated noise.
    min_df = 2 if len(texts) >= 30 else 1
    vectorizer = CountVectorizer(analyzer=str.split, min_df=min_df)
    dtm = vectorizer.fit_transform(texts)
    has_features = dtm.getnnz(axis=1) > 0
    df_valid = df.loc[texts.index[has_features]].reset_index(drop=True)
    dtm = dtm[has_features]
    if len(df_valid) < 2:
        raise ValueError("Need at least 2 documents with retained vocabulary for LDA.")

    return LDACorpus(
        documents=df_valid,
        tokens=[text.split() for text in texts.loc[has_features]],
        matrix=dtm,
        vocabulary=vectorizer.get_feature_names_out(),
        candidate_count=len(df),
        excluded_no_tokens=int((~has_tokens).sum()),
        excluded_no_retained_terms=int((~has_features).sum()),
        min_df=min_df,
    )


def make_lda_model(n_topics: int, random_state: int = LDA_RANDOM_STATE) -> LatentDirichletAllocation:
    return LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=LDA_MAX_ITER,
        learning_method="batch",
        random_state=random_state,
        doc_topic_prior=LDA_PRIOR,
        topic_word_prior=LDA_PRIOR,
    )


def run_lda_analysis(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    output_path: Path | str = DEFAULT_TOPICS_PATH,
    doc_topics_path: Path | str | None = DEFAULT_DOC_TOPICS_PATH,
    n_topics: int = DEFAULT_N_TOPICS,
    top_n_words: int = 10,
    random_state: int = LDA_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run LDA topic modeling on word-segmented Vietnamese text using term frequencies."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    corpus = load_lda_corpus(in_path)
    df_valid, dtm, feature_names = corpus.documents, corpus.matrix, corpus.vocabulary
    effective_topics = min(max(1, n_topics), len(df_valid))
    if effective_topics != n_topics:
        logger.warning("Adjusted n_topics=%d to %d for %d modelled documents.",
                       n_topics, effective_topics, len(df_valid))
    vocab_size = len(feature_names)
    lda = make_lda_model(effective_topics, random_state)
    doc_topic_matrix = lda.fit_transform(dtm)

    interpretability_note = _assess_interpretability(len(df_valid), effective_topics, vocab_size)

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
        "n_documents": len(df_valid),
        "n_candidate_documents": corpus.candidate_count,
        "excluded_no_tokens": corpus.excluded_no_tokens,
        "excluded_no_retained_terms": corpus.excluded_no_retained_terms,
        "min_document_frequency": corpus.min_df,
        "vocab_size": vocab_size,
        "n_topics": effective_topics,
        "random_state": random_state,
        "max_iter": LDA_MAX_ITER,
        "doc_topic_prior": LDA_PRIOR,
        "topic_word_prior": LDA_PRIOR,
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
