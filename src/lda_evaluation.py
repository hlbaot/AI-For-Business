"""Compare LDA topic counts on one fixed corpus and export evidence for review."""

from __future__ import annotations

import argparse
import hashlib
import logging
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

from src.classification import assess_study_scope
from src.lda import (
    DEFAULT_INPUT_PATH, DEFAULT_N_TOPICS, DEFAULT_TOPICS_PATH, LDA_MAX_ITER, LDA_PRIOR, LDA_RANDOM_STATE,
    load_lda_corpus, make_lda_model,
)
from src.visualize import plot_lda_comparison

K_VALUES = (2, 3, 4, 5, 6)
TOP_WORDS = 20
SAMPLE_SIZE = 15
RANDOM_SAMPLE_SIZE = 24
RANDOM_SAMPLE_SEED = 42
WINDOW_SIZE = 110
PENDING = "pending_manual_review"
REVIEW_COLUMNS = [
    "text_id", "dominant_topic", "topic_probability", "text_cleaned",
    "suspected_off_topic", "review_reason", "manual_review_status",
    "relevance_reason",
]
logger = logging.getLogger(__name__)


def evaluate_topic_counts(
    input_path: Path, output_dir: Path, review_k: int = DEFAULT_N_TOPICS,
) -> pd.DataFrame:
    """Fit K=2..6 with fixed priors and compute C_v, diversity and train perplexity."""
    if review_k not in K_VALUES:
        raise ValueError(f"review_k must be one of {K_VALUES}.")
    classified_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    corpus = load_lda_corpus(input_path)
    docs = corpus.documents
    if not {"text_id", "text_cleaned"} <= set(docs.columns):
        raise ValueError("LDA review requires text_id and text_cleaned in the classified CSV.")
    if len(docs) < max(K_VALUES) or len(corpus.vocabulary) < 2:
        raise ValueError("Evaluation requires at least 6 modelled documents and 2 vocabulary terms.")

    top_n = min(TOP_WORDS, len(corpus.vocabulary))
    dictionary = Dictionary(corpus.tokens)
    scope_flags = docs["text_cleaned"].fillna("").map(assess_study_scope)
    random_indices = np.random.default_rng(RANDOM_SAMPLE_SEED).choice(
        len(docs), size=min(RANDOM_SAMPLE_SIZE, len(docs)), replace=False
    )
    score_rows = []
    review_rows = []
    random_rows = []

    def review_row(k: int, row: int, topic: int, probability: float) -> dict:
        doc = docs.iloc[row]
        flag, signals = scope_flags.iloc[row]
        return {
            "k": k,
            "text_id": doc["text_id"],
            "dominant_topic": f"Topic_{topic}",
            "topic_probability": probability,
            "text_cleaned": doc["text_cleaned"],
            "suspected_off_topic": bool(flag),
            "review_reason": f"{flag}: {signals}" if flag else "",
            "manual_review_status": PENDING,
            "relevance_reason": doc.get("relevance_reason", ""),
        }

    summary = [
        "SO SÁNH SỐ CHỦ ĐỀ LDA",
        f"Đầu vào: {input_path}",
        f"SHA-256 của CSV phân loại: {classified_hash}",
        f"Văn bản: {len(docs)}/{corpus.candidate_count} ứng viên thuộc phạm vi LDA.",
        "Tập phân tích dùng relevance_status=relevant, kể cả văn bản thuộc Other.",
        f"Loại khi lập ma trận: {corpus.excluded_no_tokens} không có token, "
        f"{corpus.excluded_no_retained_terms} không có từ trong từ vựng.",
        f"CountVectorizer: min_df={corpus.min_df}, từ vựng={len(corpus.vocabulary)}.",
        f"LDA: batch, max_iter={LDA_MAX_ITER}, random_state={LDA_RANDOM_STATE}, "
        f"doc_topic_prior=topic_word_prior={LDA_PRIOR:.6f} (1/3).",
        f"C_v: Gensim, cửa sổ {WINDOW_SIZE} token, top {top_n} từ, "
        "tập tham chiếu là cùng văn bản đã tiền xử lý dùng để huấn luyện.",
        f"Diversity: số từ duy nhất chia cho K × {top_n}. "
        "Perplexity: ước lượng trên tập huấn luyện, dùng để đối chiếu bổ sung.",
        f"Mỗi chủ đề xuất tối đa {SAMPLE_SIZE} bình luận có xác suất cao nhất "
        "trong các văn bản mà chủ đề đó chiếm ưu thế; dùng xác suất chưa làm tròn để xếp hạng.",
        f"Mẫu ngẫu nhiên: {len(random_indices)} văn bản chọn đều, không hoàn lại, "
        f"seed={RANDOM_SAMPLE_SEED}. So sánh các K bằng cùng các ID trong lda_k_random_review_samples.csv.",
        f"lda_topic_review.csv trích mẫu xác suất cao của K={review_k}.",
        "Tín hiệu lệch phạm vi là gợi ý theo luật, kể cả khi ngữ cảnh chưa đủ để đổi nhãn. "
        "Mẫu xác suất cao phục vụ đọc định tính "
        "và không ước lượng tỷ lệ lệch đề tài của toàn bộ chủ đề.",
        "Chưa có điểm diễn giải do người đánh giá xác nhận. Không tự động chọn K.",
        "Nguồn C_v: Röder, Both & Hinneburg (2015), https://doi.org/10.1145/2684822.2685324",
        f"Môi trường: Python {platform.python_version()}, " + ", ".join(
            f"{package} {version(package)}"
            for package in ("gensim", "scikit-learn", "pandas", "numpy", "scipy", "underthesea")
        ),
    ]

    for k in K_VALUES:
        logger.info("Evaluating K=%d on %d documents", k, len(docs))
        model = make_lda_model(k)
        probabilities = model.fit_transform(corpus.matrix)
        dominant = probabilities.argmax(axis=1)
        indices = np.argsort(model.components_, axis=1)[:, ::-1][:, :top_n]
        top_terms = corpus.vocabulary[indices].tolist()
        cv_model = CoherenceModel(
            topics=top_terms, texts=corpus.tokens, dictionary=dictionary,
            coherence="c_v", topn=top_n, window_size=WINDOW_SIZE, processes=1,
        )
        topic_coherences = cv_model.get_coherence_per_topic()
        coherence = float(np.mean(topic_coherences))
        diversity = len({word for topic in top_terms for word in topic}) / (k * top_n)
        perplexity = float(model.perplexity(corpus.matrix))
        if not np.isfinite([coherence, diversity, perplexity]).all():
            raise ValueError(f"Non-finite evaluation metrics for K={k}.")
        score_rows.append({
            "k": k,
            "coherence": coherence,
            "topic_diversity": diversity,
            "perplexity": perplexity,
            "interpretability_notes": PENDING,
        })
        summary.append(f"\nK={k} | C_v={coherence:.6f} | diversity={diversity:.6f} | perplexity={perplexity:.4f}")

        for row in random_indices:
            topic = dominant[row]
            random_rows.append(review_row(k, row, topic, probabilities[row, topic]))

        for topic, keywords in enumerate(top_terms):
            rows = np.flatnonzero(dominant == topic)
            by_probability = rows[np.argsort(-probabilities[rows, topic], kind="stable")]
            sample = by_probability[:SAMPLE_SIZE]
            topic_reviews = [review_row(k, row, topic, probabilities[row, topic]) for row in sample]
            review_rows.extend(topic_reviews)
            flag_count = sum(row["suspected_off_topic"] for row in topic_reviews)
            summary.append(
                f"Topic_{topic} | C_v={topic_coherences[topic]:.6f} | "
                f"số văn bản={len(rows)} | gợi ý lệch phạm vi trong mẫu={flag_count}/{len(sample)}\n"
                + ", ".join(keywords)
            )

    if hashlib.sha256(input_path.read_bytes()).hexdigest() != classified_hash:
        raise ValueError("The classified CSV changed during evaluation. Run the evaluation again.")

    scores = pd.DataFrame(score_rows)
    reviews = pd.DataFrame(review_rows, columns=["k", *REVIEW_COLUMNS])
    figure = output_dir / "figures" / "lda_k_comparison.png"
    plot_lda_comparison(scores, figure)
    tables = {
        "lda_evaluation.csv": scores,
        "lda_topic_review.csv": reviews.loc[reviews["k"].eq(review_k), REVIEW_COLUMNS],
        "lda_k_review_samples.csv": reviews,
        "lda_k_random_review_samples.csv": pd.DataFrame(random_rows, columns=["k", *REVIEW_COLUMNS]),
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False, encoding="utf-8-sig", float_format="%.8f")
    summary.append("\nSHA-256 của các tệp đánh giá:")
    for path in (figure, *(output_dir / filename for filename in tables)):
        summary.append(f"{path.relative_to(output_dir).as_posix()}  {hashlib.sha256(path.read_bytes()).hexdigest()}")
    (output_dir / "lda_evaluation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TOPICS_PATH.parent)
    parser.add_argument("--review-k", type=int, choices=K_VALUES, default=DEFAULT_N_TOPICS,
                        help="Use the same K as the main pipeline for lda_topic_review.csv.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    evaluate_topic_counts(args.input, args.output_dir, args.review_k)


if __name__ == "__main__":
    main()
