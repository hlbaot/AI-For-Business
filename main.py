"""Main pipeline runner: Cleaning -> Classification -> TF-IDF -> LDA -> Orange Export."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.classification import classify_dataset
from src.cleaning import clean_dataset
from src.lda import DEFAULT_N_TOPICS, run_lda_analysis
from src.text_features import vietnamese_segmented_tokenizer
from src.tfidf import run_tfidf_analysis
from src.visualize import generate_all_figures

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"
DEFAULT_CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
DEFAULT_CLASSIFIED_PATH = PROJECT_ROOT / "data" / "processed" / "classified_data.csv"
DEFAULT_REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "manual_review_needed.csv"
DEFAULT_RELEVANCE_REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "relevance_review.csv"
DEFAULT_SCOPE_REVIEW_PATH = PROJECT_ROOT / "results" / "suspected_off_topic.csv"
DEFAULT_ORANGE_PATH = PROJECT_ROOT / "data" / "exports" / "orange_data.csv"
DEFAULT_TFIDF_PATH = PROJECT_ROOT / "results" / "tfidf_results.csv"
DEFAULT_LDA_TOPICS_PATH = PROJECT_ROOT / "results" / "lda_topics.csv"
DEFAULT_LDA_DOC_TOPICS_PATH = PROJECT_ROOT / "results" / "lda_doc_topics.csv"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"


def configure_logging() -> None:
    """Configure clear console logging for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # override handlers pre-installed by third-party libs (e.g. underthesea)
    )


def ensure_directories() -> None:
    """Create required output directories if they do not exist."""
    for folder in (
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "exports",
        PROJECT_ROOT / "results",
    ):
        folder.mkdir(parents=True, exist_ok=True)


def export_for_orange(
    input_path: Path | str = DEFAULT_CLASSIFIED_PATH,
    output_path: Path | str = DEFAULT_ORANGE_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Export classified data into a clean UTF-8-SIG CSV ready for Orange Data Mining."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists() or in_path.stat().st_size == 0:
        raise ValueError(
            f"Classified dataset is missing or empty: {in_path}. "
            "Run the full pipeline before exporting for Orange."
        )

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    required = ["text_id", "text_cleaned", "text_segmented", "category"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Orange export: {missing}")

    optional = [c for c in ("created_at", "needs_manual_review") if c in df.columns]
    orange_df = df[required + optional].copy()
    orange_df["text_model"] = df["text_segmented"].map(
        lambda text: " ".join(vietnamese_segmented_tokenizer(text))
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    orange_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    stats = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "exported_rows": len(orange_df),
        "columns": list(orange_df.columns),
        "word_cloud_text_feature": "text_model",
    }
    logging.getLogger(__name__).info(
        "Orange export complete: %d rows written to %s", len(orange_df), out_path
    )
    return orange_df, stats


def resolve_input_path(requested_path: Path) -> Path:
    """Resolve input path, falling back to a non-empty .xlsx/.csv in data/raw if raw_data.csv is empty."""
    if requested_path.exists() and requested_path.stat().st_size > 0:
        return requested_path

    if requested_path == DEFAULT_RAW_PATH:
        raw_dir = PROJECT_ROOT / "data" / "raw"
        candidates = sorted(
            p
            for p in raw_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
            and p.name not in {"raw_data.csv", "sample_test_data.csv"}
            and p.stat().st_size > 0
        )
        if len(candidates) == 1:
            logging.getLogger("main").info(
                "raw_data.csv is empty/missing; auto-detected input file: %s",
                candidates[0].name,
            )
            return candidates[0]

    return requested_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Vietnamese study-abroad NLP pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Path to raw input CSV (default: data/raw/raw_data.csv).",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=DEFAULT_N_TOPICS,
        help=f"Number of LDA topics (default: {DEFAULT_N_TOPICS}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Top N keywords per group for TF-IDF (default: 15).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only regenerate Orange export from existing classified_data.csv.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only regenerate charts from existing results (TF-IDF, LDA, categories).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the pipeline and save run metrics."""
    configure_logging()
    logger = logging.getLogger("main")
    args = parse_args()
    ensure_directories()

    try:
        if args.export_only:
            logger.info("Running in --export-only mode (skipping Cleaning, TF-IDF, LDA).")
            _, orange_stats = export_for_orange(DEFAULT_CLASSIFIED_PATH, DEFAULT_ORANGE_PATH)
            logger.info("Done: %s", orange_stats["output_path"])
            return 0

        if args.plots_only:
            logger.info("Running in --plots-only mode (rendering charts from existing results).")
            fig_stats = generate_all_figures()
            logger.info("Done: %d figures in %s", len(fig_stats["figures"]), fig_stats["figure_dir"])
            return 0

        input_path = resolve_input_path(args.input)
        raw_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        logger.info("=== Step 1/6: Data Cleaning ===")
        _, cleaning_stats = clean_dataset(input_path, DEFAULT_CLEANED_PATH)

        logger.info("=== Step 2/6: Rule-Based Classification and Relevance ===")
        _, classification_stats = classify_dataset(
            DEFAULT_CLEANED_PATH, DEFAULT_CLASSIFIED_PATH, DEFAULT_REVIEW_PATH,
            scope_review_path=DEFAULT_SCOPE_REVIEW_PATH,
            relevance_review_path=DEFAULT_RELEVANCE_REVIEW_PATH,
        )

        logger.info("=== Step 3/6: TF-IDF Keyword Analysis ===")
        _, tfidf_stats = run_tfidf_analysis(
            DEFAULT_CLASSIFIED_PATH, DEFAULT_TFIDF_PATH, top_n=args.top_n
        )

        logger.info("=== Step 4/6: LDA Topic Modeling ===")
        _, lda_stats = run_lda_analysis(
            DEFAULT_CLASSIFIED_PATH,
            DEFAULT_LDA_TOPICS_PATH,
            DEFAULT_LDA_DOC_TOPICS_PATH,
            n_topics=args.n_topics,
        )

        logger.info("=== Step 5/6: Orange Data Mining Export ===")
        _, orange_stats = export_for_orange(DEFAULT_CLASSIFIED_PATH, DEFAULT_ORANGE_PATH)

        logger.info("=== Step 6/6: Visualization (TF-IDF & LDA charts) ===")
        figure_stats = generate_all_figures()

        if hashlib.sha256(input_path.read_bytes()).hexdigest() != raw_hash:
            raise ValueError("The raw input changed during processing. Run the pipeline again.")

        outputs = (
            DEFAULT_CLEANED_PATH, DEFAULT_CLASSIFIED_PATH, DEFAULT_REVIEW_PATH, DEFAULT_RELEVANCE_REVIEW_PATH,
            DEFAULT_SCOPE_REVIEW_PATH, DEFAULT_ORANGE_PATH, DEFAULT_TFIDF_PATH,
            DEFAULT_LDA_TOPICS_PATH, DEFAULT_LDA_DOC_TOPICS_PATH,
            *(Path(path) for path in figure_stats["figures"]),
        )

        metrics = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "raw_input_sha256": raw_hash,
            "classified_csv_sha256": hashlib.sha256(DEFAULT_CLASSIFIED_PATH.read_bytes()).hexdigest(),
            "output_sha256": {
                path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in outputs
            },
            "cleaning": cleaning_stats,
            "classification": classification_stats,
            "tfidf": tfidf_stats,
            "lda": lda_stats,
            "orange_export": orange_stats,
            "figures": figure_stats,
        }
        DEFAULT_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_METRICS_PATH.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Pipeline completed successfully. Metrics saved to %s", DEFAULT_METRICS_PATH)
        return 0

    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        logger.error("Pipeline stopped due to data/input error: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive catch for CLI
        logger.exception("Unexpected error during pipeline execution: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
