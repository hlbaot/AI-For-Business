"""Visualization module: render TF-IDF and LDA results into PNG charts (matplotlib only)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_FIG_DIR = RESULTS_DIR / "figures"
DEFAULT_TFIDF_PATH = RESULTS_DIR / "tfidf_results.csv"
DEFAULT_LDA_TOPICS_PATH = RESULTS_DIR / "lda_topics.csv"
DEFAULT_LDA_DOC_TOPICS_PATH = RESULTS_DIR / "lda_doc_topics.csv"
DEFAULT_CLASSIFIED_PATH = PROJECT_ROOT / "data" / "processed" / "classified_data.csv"

CATEGORY_ORDER = ["Motivation", "Purpose", "Preparation"]
CATEGORY_COLORS = {
    "ALL": "#4C72B0",
    "Motivation": "#DD8452",
    "Purpose": "#55A868",
    "Preparation": "#C44E52",
    "Other": "#8C8C8C",
}
TFIDF_FILE_NAMES = {
    "Motivation": "03_tfidf_nhom_dong_luc_motivation.png",
    "Purpose": "04_tfidf_nhom_muc_dich_purpose.png",
    "Preparation": "05_tfidf_nhom_chuan_bi_preparation.png",
}
TOPIC_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",  # supports Vietnamese diacritics
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _label(word: str) -> str:
    """Make segmented tokens readable (du_học -> du học)."""
    return str(word).replace("_", " ")


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return str(path)


def _hbar(ax: plt.Axes, words: list[str], values: list[float], color: str, title: str, xlabel: str) -> None:
    y = np.arange(len(words))
    ax.barh(y, values, color=color, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([_label(w) for w in words])
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8)
    ax.margins(x=0.15)


# ---------------------------------------------------------------- TF-IDF
def plot_tfidf(tfidf_path: Path, fig_dir: Path) -> list[str]:
    df = pd.read_csv(tfidf_path, encoding="utf-8-sig")
    saved: list[str] = []

    # 1. Overall top keywords
    all_df = df[df["group"] == "ALL"].sort_values("rank")
    if not all_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        _hbar(
            ax,
            all_df["keyword"].tolist(),
            all_df["tfidf_score"].tolist(),
            CATEGORY_COLORS["ALL"],
            f"Top {len(all_df)} TF-IDF keywords – All 3 groups combined (n={int(all_df['group_doc_count'].iloc[0])})",
            "Mean TF-IDF score",
        )
        saved.append(_save(fig, fig_dir / "02_tfidf_tu_khoa_tong_the.png"))

    # 2. One image per main category
    cats = [c for c in CATEGORY_ORDER if c in df["group"].unique()]
    if cats:
        for cat in [c for c in cats if c in TFIDF_FILE_NAMES]:
            sub = df[df["group"] == cat].sort_values("rank")
            fig, ax = plt.subplots(figsize=(9, 6))
            _hbar(
                ax,
                sub["keyword"].tolist(),
                sub["tfidf_score"].tolist(),
                CATEGORY_COLORS.get(cat, "#4C72B0"),
                f"Top TF-IDF keywords – {cat} (n={int(sub['group_doc_count'].iloc[0])})",
                "Mean TF-IDF score",
            )
            saved.append(_save(fig, fig_dir / f"{TFIDF_FILE_NAMES[cat]}"))
    return saved


# ---------------------------------------------------------------- LDA
def plot_lda(topics_path: Path, doc_topics_path: Path, fig_dir: Path) -> list[str]:
    saved: list[str] = []
    topics = pd.read_csv(topics_path, encoding="utf-8-sig")
    topic_ids = sorted(topics["topic_id"].unique())

    # 1. Top words per topic
    n = len(topic_ids)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6))
    axes = np.atleast_1d(axes)
    for ax, tid in zip(axes, topic_ids):
        sub = topics[topics["topic_id"] == tid].sort_values("rank")
        _hbar(
            ax,
            sub["keyword"].tolist(),
            sub["weight"].tolist(),
            TOPIC_COLORS[tid % len(TOPIC_COLORS)],
            f"Topic_{tid}",
            "Word probability within topic",
        )
    fig.suptitle("LDA – Top keywords per topic", fontsize=15, fontweight="bold")
    saved.append(_save(fig, fig_dir / "06_lda_tu_khoa_tung_chu_de.png"))

    if not doc_topics_path.exists():
        return saved
    docs = pd.read_csv(doc_topics_path, encoding="utf-8-sig")
    topic_labels = [f"Topic_{t}" for t in topic_ids]

    # 2. Dominant topic distribution
    counts = docs["dominant_topic"].value_counts().reindex(topic_labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=[TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in topic_ids])
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / total:.1%})", ha="center", va="bottom")
    ax.set_title(f"Documents per dominant LDA topic (n={total})", fontweight="bold")
    ax.set_ylabel("Number of documents")
    ax.margins(y=0.18)
    saved.append(_save(fig, fig_dir / "07_lda_phan_bo_chu_de.png"))

    # 3. Heatmap: LDA topic vs rule-based category
    if "rule_category" in docs.columns:
        ct = pd.crosstab(docs["rule_category"], docs["dominant_topic"])
        ct = ct.reindex(index=[c for c in CATEGORY_ORDER if c in ct.index], columns=topic_labels, fill_value=0)
        pct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0)
        fig, ax = plt.subplots(figsize=(1.8 * n + 3, 4.5))
        im = ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pct.columns)))
        ax.set_xticklabels(pct.columns)
        ax.set_yticks(range(len(pct.index)))
        ax.set_yticklabels(pct.index)
        for i in range(pct.shape[0]):
            for j in range(pct.shape[1]):
                val = pct.values[i, j]
                ax.text(j, i, f"{val:.0%}\n({ct.values[i, j]})", ha="center", va="center",
                        color="white" if val > 0.55 else "black", fontsize=9)
        ax.set_title("LDA topic distribution by category (row %)", fontweight="bold")
        ax.set_xlabel("Dominant LDA topic")
        ax.set_ylabel("Category (rule-based)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        saved.append(_save(fig, fig_dir / "08_lda_chu_de_vs_nhom_phan_loai.png"))

    return saved


# ---------------------------------------------------------------- Categories
def plot_category_distribution(classified_path: Path, fig_dir: Path) -> list[str]:
    if not classified_path.exists():
        return []
    df = pd.read_csv(classified_path, encoding="utf-8-sig")
    if "category" not in df.columns:
        return []
    counts = df["category"].value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = [CATEGORY_COLORS[c] for c in counts.index]
    bars = ax1.bar(counts.index, counts.values, color=colors)
    for b, v in zip(bars, counts.values):
        ax1.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    ax1.set_title("Documents per category", fontweight="bold")
    ax1.set_ylabel("Number of documents")
    ax1.margins(y=0.12)
    ax2.pie(counts.values, labels=counts.index, colors=colors, autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white"})
    ax2.set_title("Category share", fontweight="bold")
    fig.suptitle(f"Rule-based category distribution (n={counts.sum()})", fontsize=14, fontweight="bold")
    return [_save(fig, fig_dir / "01_phan_bo_nhom_phan_loai.png")]


def generate_all_figures(
    tfidf_path: Path | str = DEFAULT_TFIDF_PATH,
    lda_topics_path: Path | str = DEFAULT_LDA_TOPICS_PATH,
    lda_doc_topics_path: Path | str = DEFAULT_LDA_DOC_TOPICS_PATH,
    classified_path: Path | str = DEFAULT_CLASSIFIED_PATH,
    fig_dir: Path | str = DEFAULT_FIG_DIR,
) -> dict[str, Any]:
    """Render every chart and return the list of saved files."""
    fig_dir = Path(fig_dir)
    saved: list[str] = []
    saved += plot_category_distribution(Path(classified_path), fig_dir)
    saved += plot_tfidf(Path(tfidf_path), fig_dir)
    saved += plot_lda(Path(lda_topics_path), Path(lda_doc_topics_path), fig_dir)
    logger.info("Visualization complete: %d figures saved to %s", len(saved), fig_dir)
    return {"figure_dir": str(fig_dir), "figures": saved}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    generate_all_figures()
