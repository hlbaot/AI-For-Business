"""Explainable rule-based text classification module (upgradeable to supervised learning)."""

from __future__ import annotations

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.relevance import RELEVANCE_STATUSES, RelevanceDecision, assess_relevance, select_analysis_documents
from src.text_features import normalize_term_forms

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
DEFAULT_CLASSIFIED_PATH = PROJECT_ROOT / "data" / "processed" / "classified_data.csv"
DEFAULT_REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "manual_review_needed.csv"

CATEGORIES = ("Motivation", "Purpose", "Preparation", "Other")

STUDY_CONTEXT = re.compile(
    r"\b(?:du học|du hk|dhs|trường tiếng|trường nhật ngữ|trường chuyên môn|"
    r"senmon|visa (?:du học|học sinh|sinh viên)|tư cách lưu trú du học|"
    r"học phí|học bổng|mext|jasso|eju|học ở nhật|học tại nhật|"
    r"(?:học|vào|thi|xin vào) trường|nhập học|chuyển trường)\b"
)
LABOR_CONTEXT = re.compile(
    r"\b(?:xuất khẩu lao động|xklđ|xkld|chuyển (?:việc|(?:sang )?công ty)|"
    r"(?:nghỉ|đổi) việc|(?:đổi|nghỉ) công ty|công ty (?:cũ|mới|tiếp nhận)|"
    r"hợp đồng lao động|visa (?:lao động|đi làm|kỹ sư|kĩ sư|tokutei?)|working visa|"
    r"(?:dạng|diện|phỏng vấn) thực tập sinh|tu nghiệp sinh|"
    r"người lao động|tokutei?|kỹ năng đặc định|nghiệp đoàn|nenkin)\b"
)
RESIDENCE_CONTEXT = re.compile(
    r"\b(?:gia hạn (?:visa|viza|lưu trú|tư cách lưu trú|thẻ (?:gai|ngoại kiều))|"
    r"hồ sơ gia hạn|thủ tục gia hạn|thẻ (?:gai|ngoại kiều)|tư cách lưu trú|"
    r"thăm thân|visa (?:gia đình|kết hôn|multi)|bảo hiểm phụ thuộc)\b"
)
POSSIBLE_WORK_CONTEXT = re.compile(
    r"\b(?:đi nhật diện kỹ sư|bảng lương|lương cơ bản|môi giới|"
    r"công ty[^.?!]{0,45}xin visa|offer[^.?!]{0,65}lương)\b"
)


def assess_study_scope(text: str) -> tuple[str, str]:
    """Flag labor and residence signals that require contextual review."""
    text = normalize_term_forms(unicodedata.normalize("NFC", text).lower()).replace("_", " ")
    labor = LABOR_CONTEXT.search(text)
    residence = RESIDENCE_CONTEXT.search(text)
    has_study_context = bool(STUDY_CONTEXT.search(text))
    if not (labor or residence):
        work = POSSIBLE_WORK_CONTEXT.search(text)
        if work and not has_study_context:
            return "possible_work_context", work.group()
        return "", ""

    signals = " | ".join(match.group() for match in (labor, residence) if match)
    if has_study_context:
        return "mixed_study_and_labor_or_residence", signals
    if labor:
        return "labor_without_explicit_study_context", signals
    return "residence_without_explicit_study_context", signals

# Weighted keyword/phrase rules for each target category.
# Higher weight (2) is given to strong, unambiguous phrases.
CATEGORY_RULES: dict[str, dict[str, int]] = {
    "Motivation": {
        "yêu thích": 2,
        "đam mê": 2,
        "ngưỡng mộ": 2,
        "trải nghiệm": 1,
        "văn hóa nhật": 2,
        "văn hóa": 1,
        "anime": 2,
        "manga": 2,
        "con người nhật": 2,
        "môi trường sống": 2,
        "an toàn": 1,
        "văn minh": 2,
        "kỷ luật": 2,
        "tự lập": 2,
        "gia đình định hướng": 2,
        "bố mẹ muốn": 2,
        "bạn bè rủ": 2,
        "cơ hội việc làm": 1,
        "thu nhập cao": 2,
        "lương cao": 2,
        "áp lực": 1,
        "thay đổi bản thân": 2,
        "mở mang tầm mắt": 2,
        "lý do": 1,
        "động lực": 2,
        "thích nước nhật": 2,
        "thích nhật": 2,
    },
    "Purpose": {
        "mục tiêu": 2,
        "mục đích": 2,
        "lấy bằng": 2,
        "bằng đại học": 2,
        "bằng thạc sĩ": 2,
        "senmon": 2,
        "trường chuyên môn": 2,
        "ngành cntt": 2,
        "công nghệ thông tin": 2,
        "ngành it": 2,
        "kỹ thuật": 1,
        "điều dưỡng": 2,
        "kinh tế": 1,
        "quản trị kinh doanh": 2,
        "định cư": 2,
        "làm việc lâu dài": 2,
        "ở lại nhật": 2,
        "làm việc tại nhật": 2,
        "về việt nam làm": 2,
        "công ty nhật": 2,
        "nâng cao chuyên môn": 2,
        "trình độ chuyên môn": 2,
        "đạt n2": 2,
        "đạt n1": 2,
        "jlpt n1": 2,
        "phát triển sự nghiệp": 2,
        "tương lai": 1,
    },
    "Preparation": {
        "chuẩn bị": 2,
        "hồ sơ": 2,
        "làm hồ sơ": 2,
        "xin visa": 2,
        "visa": 1,
        "coe": 2,
        "chứng minh tài chính": 2,
        "sổ tiết kiệm": 2,
        "tiết kiệm tiền": 2,
        "học phí": 2,
        "chi phí": 1,
        "học bổng": 2,
        "săn học bổng": 2,
        "mext": 2,
        "jasso": 2,
        "học tiếng nhật": 2,
        "luyện thi": 2,
        "thi jlpt": 2,
        "jlpt n5": 2,
        "jlpt n4": 2,
        "jlpt n3": 2,
        "bằng n4": 2,
        "bằng n3": 2,
        "eju": 2,
        "ký túc xá": 2,
        "tìm nhà": 2,
        "làm thêm": 1,
        "baito": 2,
        "khám sức khỏe": 2,
        "giấy tờ": 2,
        "công chứng": 2,
    },
}


@dataclass
class ClassificationResult:
    """Structured prediction for a single text record."""

    category: str
    confidence_level: str
    needs_manual_review: bool
    review_reason: str
    matched_keywords: str
    rule_scores: str
    study_scope_flag: str = ""
    study_scope_signals: str = ""
    relevance: RelevanceDecision | None = None


class BaseTextClassifier(ABC):
    """Abstract interface allowing seamless upgrade from rules to supervised ML."""

    @abstractmethod
    def predict_one(self, text_cleaned: str, text_segmented: str = "") -> ClassificationResult:
        """Classify a single document."""

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify all rows in a cleaned DataFrame."""
        if "text_cleaned" not in df.columns:
            raise ValueError("Input DataFrame must contain 'text_cleaned' column.")

        seg_col = df["text_segmented"] if "text_segmented" in df.columns else pd.Series([""] * len(df))
        results = [
            self.predict_one(str(cleaned), str(seg))
            for cleaned, seg in zip(df["text_cleaned"].fillna(""), seg_col.fillna(""))
        ]

        out = df.copy()
        out["category"] = [r.category for r in results]
        out["confidence_level"] = [r.confidence_level for r in results]
        out["needs_manual_review"] = [r.needs_manual_review for r in results]
        out["review_reason"] = [r.review_reason for r in results]
        out["matched_keywords"] = [r.matched_keywords for r in results]
        out["rule_scores"] = [r.rule_scores for r in results]
        out["study_scope_flag"] = [r.study_scope_flag for r in results]
        out["study_scope_signals"] = [r.study_scope_signals for r in results]
        decisions = [
            result.relevance or assess_relevance(str(text))
            for result, text in zip(results, df["text_cleaned"].fillna(""))
        ]
        out["relevance_status"] = [decision.status for decision in decisions]
        out["relevance_reason"] = [decision.reason for decision in decisions]
        out["needs_manual_review"] |= out["relevance_status"].eq("uncertain")
        return out


class RuleBasedClassifier(BaseTextClassifier):
    """Explainable keyword/phrase rule classifier that flags ambiguous cases."""

    def __init__(self, rules: dict[str, dict[str, int]] | None = None) -> None:
        self.rules = rules or CATEGORY_RULES
        self.patterns = {
            keyword: re.compile(rf"\b{re.escape(keyword)}\b")
            for weights in self.rules.values()
            for keyword in weights
        }

    def _score_text(self, text_cleaned: str, text_segmented: str) -> tuple[dict[str, int], dict[str, list[str]]]:
        haystack_clean = text_cleaned.lower()
        haystack_seg = text_segmented.lower().replace("_", " ")

        scores: dict[str, int] = {}
        matched: dict[str, list[str]] = {}

        for category, kw_weights in self.rules.items():
            cat_score = 0
            cat_matches: list[str] = []
            for kw, weight in kw_weights.items():
                pattern = self.patterns[kw]
                if pattern.search(haystack_clean) or pattern.search(haystack_seg):
                    cat_score += weight
                    cat_matches.append(kw)
            scores[category] = cat_score
            matched[category] = cat_matches

        return scores, matched

    def predict_one(self, text_cleaned: str, text_segmented: str = "") -> ClassificationResult:
        scores, matched = self._score_text(text_cleaned, text_segmented)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_cat, top_score = ranked[0]
        second_cat, second_score = ranked[1]

        score_summary = ", ".join(f"{cat}={sc}" for cat, sc in ranked)
        scope_flag, scope_signals = assess_study_scope(text_cleaned)
        relevance = assess_relevance(text_cleaned)
        scope_fields = dict(study_scope_flag=scope_flag, study_scope_signals=scope_signals, relevance=relevance)
        if relevance.status == "off_topic":
            return ClassificationResult(
                "Other", "low", True, relevance.reason, "; ".join(matched[top_cat]),
                score_summary, **scope_fields,
            )

        if top_score == 0:
            return ClassificationResult(
                category="Other",
                confidence_level="low",
                needs_manual_review=True,
                review_reason="no_matching_keywords",
                matched_keywords="",
                rule_scores=score_summary,
                **scope_fields,
            )

        if top_score == second_score:
            combined_kw = matched[top_cat] + matched[second_cat]
            return ClassificationResult(
                category="Other",
                confidence_level="ambiguous",
                needs_manual_review=True,
                review_reason=f"ambiguous_tie_{top_cat}_vs_{second_cat}",
                matched_keywords="; ".join(dict.fromkeys(combined_kw)),
                rule_scores=score_summary,
                **scope_fields,
            )

        margin = top_score - second_score
        if second_score > 0:
            confidence = "medium"
            needs_review = True
            match_type = "close_secondary_category" if margin == 1 else "multiple_category_matches"
            reason = f"{match_type}:{second_cat}"
        elif top_score == 1:
            confidence = "low"
            needs_review = True
            reason = "single_weak_keyword"
        else:
            confidence = "high"
            needs_review = False
            reason = ""

        if scope_flag or relevance.status == "uncertain":
            confidence = "medium"
            needs_review = True
            relevance_note = relevance.reason if relevance.status == "uncertain" else ""
            reason = " | ".join(filter(None, (reason, scope_flag, relevance_note)))

        return ClassificationResult(
            category=top_cat,
            confidence_level=confidence,
            needs_manual_review=needs_review,
            review_reason=reason,
            matched_keywords="; ".join(matched[top_cat]),
            rule_scores=score_summary,
            **scope_fields,
        )


def classify_dataset(
    input_path: Path | str = DEFAULT_CLEANED_PATH,
    output_path: Path | str = DEFAULT_CLASSIFIED_PATH,
    review_output_path: Path | str | None = DEFAULT_REVIEW_PATH,
    classifier: BaseTextClassifier | None = None,
    scope_review_path: Path | str | None = None,
    relevance_review_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classify cleaned texts and optionally export records needing manual review."""
    in_path = Path(input_path)
    out_path = Path(output_path)

    if not in_path.exists() or in_path.stat().st_size == 0:
        raise ValueError(f"Cleaned dataset is missing or empty: {in_path}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Cleaned dataset has 0 rows: {in_path}")

    clf = classifier or RuleBasedClassifier()
    classified_df = clf.predict_dataframe(df)
    if "relevance_status" not in classified_df.columns:
        decisions = classified_df["text_cleaned"].fillna("").map(assess_relevance)
        classified_df["relevance_status"] = decisions.map(lambda decision: decision.status)
        classified_df["relevance_reason"] = decisions.map(lambda decision: decision.reason)
    classified_df.loc[classified_df["relevance_status"].eq("off_topic"), "category"] = "Other"
    classified_df.loc[classified_df["relevance_status"].eq("uncertain"), "needs_manual_review"] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    classified_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    review_df = classified_df.loc[classified_df["needs_manual_review"]].copy()
    if review_output_path is not None:
        rev_path = Path(review_output_path)
        rev_path.parent.mkdir(parents=True, exist_ok=True)
        review_df.to_csv(rev_path, index=False, encoding="utf-8-sig")

    scope_flags = classified_df.get("study_scope_flag", pd.Series("", index=classified_df.index))
    flagged = scope_flags.ne("")
    if scope_review_path is not None:
        scope_flagged = flagged | classified_df["relevance_status"].eq("off_topic")
        scope_review = classified_df.loc[scope_flagged, [
            "text_id", "category", "text_cleaned", "relevance_status", "relevance_reason",
        ]].copy()
        scope_review["study_scope_flag"] = scope_flags.loc[scope_flagged]
        scope_review["study_scope_signals"] = classified_df.get("study_scope_signals", "")
        scope_review["manual_review_status"] = "pending_manual_review"
        scope_path = Path(scope_review_path)
        scope_path.parent.mkdir(parents=True, exist_ok=True)
        scope_review.to_csv(scope_path, index=False, encoding="utf-8-sig")

    if relevance_review_path is not None:
        relevance_review = classified_df[[
            "text_id", "text_cleaned", "category", "relevance_status", "relevance_reason",
        ]].copy()
        relevance_review["manual_review_status"] = [
            RelevanceDecision(status, "").manual_review_status for status in relevance_review["relevance_status"]
        ]
        relevance_path = Path(relevance_review_path)
        relevance_path.parent.mkdir(parents=True, exist_ok=True)
        relevance_review.to_csv(relevance_path, index=False, encoding="utf-8-sig")

    category_counts = {
        cat: int((classified_df["category"] == cat).sum()) for cat in CATEGORIES
    }
    stats: dict[str, Any] = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "manual_review_output_path": str(review_output_path) if review_output_path else None,
        "total_classified": len(classified_df),
        "category_distribution": category_counts,
        "manual_review_count": len(review_df),
        "relevance": {
            "review_output_path": str(relevance_review_path) if relevance_review_path else None,
            "status_counts": classified_df["relevance_status"].value_counts().reindex(RELEVANCE_STATUSES, fill_value=0).to_dict(),
            "analysis_category_distribution": select_analysis_documents(classified_df)["category"].value_counts().to_dict(),
        },
        "study_scope_flag_counts": scope_flags.loc[flagged].value_counts().to_dict(),
        "accuracy": None,
        "evaluation_note": (
            "Experimental rule-based classifier; accuracy is not reported because "
            "no ground-truth labeled evaluation set is available yet."
        ),
    }
    logger.info(
        "Classification complete: %s | Manual review needed: %d/%d",
        category_counts,
        len(review_df),
        len(classified_df),
    )
    return classified_df, stats
