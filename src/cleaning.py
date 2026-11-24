"""Data cleaning and Vietnamese text preprocessing module."""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from underthesea import word_tokenize

from src.text_features import EMOJI_SHORTCODE, normalize_term_forms, vietnamese_segmented_tokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "raw_data.csv"
DEFAULT_CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
PROJECT_WORKBOOK = DEFAULT_RAW_PATH.with_name("DATA_COLLECTION_TEXT_ANALYSIS_STUDY_IN_JAPAN_P2.xlsx")
PROJECT_REVIEW_PATH = PROJECT_ROOT / "data" / "reviewed_exclusions.csv"

MIN_VALID_DATE = pd.Timestamp("2022-02-01")
REQUIRED_COLUMNS = ("text", "created_at")

# Explainable spam / advertisement / irrelevant content rules
SPAM_KEYWORDS: tuple[str, ...] = (
    "liên hệ ngay",
    "inbox ngay",
    "ib ngay",
    "tư vấn miễn phí",
    "cam kết đậu",
    "bán sim",
    "chốt đơn",
    "giảm giá sốc",
    "link bio",
    "đăng ký ngay hôm nay",
)
TEMPLATE_INTRO_PATTERN = re.compile(r"^(?:dưới đây là|đây là) (?:nội dung |một )?bình luận khoảng \d+")

IRRELEVANT_EXACT: frozenset[str] = frozenset(
    {
        "ok",
        "oke",
        "hóng",
        "chấm",
        ".",
        "..",
        "...",
        "ib",
        "inbox",
        "up",
        "test",
    }
)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")
MENTION_PATTERN = re.compile(r"(?<!\w)@[\w.-]+")
PERSON_NAME_PATTERN = re.compile(
    r"(\b(?i:chào|bạn|anh|chị|tên (?:mình|em|tôi) là)\s+)"
    r"(?:Nguyễn|Trần|Lê|Phạm|Hoàng|Huỳnh|Phan|Vũ|Võ|Đặng|Bùi|Đỗ|Hồ|Ngô|Dương)"
    r"(?:\s+[A-ZÀ-ỸĐ][a-zà-ỹđ]{1,}){1,3}\b"
)
ID_NUMBER_PATTERN = re.compile(r"\b\d{9,12}\b")
ALLOWED_CHAR_PATTERN = re.compile(r"[^\w\s.,?!/:;()\-+%\[\]]", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
SEGMENT_PUNCT_PATTERN = re.compile(r"^[.,?!:;()\-+/%]+$")


def read_csv_safely(path: Path) -> pd.DataFrame:
    """Read a CSV file with UTF-8/UTF-8-SIG support and clear error handling."""
    for encoding in ("utf-8-sig", "utf-8", "cp1258"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            if df.empty:
                raise ValueError(f"Input CSV has headers but 0 rows: {path}")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Input CSV contains no parseable data: {path}") from exc

    raise ValueError(f"Unable to decode CSV file with supported encodings: {path}")


def read_excel_safely(path: Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Read an Excel (.xlsx/.xls) file, auto-detecting the DATA sheet and header row."""
    xl = pd.ExcelFile(path)
    target_sheet: str | int
    if sheet_name is not None:
        target_sheet = sheet_name
    elif "DATA" in xl.sheet_names:
        target_sheet = "DATA"
    else:
        target_sheet = xl.sheet_names[0]

    df = pd.read_excel(path, sheet_name=target_sheet, header=0)
    # Handle template sheets where row 0 is a title banner and row 1 is the header
    unnamed_ratio = sum(str(c).startswith("Unnamed:") for c in df.columns) / max(len(df.columns), 1)
    if unnamed_ratio >= 0.5 and not df.empty:
        first_row_vals = {str(v).strip().lower() for v in df.iloc[0].dropna().tolist()}
        if first_row_vals & {"text_id", "text", "text_raw", "text_anonymized", "post_date", "created_at"}:
            df = pd.read_excel(path, sheet_name=target_sheet, header=1)

    if df.empty:
        raise ValueError(f"Input Excel sheet '{target_sheet}' has 0 rows: {path}")
    return df


def read_input_safely(path: Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Read raw input data from either CSV or Excel (.xlsx/.xls)."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_safely(path)
    if suffix in {".xlsx", ".xls"}:
        return read_excel_safely(path, sheet_name=sheet_name)
    raise ValueError(f"Unsupported input format '{path.suffix}'. Expected .csv or .xlsx/.xls.")


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column aliases, validate required columns, and ensure `text_id` exists."""
    out = df.copy()

    if "text" not in out.columns:
        for alias in ("text_raw", "text_anonymized", "content", "comment"):
            if alias in out.columns:
                out["text"] = out[alias]
                break

    if "created_at" not in out.columns:
        for alias in ("post_date", "date", "collected_date"):
            if alias in out.columns:
                out["created_at"] = out[alias]
                break

    if "source" not in out.columns and "source_platform" in out.columns:
        out["source"] = out["source_platform"]

    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Expected at least {list(REQUIRED_COLUMNS)} (or aliases like text_raw/post_date), "
            f"got {list(out.columns)}."
        )

    if "text_id" not in out.columns:
        out["text_id"] = [f"TXT_{i:04d}" for i in range(1, len(out) + 1)]
    else:
        missing_id = out["text_id"].isna() | (out["text_id"].astype(str).str.strip() == "")
        if missing_id.any():
            out.loc[missing_id, "text_id"] = [
                f"TXT_{idx + 1:04d}" for idx in out.index[missing_id]
            ]
    return out


def detect_spam_or_irrelevant(text: str) -> tuple[bool, str | None]:
    """Check whether a comment is spam, an ad, or irrelevant using explainable rules."""
    lowered = unicodedata.normalize("NFC", text).lower().strip()
    if lowered in IRRELEVANT_EXACT:
        return True, "irrelevant_short_placeholder"
    if TEMPLATE_INTRO_PATTERN.match(lowered):
        return True, "comment_template_intro"

    # Strip emojis and URLs when checking meaningful length
    no_emoji = EMOJI_PATTERN.sub(" ", lowered)
    no_url = URL_PATTERN.sub(" ", no_emoji)
    words = [w for w in re.split(r"\s+", no_url.strip()) if re.search(r"\w", w)]

    if len(words) < 3 or len(" ".join(words)) < 10:
        return True, "irrelevant_too_short"

    for kw in SPAM_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            return True, f"spam_keyword:{kw}"

    url_count = len(URL_PATTERN.findall(lowered))
    has_phone = bool(PHONE_PATTERN.search(lowered))
    recruitment = bool(re.search(r"\b(?:tuyển dụng|chốt đơn|đăng ký ngay)\b", lowered))
    if url_count and (has_phone or recruitment):
        return True, "spam_promotional_links_or_contact"

    return False, None


def anonymize_pii(text: str) -> str:
    """Mask contact details, identifiers, and names in common forms of address."""
    masked = EMAIL_PATTERN.sub("[EMAIL]", unicodedata.normalize("NFC", text))
    masked = URL_PATTERN.sub("[URL]", masked)
    masked = PHONE_PATTERN.sub("[PHONE]", masked)
    masked = MENTION_PATTERN.sub("[USER]", masked)
    masked = PERSON_NAME_PATTERN.sub(r"\1[USER]", masked)
    masked = ID_NUMBER_PATTERN.sub("[ID]", masked)
    return masked


def normalize_vietnamese_text(text: str) -> str:
    """Normalize Unicode, remove emojis, and clean spacing while preserving diacritics and numbers."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = EMOJI_PATTERN.sub(" ", normalized)
    normalized = EMOJI_SHORTCODE.sub(" ", normalized)
    normalized = ALLOWED_CHAR_PATTERN.sub(" ", normalized)
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return normalized


PII_PLACEHOLDER_PATTERN = re.compile(r"\[(?:EMAIL|URL|PHONE|USER|ID)\]", flags=re.IGNORECASE)


def segment_vietnamese_text(text: str) -> str:
    """Segment Vietnamese words with underthesea and keep domain terms intact."""
    if not text:
        return ""

    # Remove anonymization placeholders before word_tokenize so underthesea does not split brackets
    text_for_nlp = WHITESPACE_PATTERN.sub(" ", PII_PLACEHOLDER_PATTERN.sub(" ", text)).strip()
    if not text_for_nlp:
        return ""

    # Tokenize with underthesea; format='text' joins compound words with '_'
    segmented = word_tokenize(text_for_nlp, format="text")
    tokens: list[str] = []
    for raw_tok in segmented.split():
        tok = raw_tok.strip(".,?!:;\"'()").lower()
        if not tok or SEGMENT_PUNCT_PATTERN.match(tok):
            continue
        tokens.append(tok)

    return normalize_term_forms(" ".join(tokens))


def clean_dataset(
    input_path: Path | str = DEFAULT_RAW_PATH,
    output_path: Path | str = DEFAULT_CLEANED_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full cleaning pipeline without modifying the raw input file."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    if in_path.resolve() == out_path.resolve() or (out_path.exists() and in_path.exists() and out_path.samefile(in_path)):
        raise ValueError("The cleaning output must differ from the raw input file.")

    logger.info("Reading raw data from %s", in_path)
    raw_df = read_input_safely(in_path)
    df = validate_columns(raw_df)

    total_raw = len(df)
    drop_reasons: dict[str, int] = {}

    def record_drop(reason: str, count: int) -> None:
        if count > 0:
            drop_reasons[reason] = drop_reasons.get(reason, 0) + int(count)

    # 1. Drop empty / null text rows
    text_series = df["text"].fillna("").astype(str).str.strip()
    if "text_anonymized" in df.columns:
        anonymized = df["text_anonymized"].fillna("").astype(str).str.strip()
        text_series = text_series.where(text_series.ne(""), anonymized)
    empty_mask = text_series == ""
    record_drop("empty_or_null_text", int(empty_mask.sum()))
    df = df.loc[~empty_mask].copy()
    df["text"] = text_series.loc[df.index]

    # 2. Filter dates before deduplication so an out-of-range copy cannot suppress a valid row.
    parsed_dates = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    record_drop("invalid_date", int(parsed_dates.isna().sum()))
    record_drop("before_or_in_jan_2022", int(parsed_dates.lt(MIN_VALID_DATE).sum()))
    valid_date_mask = parsed_dates.ge(MIN_VALID_DATE)
    df = df.loc[valid_date_mask].copy()
    df["created_at"] = parsed_dates.loc[valid_date_mask].dt.strftime("%Y-%m-%d")

    # 3. Drop duplicate texts (case- and whitespace-normalized)
    dedup_key = (
        df["text"]
        .apply(lambda s: unicodedata.normalize("NFC", s).lower())
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    dup_mask = dedup_key.duplicated(keep="first")
    record_drop("duplicate_text", int(dup_mask.sum()))
    df = df.loc[~dup_mask].copy()

    # 4. Filter spam, ads, and irrelevant content
    if in_path.resolve() == PROJECT_WORKBOOK.resolve() and PROJECT_REVIEW_PATH.is_file():
        reviewed_ids = pd.read_csv(PROJECT_REVIEW_PATH, dtype={"text_id": str})["text_id"]
        reviewed_exclusions = df["text_id"].isin(reviewed_ids)
        record_drop("manual_review_exclusion", int(reviewed_exclusions.sum()))
        df = df.loc[~reviewed_exclusions].copy()

    spam_flags = df["text"].apply(detect_spam_or_irrelevant)
    is_spam = spam_flags.apply(lambda x: x[0])
    for reason in spam_flags.loc[is_spam].apply(lambda x: x[1]):
        record_drop(str(reason), 1)
    df = df.loc[~is_spam].copy()

    # 5. Anonymize PII, normalize Vietnamese, and word-segment
    text_for_analysis = df["text"]
    if "text_anonymized" in df.columns:
        anonymized = df["text_anonymized"].fillna("").astype(str).str.strip()
        text_for_analysis = anonymized.where(anonymized.ne(""), text_for_analysis)
    df["text_cleaned"] = text_for_analysis.apply(anonymize_pii).apply(normalize_vietnamese_text)
    df["text_segmented"] = df["text_cleaned"].apply(segment_vietnamese_text)

    # Drop any rows that became empty after cleaning/segmentation
    post_empty_mask = (df["text_cleaned"].str.strip() == "") | (
        df["text_segmented"].str.strip() == ""
    )
    record_drop("empty_after_cleaning", int(post_empty_mask.sum()))
    df = df.loc[~post_empty_mask].copy()

    has_content = df["text_segmented"].map(
        lambda text: bool(vietnamese_segmented_tokenizer(text))
    ).astype(bool)
    noise_only = ~has_content
    record_drop("no_content_terms", int(noise_only.sum()))
    df = df.loc[~noise_only].copy()

    # Drop potential PII metadata columns like author/username/email if present
    pii_cols = [c for c in ("author", "username", "user", "email", "phone") if c in df.columns]
    if pii_cols:
        df = df.drop(columns=pii_cols)

    keep_cols = ["text_id", "created_at", "text_cleaned", "text_segmented"]
    if "source" in df.columns:
        keep_cols.append("source")
    df_out = df[keep_cols].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")

    stats: dict[str, Any] = {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "total_raw_records": total_raw,
        "kept_records": len(df_out),
        "dropped_records": total_raw - len(df_out),
        "drop_reasons": drop_reasons,
    }
    logger.info(
        "Cleaning complete: %d raw -> %d kept (%d dropped). Reasons: %s",
        stats["total_raw_records"],
        stats["kept_records"],
        stats["dropped_records"],
        drop_reasons,
    )
    return df_out, stats
