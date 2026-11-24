"""Shared vocabulary preparation for TF-IDF, LDA and Orange."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

STOPWORDS_PATH = Path(__file__).resolve().parent.parent / "data/exports/vietnamese_stopwords.txt"
VIETNAMESE_STOPWORDS = frozenset(STOPWORDS_PATH.read_text(encoding="utf-8-sig").split())

# Match across both word boundaries produced by underthesea and plain spaces.
# Protect meaningful phrases before dropping individual words such as "xin".
TERM_FORMS = {
    "tiếng_nhật": ("tiếng nhật",),
    "tiếng_anh": ("tiếng anh",),
    "nhật_bản": ("nhật bản",),
    "việt_nam": ("việt nam", "vn"),
    "kỹ_sư": ("kỹ sư", "kĩ sư"),
    "ký_túc_xá": ("ký túc xá", "kí túc xá", "ktx"),
    "công_ty": ("công ty", "cty"),
    "đại_học": ("đại học", "đh"),
    "công_nghệ_thông_tin": ("công nghệ thông tin", "cntt"),
    "học_bổng": ("học bổng",),
    "học_phí": ("học phí",),
    "du_học": ("du học", "du hk"),
    "du_học_sinh": ("du học sinh", "dhs"),
    "hồ_sơ": ("hồ sơ",),
    "giấy_tờ": ("giấy tờ",),
    "xin_visa": ("xin visa", "xin viza"),
    "gia_hạn_visa": ("gia hạn visa", "gia hạn viza"),
    "visa": ("viza",),
    "xin_việc": ("xin việc",),
    "xin_học_bổng": ("xin học bổng",),
    "thu_nhập": ("thu nhập",),
    "lương_cao": ("lương cao",),
    "làm_thêm": ("làm thêm", "baito"),
    "làm_việc": ("làm việc",),
    "việc_làm": ("việc làm",),
    "chi_phí": ("chi phí",),
    "tốt_nghiệp": ("tốt nghiệp",),
    "xuất_khẩu_lao_động": ("xuất khẩu lao động", "xklđ", "xkld"),
    "thực_tập_sinh": ("thực tập sinh", "tts"),
    "cảm_ơn": ("cảm ơn", "cám ơn", "cảm ươn", "cam on"),
    "xin_hỏi": ("xin hỏi", "cho hỏi"),
    "có_thể": ("có thể",),
    "như_thế_nào": ("như thế nào", "như nào"),
}
_FORM_TO_TERM = {form.replace(" ", "_"): term for term, forms in TERM_FORMS.items() for form in forms}
_TERM_PATTERN = re.compile(
    r"(?<![^\W_])(?:"
    + "|".join(
        r"[ _]+".join(map(re.escape, form.split("_")))
        for form in sorted(_FORM_TO_TERM, key=len, reverse=True)
    )
    + r")(?![^\W_])"
)
_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_EXAM_CODE = re.compile(r"(?<![^\W_])(jlpt|eju|n[1-5])(?![^\W_])")
_FILLER_PARTS = frozenset(
    "ơi ạ nha nhé nhe nhéee á nè nhỉ ơii ơiii ạaa ko hok k kh hong hông khum dc đc".split()
)
_PLACEHOLDER = re.compile(r"\[(?:EMAIL|URL|PHONE|USER|ID)\]", re.IGNORECASE)
EMOJI_SHORTCODE = re.compile(r":[a-z0-9]+(?:[-_][a-z0-9]+)+:", re.IGNORECASE)
_NOISE = re.compile(r"(?:(?:ha|he|hi){2,}|[kh]+|ạ+|ơ+i+|\d+(?:tr|k|kg|man)?)")


def normalize_term_forms(text: str) -> str:
    """Unify selected Vietnamese compounds and unambiguous domain abbreviations."""
    normalized = _TERM_PATTERN.sub(
        lambda match: _FORM_TO_TERM[re.sub(r"[ _]+", "_", match.group())], text
    )
    # Underthesea can group adjacent codes as jlpt_n2_n3.
    parts = _EXAM_CODE.sub(r" \1 ", normalized).split()
    return " ".join(token for part in parts if (token := part.strip("_")))


def vietnamese_segmented_tokenizer(text: str) -> list[str]:
    """Normalize Vietnamese terms and remove conversational and numeric noise."""
    if not isinstance(text, str):
        return []

    normalized = unicodedata.normalize("NFC", text).lower()
    normalized = EMOJI_SHORTCODE.sub(" ", _PLACEHOLDER.sub(" ", normalized))
    normalized = normalize_term_forms(normalized)
    tokens: list[str] = []
    for raw_token in _WORD_PATTERN.findall(normalized):
        token = "_".join(
            part for part in raw_token.split("_") if part and part not in _FILLER_PARTS
        )
        if len(token) < 2 or token in VIETNAMESE_STOPWORDS or _NOISE.fullmatch(token):
            continue
        if any(char.isalpha() for char in token):
            tokens.append(token)
    return tokens
