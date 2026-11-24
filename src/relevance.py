"""Assess whether cleaned comments provide evidence about studying in Japan."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from src.text_features import normalize_term_forms

RELEVANCE_STATUSES = ("relevant", "off_topic", "uncertain")

# Look for specific activities or compatible cues. Generic words such as visa and
# lương carry no decision on their own. A distinct labor route can coexist with study.
JAPAN = re.compile(r"\b(?:(?<!tiếng )nhật(?: bản)?(?! ngữ\b)|tokyo|osaka|beppu|apu|yên|jpy)\b")
STUDY_REFERENCE = re.compile(r"\b(?:du học(?: sinh)?|trường tiếng|trường nhật ngữ|senmon|cao học)\b")
ACADEMIC_ACTION = re.compile(
    r"\b(?:muốn học|đi học|học tiếp|học lên|chọn trường|chọn ngành|học ngành|đăng ký|đăng kí|"
    r"tìm hiểu|dự định|chuẩn bị|chi phí|học phí|học bổng|hồ sơ|xin coe|"
    r"nhập học|tuyển sinh|nộp đơn|apply|tốt nghiệp|làm thêm|baito|học tiếng nhật|thi|động lực|kế hoạch)\b"
)
APPLICATION = re.compile(
    r"\b(?:học bổng|học phí|nhập học|tuyển sinh|chọn trường|chọn ngành|học ngành|"
    r"học đại học|học thạc sĩ|học cao học|học nghiên cứu sinh|trường đại học|"
    r"đề cương nghiên cứu|research proposal|nguyện vọng)\b"
)
SPECIFIC_STUDY = re.compile(
    r"\b(?:visa (?:diện )?(?:du học|sinh viên|học sinh)|"
    r"du học (?:(?:ở|tại|bên) )?nhật(?: bản)?|"
    r"(?:sang nhật|ở nhật|tại nhật) (?:để )?học (?:đại học|thạc sĩ|cao học|trường tiếng)|"
    r"học (?:đại học|thạc sĩ|cao học|nghiên cứu sinh) (?:ở|tại|bên) nhật|"
    r"(?:qua|sang) nhật (?:để )?học tiếng(?: nhật)?|học tiếng nhật[^.!?]{0,25}để du học|"
    r"(?:theo học|đang học|nhập học)[^.!?]{0,35}(?:đại học tokyo|đại học kyoto)|"
    r"(?:học|vào|chọn|xin vào) (?:trường tiếng|senmon|trường nhật ngữ)|"
    r"(?:xin|săn|nộp|tìm|đăng ký) học bổng[^.!?]{0,36}(?:mext|jasso|asahi|yomiuri)|"
    r"học bổng (?:báo|mext|jasso|asahi|yomiuri)|"
    r"(?:học|học ở|học tại|học trường|chọn) apu)\b"
)
JP_EXAM_FUND = re.compile(r"\b(?:eju|mext|jasso|học bổng báo)\b")
EXAM_FUND_CONTEXT = re.compile(
    r"\b(?:học|thi|ôn|luyện|đăng ký|đăng kí|hỗ trợ|hồ sơ|nộp|apply|điều kiện|"
    r"đề cương|toán|lý|hóa|điểm|học bổng|nhập học)\b"
)
LANGUAGE_OR_EXAM = re.compile(r"\b(?:tiếng nhật|jlpt|n[1-5]|j-test)\b")
STUDENT_COST = re.compile(r"\b(?:học phí|học bổng|trợ cấp sinh hoạt|miễn học phí)\b")
STUDENT_WORK = re.compile(
    r"\b(?:du học sinh[^.!?]{0,55}(?:làm thêm|baito)|(?:làm thêm|baito)[^.!?]{0,55}du học sinh|"
    r"(?:du học sinh|trường tiếng)[^.!?]{0,35}vừa học vừa làm)\b"
)
POST_STUDY = re.compile(
    r"\b(?:sau khi (?:tốt nghiệp|ra trường|học xong|học trường tiếng)|"
    r"(?:tốt nghiệp|ra trường)[^.!?]{0,35}(?:rồi|xong)[^.!?]{0,35}(?:xin việc|đi làm|làm việc|chuyển)|"
    r"học xong[^.!?]{0,45}(?:xin việc|đi làm|làm việc|chuyển))\b"
)
LANGUAGE_SCHOOL_TO_WORK = re.compile(
    r"\b(?:du học tiếng|học (?:trường tiếng|senmon))[^.!?]{0,45}"
    r"(?:xong|rồi|sau đó)[^.!?]{0,30}(?:chuyển |xin việc|đi làm|làm việc)"
)
ALTERNATE_WORK_ROUTE = re.compile(
    r"\b(?:hay đi làm|hoặc đi làm|đi làm luôn|sang làm luôn|trực tiếp đi làm|bỏ học để đi làm|"
    r"(?:đi|sang) nhật làm việc[^.!?]{0,24}(?:chỉ|không cần) học|"
    r"(?:du học|trường tiếng)[^.!?]{0,22}(?:hay|hay là|hoặc)[^.!?]{0,28}"
    r"(?:thực tập sinh|kỹ sư|kĩ sư|tokutei?|xuất khẩu lao động))\b"
)
LABOR_ACTIVITY = re.compile(
    r"\b(?:xuất khẩu lao động|xklđ|xkld|im japan|thực tập sinh kỹ năng|tu nghiệp sinh|"
    r"chuyển (?:việc|(?:sang )?công ty)|(?:đổi|nghỉ) (?:việc|công ty)|"
    r"visa (?:diện )?(?:lao động|đi làm|kỹ sư|kĩ sư|tokutei?)|working visa|"
    r"(?:muốn|định|tính) (?:đi|sang) (?:nhật )?(?:làm )?(?:kỹ sư|tokutei?)|"
    r"(?:chuyển|đi) (?:sang )?(?:shuy|shyu|shūshoku)|"
    r"(?:đi|sang|theo) (?:nhật )?diện (?:kỹ sư|kĩ sư|tokutei?)|hợp đồng lao động)\b"
)
LABOR_ROLE = re.compile(r"\b(?:kỹ sư|kĩ sư|tokutei?|kỹ năng đặc định|thực tập sinh|nghiệp đoàn|lao động)\b")
EMPLOYMENT = re.compile(
    r"\b(?:tuyển dụng|tuyển người|phỏng vấn|đơn hàng|môi giới|mức lương|lương|thu nhập|"
    r"đi làm|làm việc|xin việc|xin visa|chuyển|đi nhật|sang nhật|đi theo diện)\b|"
    r"\btháng (?:làm|nhận)\s+\d[^.!?]{0,16}(?:tr\b|triệu\b|man\b|yên\b)"
)
SALARY_CONTEXT = re.compile(r"\b(?:lương|thu nhập|tiền nhận|về tay|nhận về|tiết kiệm)\b")
EMPLOYER_CONTEXT = re.compile(r"\b(?:tăng ca|bảng lương|lương cơ bản|ca đêm|offer|giờ làm)\b")
OVERSEAS_WORK = re.compile(r"\bđi xuất khẩu\b[^.!?]{0,65}\b(?:làm|kiếm tiền|chuyển tiền)\b")
EMPLOYER_VISA = re.compile(
    r"\b(?:công ty[^.!?]{0,55}(?:xin visa|làm coe|tuyển người)|"
    r"(?:xin visa|làm coe)[^.!?]{0,45}công ty)\b"
)
NON_STUDENT_RESIDENCE = re.compile(
    r"\b(?:visa (?:thăm thân|gia đình|kết hôn|multi)|bảo hiểm phụ thuộc|"
    r"đón (?:vợ|chồng|gia đình)[^.!?]{0,25}đoàn tụ)\b"
)
GENERAL_RESIDENCE = re.compile(r"\b(?:gia hạn (?:visa|lưu trú|thẻ (?:gai|ngoại kiều))|thẻ ngoại kiều)\b")
MEDIA_PRODUCTION = re.compile(r"\b(?:góc quay|dựng video|edit|góc máy)\b")
MEDIA_CHANNEL = re.compile(r"\b(?:youtube|kênh|lượt xem|view|ra đều video)\b")
GREETING = re.compile(r"\bchào\b")
OTHER_STUDY_DESTINATION = re.compile(
    r"\b(?:du học|đi học|học (?:ở|tại|bên)) (?:hàn quốc|hàn|đài loan|trung quốc|úc|mỹ|anh quốc|canada)\b"
)

STUDY_SOURCE = re.compile(
    r"\b(?:du học(?: sinh)?|dhs|eju|mext|jasso|apu|senmon|trường nhật ngữ|"
    r"(?:vào|đỗ vào|học) đại học[^.!?]{0,18}(?:ở|tại|bên) nhật)\b"
)
ACADEMIC_DETAIL = re.compile(
    r"\b(?:chứng minh tài chính|xác minh tài chính|học phí|học bổng|nhập học|tuyển sinh|"
    r"chọn trường|chọn ngành|trường nào|ngành nào|học ngành|bậc học|học lên|học tiếp|"
    r"nộp (?:đơn|hồ sơ)|xin coe|research proposal|đề cương nghiên cứu|đề cương học tập|"
    r"đại sứ quán|điểm (?:chuẩn|đầu vào)|định sang học|chương trình học)\b"
)
CENTER_DETAIL = re.compile(r"\b(?:trung tâm nào|trung tâm nào đó)\b")
CENTER_STUDY_CUE = re.compile(r"\b(?:học ở|học tại|đăng ký học|đăng kí học|học tiếng|chi phí)\b")
STUDY_COST_SOURCE = re.compile(r"\b(?:chi phí|hết bao nhiêu tiền|cần bao nhiêu để đi học)\b")
STUDY_COST_DETAIL = re.compile(r"\b(?:chi phí|tiền nhà|sinh hoạt phí|tiền học|vay tiền)\b")
DEPENDENT_PATH = re.compile(r"\b(?:đi )?theo (?:cha mẹ|bố mẹ|ba mẹ|gia đình|chồng|vợ)\b")
PROMOTIONAL_POINTER = re.compile(r"\b(?:tìm hiểu thêm|xem thêm|tham khảo thêm)[^.!?]{0,100}\btại\s*:\s*(?:\[url\])?$")
STUDENT_VISA_SOURCE = re.compile(
    r"\b(?:visa|coe|thị thực|tư cách lưu trú)[^.!?]{0,22}(?:du học|du học sinh|sinh viên)\b"
)
VISA_DETAIL = re.compile(r"\b(?:giấy tờ|hồ sơ|hộ chiếu|thẻ visa|thẻ ngoại kiều|gia hạn|đơn xin|nộp|nhận visa|coe)\b")
FAMILY_VISA_DETAIL = re.compile(r"\b(?:vợ|chồng|gia đình|thăm thân|phụ thuộc)\b")
WORK_SOURCE_DETAIL = re.compile(
    r"\b(?:lương|thu nhập|tăng ca|hợp đồng|đơn hàng|tuyển dụng|tuyển người|phỏng vấn|"
    r"môi giới|chuyển công ty|chuyển việc|xin việc|làm việc|giấy thuế)\b"
)


@dataclass(frozen=True)
class RelevanceDecision:
    status: str
    reason: str

    @property
    def manual_review_status(self) -> str:
        return "pending_manual_review" if self.status == "uncertain" else "rule_based_not_human_reviewed"


def _scope_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"\bdu hoc\b", "du học", text)
    text = re.sub(r"\bnhât\b", "nhật", text)
    return normalize_term_forms(text).replace("_", " ")


def assess_relevance(text: str, source_name: str = "") -> RelevanceDecision:
    """Use the source only when the comment contains compatible evidence."""
    text = _scope_text(text)
    study_ref = STUDY_REFERENCE.search(text)
    action = ACADEMIC_ACTION.search(text)
    academic = APPLICATION.search(text)
    japan = JAPAN.search(text)
    student_work = STUDENT_WORK.search(text)
    specific = SPECIFIC_STUDY.search(text)
    exam_fund = JP_EXAM_FUND.search(text)

    # General educational vocabulary may describe a labor course. Only a direct
    # study pathway overrides strong cues about a different activity.
    strong_study = bool(specific or student_work or (exam_fund and EXAM_FUND_CONTEXT.search(text)))
    contextual_study = bool(
        (study_ref and (action or japan)) or (academic and japan)
        or (STUDENT_COST.search(text) and LANGUAGE_OR_EXAM.search(text))
    )

    labor = LABOR_ACTIVITY.search(text) or EMPLOYER_VISA.search(text) or (japan and OVERSEAS_WORK.search(text))
    role, employment = LABOR_ROLE.search(text), EMPLOYMENT.search(text)
    salary, employer = SALARY_CONTEXT.search(text), EMPLOYER_CONTEXT.search(text)
    alternate = ALTERNATE_WORK_ROUTE.search(text)
    residence = NON_STUDENT_RESIDENCE.search(text)
    labor_route = labor or (role and employment) or alternate
    main_scope_conflict = labor_route or residence

    if main_scope_conflict:
        ordered_path = LANGUAGE_SCHOOL_TO_WORK.search(text) or (
            (strong_study or contextual_study) and POST_STUDY.search(text)
        )
        if ordered_path and not alternate and not residence:
            return RelevanceDecision("relevant", "study_path_before_employment")
        if strong_study or study_ref:
            return RelevanceDecision("uncertain", "study_and_labor_or_residence")
        if labor_route:
            evidence = labor or role or alternate
            return RelevanceDecision("off_topic", f"labor_without_study: {evidence.group()}")
        return RelevanceDecision("off_topic", f"non_student_residence: {residence.group()}")

    other_destination = OTHER_STUDY_DESTINATION.search(text)
    if other_destination:
        if japan or strong_study:
            return RelevanceDecision("uncertain", "study_destinations_need_context")
        return RelevanceDecision("off_topic", f"other_study_destination: {other_destination.group()}")
    if not strong_study and len(GREETING.findall(text)) >= 4:
        return RelevanceDecision("uncertain", "repeated_greetings_without_explicit_study_path")
    if student_work:
        return RelevanceDecision("relevant", "student_part_time_with_study_context")
    if strong_study or contextual_study:
        return RelevanceDecision("relevant", "study_path_or_japan_academic_preparation")
    if salary and employer:
        return RelevanceDecision("off_topic", f"pay_and_employment_only: {salary.group()} + {employer.group()}")
    if MEDIA_PRODUCTION.search(text) and MEDIA_CHANNEL.search(text):
        return RelevanceDecision("off_topic", "video_production_without_study")
    if source_name:
        source = _scope_text(re.sub(r"#\w+", " ", source_name))
        source_has_study = STUDY_SOURCE.search(source)
        source_study = source_has_study and (JAPAN.search(source) or JP_EXAM_FUND.search(source))
        source_labor = bool(LABOR_ACTIVITY.search(source) or LABOR_ROLE.search(source))
        source_residence = bool(NON_STUDENT_RESIDENCE.search(source))
        if source_study and not (
            source_labor or source_residence or OTHER_STUDY_DESTINATION.search(source)
            or DEPENDENT_PATH.search(text) or PROMOTIONAL_POINTER.search(text)
        ):
            specific_study_detail = ACADEMIC_DETAIL.search(text) or (
                CENTER_DETAIL.search(text) and CENTER_STUDY_CUE.search(text)
            ) or (STUDY_COST_SOURCE.search(source) and STUDY_COST_DETAIL.search(text))
            if specific_study_detail or (STUDENT_VISA_SOURCE.search(source) and VISA_DETAIL.search(text)):
                return RelevanceDecision("relevant", "study_detail_supported_by_source")
        if not (study_ref or academic or ACADEMIC_DETAIL.search(text)):
            if source_residence and FAMILY_VISA_DETAIL.search(text) and VISA_DETAIL.search(text):
                return RelevanceDecision("off_topic", "family_residence_confirmed_by_source")
            if source_labor and not source_has_study and (
                WORK_SOURCE_DETAIL.search(text) or (LABOR_ROLE.search(source) and VISA_DETAIL.search(text))
            ):
                return RelevanceDecision("off_topic", "labor_detail_supported_by_source")
    if GENERAL_RESIDENCE.search(text):
        return RelevanceDecision("uncertain", "residence_type_unspecified")
    return RelevanceDecision("uncertain", "insufficient_study_context")


def select_analysis_documents(documents: pd.DataFrame) -> pd.DataFrame:
    """Share one in-scope corpus between TF-IDF, LDA and the figures."""
    if "relevance_status" in documents.columns:
        statuses = documents["relevance_status"]
        if not statuses.isin(RELEVANCE_STATUSES).all():
            raise ValueError("Missing or invalid relevance_status in the classified data.")
    else:
        # Previously exported CSVs do not have the scope columns.
        field = "text_cleaned" if "text_cleaned" in documents else "text_segmented"
        sources = documents["source_name"].fillna("") if "source_name" in documents else [""] * len(documents)
        statuses = pd.Series(
            [assess_relevance(str(text), str(source)).status for text, source in zip(documents[field].fillna(""), sources)],
            index=documents.index,
        )
    return documents.loc[statuses.eq("relevant")].reset_index(drop=True)
