import unittest

import pandas as pd

from src.classification import RuleBasedClassifier
from src.relevance import assess_relevance, select_analysis_documents


class RelevanceTests(unittest.TestCase):
    def test_study_and_scholarships(self):
        for text in (
            "Em muốn du học Nhật Bản và xin học bổng.",
            "Em đang học tiếng Nhật để du học trường tiếng.",
            "Em đang học để thi EJU vào đại học.",
        ):
            with self.subTest(text=text):
                self.assertEqual(assess_relevance(text).status, "relevant")

    def test_company_transfer_and_labor_routes(self):
        for text in (
            "Em muốn chuyển công ty tại Osaka và gia hạn visa lao động.",
            "Tuyển kỹ sư đi Nhật, công ty hẹn phỏng vấn và làm visa diện kỹ sư.",
            "Bảng lương tháng này có thêm tiền tăng ca.",
            "Em đăng ký chương trình IM Japan để thực tập kỹ thuật.",
            "Em đang học đại học ở Việt Nam và muốn đi kỹ sư CNTT tại Nhật.",
        ):
            with self.subTest(text=text):
                result = RuleBasedClassifier().predict_one(text)
                self.assertEqual(result.category, "Other")
                self.assertEqual(result.relevance.status, "off_topic")
                self.assertEqual(result.relevance.manual_review_status, "rule_based_not_human_reviewed")

    def test_student_work_and_career_path(self):
        for text in (
            "Em là du học sinh ở Nhật và đi làm thêm để trang trải học phí.",
            "Em đang học trường tiếng ở Nhật. Sau khi tốt nghiệp em muốn xin visa kỹ sư để làm ở công ty.",
            "Em dự định qua Nhật học tiếng, sau đó xin việc ngành CNTT.",
        ):
            with self.subTest(text=text):
                self.assertEqual(assess_relevance(text).status, "relevant")

    def test_conflicting_routes_require_review(self):
        for text in (
            "Em phân vân giữa du học trường tiếng và đi theo diện Tokutei làm nhà hàng.",
            "Em đang học trường tiếng. Trước khi tốt nghiệp em muốn chuyển shyu thì có được không?",
            "Du học sinh có bằng đại học chuyển diện việc làm rồi đón vợ con đoàn tụ.",
        ):
            with self.subTest(text=text):
                result = RuleBasedClassifier().predict_one(text)
                self.assertEqual(result.relevance.status, "uncertain")
                self.assertTrue(result.needs_manual_review)
                self.assertEqual(result.relevance.manual_review_status, "pending_manual_review")

    def test_generic_terms_need_context(self):
        for text in (
            "Cho em hỏi về visa.", "Công ty trả lương khi nào ạ?", "Xin hỏi visa công ty, lương thế nào?",
            "Em học ở Trường Nhật Ngữ tại Thành phố Hồ Chí Minh ạ.",
        ):
            with self.subTest(text=text):
                self.assertEqual(assess_relevance(text).status, "uncertain")
        self.assertEqual(assess_relevance("Visa du học Nhật nộp ở đâu ạ?").status, "relevant")

    def test_analysis_uses_scope_independent_from_category(self):
        data = pd.DataFrame({
            "text_id": ["a", "b", "c"],
            "text_cleaned": [
                "Học trường tiếng ở Nhật.", "Tuyển kỹ sư theo diện visa lao động.", "Xin hỏi về visa.",
            ],
            "category": ["Other", "Purpose", "Preparation"],
            "relevance_status": ["relevant", "off_topic", "uncertain"],
        })
        self.assertEqual(select_analysis_documents(data)["text_id"].tolist(), ["a"])
        legacy = data.drop(columns="relevance_status")
        self.assertEqual(select_analysis_documents(legacy)["text_id"].tolist(), ["a"])
