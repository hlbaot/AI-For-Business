# AI for Business — Phân tích du học Nhật Bản của giới trẻ Việt Nam

Đề tài Term Project môn **AI for Business** (Đại học Duy Tân): *Phân tích động lực, mục đích và sự chuẩn bị của giới trẻ Việt Nam khi du học Nhật Bản*.

## 1. Cấu trúc thư mục

- `data/raw/raw_data.csv`: Dữ liệu gốc (UTF-8 CSV, tối thiểu gồm cột `text`, `created_at`; khuyến khích có `text_id`). Không bị sửa đổi hay ghi đè bởi pipeline.
- `data/raw/sample_test_data.csv`: Bộ dữ liệu giả lập (26 dòng) dùng riêng cho kiểm thử pipeline, tách biệt hoàn toàn với `raw_data.csv`.
- `data/processed/cleaned_data.csv`: Dữ liệu sau khi làm sạch, lọc ngày (sau 01/2022), ẩn danh PII và tách từ tiếng Việt (`underthesea`).
- `data/processed/classified_data.csv`: Dữ liệu đã gán nhãn thử nghiệm (`Motivation`, `Purpose`, `Preparation`, `Other`) bằng bộ phân loại rule-based có giải thích.
- `data/processed/manual_review_needed.csv`: Danh sách các văn bản mơ hồ / thiếu từ khóa cần kiểm tra thủ công.
- `data/exports/orange_data.csv`: File xuất chuẩn UTF-8-SIG dành cho **Orange Data Mining** (gửi cho thành viên nhóm 3).
- `results/tfidf_results.csv`: Top từ khóa TF-IDF tổng thể (`ALL`) và theo từng nhóm.
- `results/lda_topics.csv`: Chủ đề tiềm ẩn từ LDA (dùng ma trận tần suất từ `CountVectorizer`).
- `results/lda_doc_topics.csv`: Phân bố xác suất chủ đề theo từng văn bản.
- `results/metrics.json`: Thống kê tổng hợp của toàn bộ pipeline.

## 2. Cách chạy

Kích hoạt môi trường ảo:

```bash
source .venv/bin/activate
```

Chạy toàn bộ pipeline với dữ liệu chính thức (`data/raw/raw_data.csv`):

```bash
python main.py
```

Chạy kiểm thử với dữ liệu giả lập:

```bash
python main.py --input data/raw/sample_test_data.csv
```

Chỉ xuất lại file cho Orange (không chạy lại các bước trước):

```bash
python main.py --export-only
```

## 3. Hướng dẫn cho người dùng Orange Data Mining

1. Mở file `data/exports/orange_data.csv` bằng widget **CSV File Import** (hoặc **Corpus** trong Text Mining).
2. Chọn `text_segmented` (hoặc `text_cleaned`) làm cột văn bản (*Text*), và `category` làm biến phân loại (*Categorical*).
3. Dùng widget **Select Rows** để lọc theo `category` (`Motivation`, `Purpose`, `Preparation`, `Other`).
4. Nối với widget **Word Cloud** để xem trực quan hóa tổng thể và theo từng nhóm mà không cần viết code.
