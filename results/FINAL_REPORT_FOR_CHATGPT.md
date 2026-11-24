# Tổng kết kiểm chứng dự án phân tích bình luận về du học Nhật Bản

**Kết quả ngày 23/11/2026.** Pipeline chính giữ **K = 3** để trình bày các chủ đề thăm dò. K = 2 dẫn đầu ba chỉ số với seed 42, nhưng ưu thế coherence và độ đa dạng thay đổi theo seed. Các mẫu đã đọc cho thấy K = 3 tách hồ sơ và học bổng khỏi nhóm về trường và trải nghiệm học rõ hơn. Chủ đề thứ ba vẫn pha trộn nội dung lao động. Bằng chứng hiện tại chưa đủ để gọi một giá trị K là tối ưu.

## A. Số liệu đã xác minh

| Chỉ tiêu | Số văn bản |
|---|---:|
| Đầu vào Excel, sheet `DATA` | 1.511 |
| Dòng bị loại | 58 |
| Dòng sạch | 1.453 |
| Motivation | 107 |
| Purpose | 93 |
| Preparation | 364 |
| Other | 889 |
| Tập phân tích TF-IDF | 564 |
| Tập phân tích LDA | 564 |
| CSV xuất cho Orange | 1.453 |

58 dòng bị loại gồm 26 dòng trước hoặc trong tháng 01/2022, 1 dòng trùng văn bản, 28 dòng thuộc [danh sách loại trừ đã được lập từ trước](../data/reviewed_exclusions.csv), 1 bình luận theo mẫu và 2 dòng quảng cáo. Số dòng thiếu văn bản hoặc có ngày không hợp lệ đều bằng 0. Tỷ lệ loại là 3,84%. Cả 1.453 dòng sạch có ID duy nhất. Điểm kiểm tra ngày hiện áp dụng từ 00:00 ngày 01/02/2022.

Có 1.063 dòng cần duyệt nhãn hoặc ngữ cảnh. Trong số đó, 214 dòng được luật gắn cờ phạm vi, gồm 137 trường hợp lao động và 30 trường hợp cư trú thiếu ngữ cảnh học, 24 trường hợp có cả tín hiệu học tập và lao động/cư trú, cùng 23 trường hợp có tín hiệu lao động yếu. Accuracy chưa được đo do thiếu tập nhãn chuẩn. [Metrics](metrics.json) · [Dữ liệu phân loại](../data/processed/classified_data.csv) · [Hàng chờ duyệt](../data/processed/manual_review_needed.csv) · [Cờ phạm vi tự động](suspected_off_topic.csv)

## B. Kiểm chứng số chủ đề LDA

### Phương pháp và kết quả định lượng

Cả năm mô hình dùng cùng 564 văn bản thuộc Motivation, Purpose và Preparation, cùng ma trận tần suất từ với `min_df=2` và từ vựng 819 mục. LDA của scikit-learn dùng batch, `max_iter=30`, seed 42 và hai tham số prior cố định bằng 1/3. Coherence Cᵥ là trung bình theo chủ đề, tính bằng Gensim trên 20 từ đứng đầu mỗi chủ đề, với cửa sổ trượt 110 token và tập tham chiếu là 564 văn bản đã tiền xử lý. Độ đa dạng là số từ duy nhất trong các danh sách top 20 chia cho `K × 20`. Training perplexity là ước lượng trên chính tập huấn luyện, nên chỉ dùng bổ sung. [Gensim CoherenceModel](https://radimrehurek.com/gensim/models/coherencemodel.html) · [Röder và cộng sự, 2015](https://doi.org/10.1145/2684822.2685324) · [LDA của scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)

| K | Coherence Cᵥ ↑ | Đa dạng từ khóa ↑ | Training perplexity ↓ |
|---:|---:|---:|---:|
| 2 | 0,368033 | 0,825000 | 545,4303 |
| 3 | 0,337879 | 0,733333 | 567,4806 |
| 4 | 0,322391 | 0,787500 | 584,8892 |
| 5 | 0,311562 | 0,740000 | 607,3822 |
| 6 | 0,343726 | 0,700000 | 638,4214 |

Kiểm tra độ nhạy bổ sung giữ nguyên ma trận và tham số, thay riêng seed khi gọi `make_lda_model(k, random_state=seed)`. Hai seed bổ sung dưới đây áp dụng cho K = 2 và K = 3. K = 2 có training perplexity thấp hơn ở cả ba seed, còn thứ hạng của Cᵥ và độ đa dạng đảo ở seed 123.

| Seed | K | Coherence Cᵥ | Đa dạng từ khóa | Training perplexity |
|---:|---:|---:|---:|---:|
| 7 | 2 | 0,341507 | 0,950000 | 534,6982 |
| 7 | 3 | 0,316889 | 0,816667 | 562,0113 |
| 123 | 2 | 0,317452 | 0,775000 | 540,0986 |
| 123 | 3 | 0,343929 | 0,816667 | 562,3093 |

[Bảng K = 2–6](lda_evaluation.csv) · [Thông số, môi trường và 20 từ đầu cho mỗi chủ đề](lda_evaluation_summary.txt) · [Biểu đồ so sánh K](figures/lda_k_comparison.png)

### Đối chiếu nội dung K = 2 và K = 3

Chương trình xuất tối đa 15 văn bản có xác suất cao nhất trong mỗi chủ đề chiếm ưu thế. Mẫu ngẫu nhiên gồm 24 ID lấy đều không hoàn lại từ tập LDA, với seed 42, rồi dùng cùng các ID để đối chiếu các cấu hình. Codex đọc toàn bộ 24 văn bản ngẫu nhiên và 5 văn bản xác suất cao nhất của từng chủ đề ở K = 2 và K = 3. Sau khi gộp trùng theo ID, tổng cộng có 41 văn bản được đọc. Tệp ghi chú lưu 73 lượt quan sát theo cấu hình và loại mẫu, có lý do cho từng ID và trạng thái chờ người xác nhận. [Mẫu xác suất cao](lda_k_review_samples.csv) · [Mẫu ngẫu nhiên](lda_k_random_review_samples.csv) · [Ghi chú định tính theo ID](lda_k_content_review.csv)

| Cấu hình, chủ đề | Toàn tập | Mẫu ngẫu nhiên | Trong 5 mẫu xác suất cao |
|---|---:|---:|---|
| K = 2, Topic_0 | 295 | 14 | 1 trực tiếp, 1 liền kề, 1 giao thoa, 2 lệch phạm vi |
| K = 2, Topic_1 | 269 | 10 | 3 trực tiếp, 1 liền kề, 1 giao thoa |
| K = 3, Topic_0 | 163 | 7 | 3 trực tiếp, 1 liền kề, 1 giao thoa |
| K = 3, Topic_1 | 188 | 9 | 3 trực tiếp, 1 liền kề, 1 giao thoa |
| K = 3, Topic_2 | 213 | 8 | 2 trực tiếp, 3 lệch phạm vi |

“Trực tiếp” gồm các trường hợp bàn rõ về du học hoặc điều kiện phục vụ lộ trình học. “Liền kề” gồm học tiếng, lời khuyên học tập hoặc phản ứng chung thiếu ví dụ cụ thể về du học. “Giao thoa” ghi nhận nội dung học tập lẫn lao động hoặc cư trú. “Chưa rõ” dùng khi bình luận thiếu ngữ cảnh để xác định mục đích. Đây là đọc định tính của Codex, chưa qua xác nhận của nhóm nghiên cứu. Các số lượng trong bảng là mô tả mẫu, chưa đủ để ước tính tỷ lệ lệch phạm vi trên toàn tập.

- **K = 2.** Topic_0 có `học_bổng`, `hồ_sơ`, `visa` nhưng gom cả `tiếng_nhật`, `làm_thêm` và `kỹ_sư`. `P2_0463` (mẫu ngẫu nhiên) bàn học bổng báo Asahi, `P1_0005` (xác suất 0,9933) bàn làm thêm để trang trải chi phí du học, còn `P2_0972` (0,9915) bàn môi giới diện kỹ sư và `P2_1029` (0,9909) bàn phỏng vấn diện thực tập sinh/kỹ sư. Topic_1 có `trường`, `đại_học`, `ngành` và được minh họa bằng `P4_0100` (0,9982) và `P4_0105` (0,9960) về APU.
- **K = 3.** Topic_0 có ba ví dụ xác suất cao về hồ sơ hay học bổng (`P2_0002`, `P1_0084`, `P2_0009`). Topic_1 giữ được các ví dụ về APU và lộ trình học. Topic_2 tập trung một số trường hợp về ngành kỹ sư, môi giới, công ty hoặc bảng lương, đồng thời vẫn chứa các ví dụ học tập, như `P1_0034` hỏi về đi du học từ bậc cao đẳng và `P1_0005` bàn chi phí du học tự túc. Trong mẫu ngẫu nhiên, `P2_0454` chuyển từ Topic_0 ở K = 2 sang Topic_2 ở K = 3, phản ánh sự giao thoa giữa học bổng và thu nhập làm thêm.
- **Quyết định.** Giữ **K = 3** cho đầu ra chính vì cách phân nhóm này cho phép mô tả riêng hồ sơ/học bổng và trải nghiệm chọn trường bằng ví dụ trực tiếp, đồng thời nhận diện rõ hơn vùng pha trộn về việc làm. Lợi thế định lượng của K = 2 ở seed 42 vẫn được báo cáo. Việc giữ K = 3 là lựa chọn thăm dò có điều kiện, chưa phải kết luận tối ưu.

### Chủ đề K = 3 trong đầu ra chính

| Chủ đề | Số văn bản | Tỷ lệ | Tên mô tả tạm thời | Top 10 từ và trọng số |
|---|---:|---:|---|---|
| Topic_0 | 163 | 28,9% | Học bổng, hồ sơ và visa | học_bổng 0,039906, visa 0,022249, chuẩn_bị 0,021709, du_học 0,020614, hồ_sơ 0,018734, trường 0,016535, thông_tin 0,013067, học_phí 0,011703, báo 0,011325, tốt_nghiệp 0,009486 |
| Topic_1 | 188 | 33,3% | Trường, ngành và trải nghiệm học | học 0,051870, trường 0,032916, nhật 0,029214, chi_phí 0,015034, tiếng 0,013819, việt_nam 0,012754, cao 0,011960, apu 0,010864, ngành 0,010580, tiếng_nhật 0,010524 |
| Topic_2 | 213 | 37,8% | Tiếng Nhật, bằng cấp, tài chính và việc làm | học 0,042959, nhật 0,023509, tiếng_nhật 0,021990, tiền 0,019971, bằng 0,017880, kỹ_sư 0,017563, động_lực 0,016584, chưa 0,013500, công_ty 0,013154, đại_học 0,012550 |

| Chủ đề | ID | Xác suất | Nội dung tóm lược và ghi chú |
|---|---|---:|---|
| Topic_0 | `P2_0002` | 0,9714 | Hỏi COE cho trường tiếng, chứng chỉ J-Test và thời hạn nộp hồ sơ. |
| Topic_0 | `P4_0011` | 0,9714 | Bàn visa kỹ sư, thăm thân và gia đình. Du học sinh xuất hiện như một ví dụ về tình trạng cư trú. |
| Topic_1 | `P4_0105` | 0,9921 | Đánh giá APU, cân nhắc môi trường học và cơ hội việc làm sau học. |
| Topic_1 | `P2_1196` | 0,9723 | Trọng tâm là việc nhà hàng và Tokutei, có nhắc du học sinh ở phần liên hệ. |
| Topic_2 | `P1_0034` | 0,9704 | Hỏi về khả năng đi du học và học nghề mới sau cao đẳng. |
| Topic_2 | `P2_0972` | 0,9853 | Bàn môi giới và phí đi diện kỹ sư. |
| Topic_2 | `P4_0023` | 0,9736 | Bàn giấy tờ và visa khi công ty tuyển người nước ngoài. |

Cần duyệt thêm các ID `P2_0361` (visa diện kỹ sư xây dựng), `P2_0781` (IM Japan là chương trình thực tập kỹ thuật dành cho lao động), `P2_0923` (lương kỹ sư CNTT) và `P2_0821` (tăng ca và thu nhập). Bốn ví dụ cũ về chuyển việc hoặc gia hạn visa `P4_0014`, `P4_0018`, `P4_0019`, `P4_0025` vẫn trong dữ liệu sạch, hiện thuộc Other. Phần chú giải giữ cả các trường hợp giao thoa như làm thêm trong lộ trình du học. [Từ khóa có trọng số](lda_topics.csv) · [Xác suất theo văn bản](lda_doc_topics.csv) · [Mẫu K = 3](lda_topic_review.csv) · [Nguồn xác nhận phạm vi IM Japan](https://colab.moha.gov.vn/tin-tuc/1310/Dieu-kien-tuyen-chon-va-cac-thong-tin-can-thiet-khi-tham-gia-chuong-trinh-thuc-tap-sinh-ky-thuat-tai-Nhat-Ban-IM-JAPAN.aspx)

### Ma trận nhãn phân loại và chủ đề để đối chiếu Hình 8

| Nhãn | n | Topic_0 | Topic_1 | Topic_2 |
|---|---:|---:|---:|---:|
| Motivation | 107 | 11 (10,3%) | 47 (43,9%) | 49 (45,8%) |
| Purpose | 93 | 8 (8,6%) | 37 (39,8%) | 48 (51,6%) |
| Preparation | 364 | 144 (39,6%) | 104 (28,6%) | 116 (31,9%) |

Tỷ lệ tính theo hàng và có thể lệch 0,1 điểm phần trăm do làm tròn. Topic là nhóm thống kê, có thể cắt ngang các nhãn phân loại.

## C. Kết quả TF-IDF

`TfidfVectorizer` được khớp riêng trên toàn tập và từng nhóm. Mean TF-IDF lấy trung bình theo cột của ma trận trong mỗi tập, gồm cả điểm 0 cho văn bản không có từ đó. Kết quả có 60 dòng, gồm 15 từ cho toàn tập và 15 từ cho mỗi nhóm. Do IDF tính riêng theo tập, nên so sánh chủ yếu theo thứ hạng và nội dung từ khóa trong từng nhóm. [CSV TF-IDF](tfidf_results.csv)

| Tập | n | Từ khóa và trọng số thực tế | Nhận xét |
|---|---:|---|---|
| Toàn bộ ba nhóm | 564 | học 0,054353, nhật 0,033084, tiếng_nhật 0,028338, học_bổng 0,026375, trường 0,026052, du_học 0,024593, chi_phí 0,024289 | Hai từ đầu bao quát nhiều nội dung. Các cụm học bổng, tiếng Nhật và chi phí cụ thể hơn. |
| Motivation | 107 | động_lực 0,070060, học 0,046512, nhật 0,041589, nhật_bản 0,033167, trải_nghiệm 0,026658, cố_gắng 0,021874 | Từ khóa có sắc thái động lực và trải nghiệm. |
| Purpose | 93 | học 0,083996, bằng 0,065827, đại_học 0,063090, senmon 0,060223, nhật 0,059458, kỹ_sư 0,055412 | Nhấn vào bậc học, bằng cấp và định hướng nghề. |
| Preparation | 364 | học 0,056748, tiếng_nhật 0,036964, học_bổng 0,036440, chi_phí 0,033830, chuẩn_bị 0,033473, hồ_sơ 0,024930 | Nhấn vào ngoại ngữ, chi phí và chuẩn bị hồ sơ. |

## D. Danh sách biểu đồ và nguồn dữ liệu

Cả tám hình chính được tạo lại. CSV đã có và chín ảnh PNG, kể cả biểu đồ K, giống từng byte với lần chạy trước vì dữ liệu phân tích và cấu hình K chính cho cùng kết quả. Kết quả mới bổ sung mẫu ngẫu nhiên, ghi chú phạm vi và dấu kiểm SHA-256.

| Hình | Tệp từ thư mục dự án | Dữ liệu sử dụng |
|---:|---|---|
| 1 | [results/figures/hinh_01_phan_bo_nhom_phan_loai.png](figures/hinh_01_phan_bo_nhom_phan_loai.png) | `data/processed/classified_data.csv`. Phân bố ba nhóm phân tích với 564 văn bản. |
| 2 | [results/figures/hinh_02_tfidf_tu_khoa_tong_the.png](figures/hinh_02_tfidf_tu_khoa_tong_the.png) | `results/tfidf_results.csv`, nhóm `ALL`, n = 564. |
| 3 | [results/figures/hinh_03_tfidf_nhom_dong_luc_motivation.png](figures/hinh_03_tfidf_nhom_dong_luc_motivation.png) | `results/tfidf_results.csv`, `Motivation`, n = 107. |
| 4 | [results/figures/hinh_04_tfidf_nhom_muc_dich_purpose.png](figures/hinh_04_tfidf_nhom_muc_dich_purpose.png) | `results/tfidf_results.csv`, `Purpose`, n = 93. |
| 5 | [results/figures/hinh_05_tfidf_nhom_chuan_bi_preparation.png](figures/hinh_05_tfidf_nhom_chuan_bi_preparation.png) | `results/tfidf_results.csv`, `Preparation`, n = 364. |
| 6 | [results/figures/hinh_06_lda_tu_khoa_tung_chu_de.png](figures/hinh_06_lda_tu_khoa_tung_chu_de.png) | `results/lda_topics.csv`. Top 10 từ theo trọng số trong từng chủ đề của K = 3. |
| 7 | [results/figures/hinh_07_lda_phan_bo_chu_de.png](figures/hinh_07_lda_phan_bo_chu_de.png) | `results/lda_doc_topics.csv`. Đếm chủ đề chiếm ưu thế trên 564 văn bản. |
| 8 | [results/figures/hinh_08_lda_chu_de_vs_nhom_phan_loai.png](figures/hinh_08_lda_chu_de_vs_nhom_phan_loai.png) | `results/lda_doc_topics.csv`. Tỷ lệ chủ đề tính riêng trong mỗi nhãn quy tắc. |
| So sánh K | [results/figures/lda_k_comparison.png](figures/lda_k_comparison.png) | Cùng bảng số liệu xuất trong `results/lda_evaluation.csv`. |

CSV Orange giữ nguyên cấu trúc, gồm `text_model` để tách phần nội dung dùng cho Word Cloud. Phạm vi đợt này dừng ở xuất và kiểm chứng CSV. Hình Word Cloud trong bản Word tham chiếu thuộc lần chạy trước.

## E. Thay đổi mã, kiểm thử và dọn thư mục

- [`src/cleaning.py`](../src/cleaning.py) lọc ngày trước khi loại trùng. Nhờ đó, dòng trùng ngoài mốc thời gian không làm mất dòng hợp lệ. Mốc được biểu diễn bằng ngày đầu tháng 02/2022 để xử lý đúng cả thời điểm lẻ giây cuối tháng 01. Ca tái hiện trước khi sửa cho 0 dòng, sau khi sửa giữ đúng dòng hợp lệ. Workbook thật vẫn cho 1.453 dòng sạch.
- [`src/lda.py`](../src/lda.py) và [`main.py`](../main.py) dùng chung cấu hình K mặc định. `metrics.json` ghi SHA-256 của Excel đầu vào, CSV phân loại và 16 đầu ra pipeline, gồm tám ảnh.
- [`src/lda_evaluation.py`](../src/lda_evaluation.py) bổ sung mẫu ngẫu nhiên dùng chung, cho phép chỉ định `--review-k` cho báo cáo mẫu khi chạy cấu hình khác mặc định, và ghi SHA-256 của năm đầu ra đánh giá. Bộ quy tắc Classification, cách tính TF-IDF, giao diện CSV Orange và ngưỡng lọc nội dung vẫn dùng cấu hình trước đó. Ghi chú định tính được lưu riêng để lần chạy tự động không ghi đè.
- Kiểm thử đã chạy trên Python 3.12.14. Đạt kiểm tra cú pháp/import, ca hồi quy Cleaning, `pip check`, `python main.py`, `python -m src.lda_evaluation`, đối chiếu ID, phân bố, trọng số, xác suất, UTF-8, chín ảnh PNG và SHA-256. `unittest discover` phát hiện 0 test repo; các kiểm tra ca biên và tích hợp được chạy trực tiếp. Khi chạy LDA với K khác mặc định trong khoảng 2–6, dùng cùng `--n-topics` và `--review-k` để tệp mẫu chính trùng cấu hình.
- Giữ `AI GROUP - Báo cáo cập nhật.docx` ở thư mục gốc. Ba tệp `AI GROUP - Bản đã chỉnh sửa.docx`, `AI GROUP - Giải thích chọn 3 topics.docx` và `AI GROUP - Đã sửa mục 6.docx` đã được chuyển vào `archive/old_reports/`. Nội dung bốn tệp Word, Excel gốc, danh sách loại trừ, README và notebook được đối chiếu SHA-256 trước và sau. Đã dọn `src/__pycache__/`. Notebook `notebooks/analysis.ipynb` đang rỗng và được giữ theo yêu cầu. README mô tả kết quả lịch sử 1.439 dòng sạch và 654 dòng phân tích. Dùng `metrics.json` và tổng kết này cho số liệu hiện hành.

## F. Đoạn văn đề xuất để cập nhật Word

### Mục 3. Cleaning

Quy trình đọc 1.511 bản ghi từ sheet DATA của workbook gốc, kiểm tra nội dung thiếu, lọc ngày và loại trùng theo văn bản chuẩn hóa trước khi áp dụng danh sách loại trừ và quy tắc lọc nhiễu. Kết quả giữ 1.453 bản ghi và loại 58 bản ghi (3,84%), gồm 26 bản ghi trước hoặc trong tháng 01/2022, một bản ghi trùng, 28 bản ghi thuộc danh sách loại trừ, một bình luận theo mẫu và hai bản ghi quảng cáo. Văn bản sạch được lưu ở dạng dễ đọc và dạng tách từ để phân tích.

### Mục 4. Classification

Bộ luật gán 107 văn bản vào Motivation, 93 vào Purpose, 364 vào Preparation và 889 vào Other. Trong 564 văn bản thuộc ba nhóm nghiên cứu, tỷ trọng tương ứng là 19,0%, 16,5% và 64,5%. Hệ thống ghi 1.063 trường hợp cần duyệt, trong đó 214 trường hợp mang tín hiệu về phạm vi. Các nhãn mô tả đầu ra của bộ quy tắc. Để đo độ chính xác, cần một tập nhãn do người đánh giá xác nhận.

### Mục 5. TF-IDF

TF-IDF sử dụng 564 văn bản thuộc ba nhóm nghiên cứu. Trọng số trung bình của từ khóa được tính trên toàn bộ văn bản của từng tập, kể cả văn bản không có từ khóa đó. Từ khóa “động lực” nổi bật trong Motivation, “bằng”, “đại học” và “senmon” trong Purpose, còn “tiếng Nhật”, “học bổng” và “chi phí” nổi bật trong Preparation. Hình 2–5 cho thấy các khác biệt về từ vựng trong tập đã phân loại.

### Mục 6. LDA

Các mô hình từ K = 2 đến K = 6 dùng chung 564 văn bản và 819 từ hoặc cụm từ. Với seed 42, K = 2 đứng đầu về coherence và độ đa dạng, đồng thời có training perplexity thấp nhất. Kiểm tra bổ sung ở seed 123 cho K = 3 xếp cao hơn về coherence và độ đa dạng. Trong mẫu xác suất cao, K = 3 tách được một cụm có ví dụ rõ về COE, hồ sơ và học bổng, cùng một cụm về trường và trải nghiệm học. Cụm thứ ba giao thoa giữa học tập, chi phí và lao động. Báo cáo trình bày K = 3 để mô tả các nhóm này và ghi nhận nhu cầu xác nhận định tính trước khi kết luận về số chủ đề.

### Mục 8. Tổng hợp kết quả

Cleaning và Classification cho 1.453 văn bản tổng cộng. TF-IDF và LDA dùng cùng 564 văn bản thuộc ba nhóm nghiên cứu. Ở K = 3, số văn bản theo chủ đề chiếm ưu thế lần lượt là 163, 188 và 213. Một số bình luận về môi giới, visa lao động và cư trú vẫn có xác suất chủ đề cao. Vì vậy, các bảng, hình và ví dụ chỉ mô tả tập văn bản đã đưa vào phân tích. Kết quả đọc định tính dựa trên một mẫu dùng chung và các mẫu xác suất cao, hiện chờ nhóm xác nhận.

### Mục 9. Kết luận

Quy trình xử lý bình luận tiếng Việt về du học Nhật Bản tạo ra bộ dữ liệu có thể đối chiếu theo ID, bảng từ khóa và mô hình chủ đề thăm dò. Nội dung chính quan sát được gồm hồ sơ và học bổng, lựa chọn trường và trải nghiệm học tập, cùng những trao đổi về ngoại ngữ, tài chính và việc làm. K = 3 được giữ cho kết quả trình bày, còn K = 2 là phương án có kết quả định lượng đáng lưu ý. Cần rà thêm ngữ cảnh nguồn và xác nhận các bình luận giao thoa, nhất là trước khi suy rộng kết luận về du học.
