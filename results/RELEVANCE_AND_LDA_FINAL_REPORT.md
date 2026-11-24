# Kiểm chứng phạm vi dữ liệu và mô hình chủ đề du học Nhật Bản

Pipeline chọn **K = 4** cho tập **199 bình luận có bằng chứng về lộ trình du học**. K = 4 tách được thảo luận về hồ sơ và học bổng khỏi thảo luận về cách trang trải chi phí. Việc đọc mẫu vẫn mang tính thăm dò. Nhóm nghiên cứu chưa xác nhận nhãn phạm vi và tên chủ đề.

## 1. Sai lệch phát hiện trong tập cũ

Trong 564 văn bản TF-IDF/LDA trước đây, **31 văn bản** được bộ luật mới phân loại `off_topic`, **392** chuyển sang `uncertain` và **141** đạt `relevant`. Bộ phân loại theo từ khóa từng đưa nội dung tuyển kỹ sư, mức lương hoặc hồ sơ công ty vào các nhóm nghiên cứu vì chúng chứa từ như “visa”, “chuẩn bị” hoặc “kỹ sư”. Các dòng ngắn về visa, chi phí và việc làm thêm cũng có thể thiếu thông tin về mục đích sang Nhật.

| ID | Nội dung quan sát được | Kết quả mới |
|---|---|---|
| `P2_0972` | Phí và môi giới đi Nhật theo diện kỹ sư | `off_topic`, chuyển sang Other |
| `P2_0391` | Đang học đại học ở Việt Nam và hỏi về việc sang Nhật làm kỹ sư CNTT | `off_topic`, lộ trình được hỏi là việc làm |
| `P2_0781` | Hỏi chương trình thực tập kỹ thuật IM Japan | `off_topic`, lộ trình lao động |
| `P2_0821` | Hỏi về số giờ tăng ca và tiền nhận về | `off_topic`, thu nhập và tăng ca |
| `P4_0014`, `P4_0018` | Chuyển công ty hoặc hồ sơ gia hạn trong bối cảnh lao động | `off_topic`, còn trong dữ liệu sạch |
| `P2_0135` | Ghép “du học” và “Tokutei” trong một câu hỏi chi phí | `uncertain`, cần đọc ngữ cảnh |
| `P2_0651` | Đang học trường tiếng và hỏi chuyển sang đi làm trước tốt nghiệp | `uncertain`, chưa có lộ trình học rồi mới làm rõ ràng |
| `P2_0918` | Nêu tên trung tâm Nhật ngữ ở Việt Nam, mục đích sang Nhật chưa rõ | `uncertain` |
| `P1_0005` | Thu nhập làm thêm và sinh hoạt phí trong bài toán du học tự túc | `relevant` |
| `P2_1050` | Dự định chuyển diện việc làm sau khi học xong trường tiếng | `relevant` |

Đối chiếu 41 ID đã có ghi chú nội dung ở lần trước cho thấy cả 9 trường hợp từng được Codex đánh dấu lệch phạm vi nay nằm ngoài tập mô hình, và 4 trường hợp giao thoa đều vào `uncertain`. Trong 14 trường hợp từng ghi là liên quan trực tiếp, 13 đạt `relevant`; `P1_0084` vẫn cần xác nhận nguồn vì bình luận hỏi hồ sơ học bổng mà không nêu nước hoặc chương trình. Các trường hợp này từng được dùng khi rà quy tắc, nên chỉ minh họa thay đổi và không phải tập kiểm định độc lập. [Ghi chú lịch sử theo ID](lda_k_content_review.csv) · [Kết quả lọc từng văn bản](../data/processed/relevance_review.csv)

## 2. Quy tắc lọc và cách dùng dữ liệu

- `relevant` yêu cầu bằng chứng như lộ trình học tại Nhật, hồ sơ hoặc học bổng có bối cảnh, EJU đi kèm mục đích học, hay làm thêm gắn với việc học. Định hướng việc làm sau tốt nghiệp được giữ nếu có trình tự học rồi làm đủ rõ.
- `off_topic` yêu cầu tín hiệu cụ thể như chuyển công ty, visa lao động, chương trình phái cử, tuyển kỹ sư gắn với doanh nghiệp, hoặc lương đi kèm tăng ca mà thiếu bối cảnh học. Một từ riêng như “visa”, “công ty”, “kỹ sư” hoặc “lương” chỉ là dấu hiệu cần ngữ cảnh. Nhãn `off_topic` ở đây là kết luận của quy tắc và có trạng thái `rule_based_not_human_reviewed`.
- `uncertain` gồm nội dung thiếu mục đích hoặc có đồng thời các lộ trình học, lao động, cư trú chưa phân biệt được. Cả 1.032 trường hợp `uncertain` nằm trong hàng chờ duyệt và hiện đứng ngoài phân tích tự động. Các dòng này vẫn có trong CSV sạch, CSV phân loại và CSV Orange.

Các trọng số Motivation/Purpose/Preparation được giữ. Classification dùng `Other` cho ca lệch phạm vi rõ, giữ nhãn gợi ý đối với ca mơ hồ và gắn cờ duyệt. TF-IDF tổng thể, LDA và đánh giá K lấy cùng tập `relevant`, gồm cả **58** bình luận liên quan vẫn được gán `Other`. TF-IDF theo từng nhóm vẫn dùng ba nhãn nghiên cứu. Chủ đề được học từ ma trận từ, độc lập với nhãn phân loại. [Module phạm vi](../src/relevance.py) · [Module phân loại](../src/classification.py) · [Hàng chờ duyệt](../data/processed/manual_review_needed.csv)

## 3. Số liệu trước và sau lọc

| Chỉ tiêu | Số dòng |
|---|---:|
| Excel gốc | 1.511 |
| Làm sạch loại theo quy tắc hiện có | 58 |
| CSV sạch và CSV phân loại | 1.453 |
| `relevant` | **199** |
| `off_topic` theo quy tắc | 222 |
| `uncertain` chờ xác nhận | 1.032 |
| Hàng chờ duyệt nhãn hoặc phạm vi | 1.345 |
| TF-IDF tổng thể và LDA mới | **199** |
| CSV xuất cho Orange | 1.453 |

Tập mới có 141 ID từ tập cũ và 58 ID trước đây nằm ngoài tập mô hình vì mang nhãn `Other`. Trong 199 dòng `relevant`, không có dòng bị mất do thiếu token hay thiếu từ trong từ vựng LDA. Số dòng Cleaning và nội dung `cleaned_data.csv` khớp bản sao trước khi sửa.

| Nhãn phân loại | Trước đợt này | Toàn bộ CSV mới | Trong tập `relevant` |
|---|---:|---:|---:|
| Motivation | 107 | 101 | 16 |
| Purpose | 93 | 85 | 25 |
| Preparation | 364 | 372 | 100 |
| Other | 889 | 895 | 58 |
| **Tổng** | **1.453** | **1.453** | **199** |

TF-IDF tổng thể dẫn đầu bởi `học`, `du_học`, `trường` và `nhật`. Những cụm như `học_bổng`, `senmon` và `eju` mô tả nội dung cụ thể hơn. Các nhóm 16, 25 và 100 văn bản có cỡ mẫu khác nhau và IDF được tính riêng theo tập. [Số liệu pipeline](metrics.json) · [Trọng số TF-IDF](tfidf_results.csv)

## 4. Thực nghiệm K và quyết định cấu hình

Mọi cấu hình dùng cùng **199 văn bản**, `CountVectorizer(min_df=2)` với **390** từ hoặc cụm từ, LDA batch tối đa 30 vòng, seed 42 và hai tham số prior cùng bằng 1/3. Cᵥ dùng Gensim, 20 từ mỗi chủ đề và cửa sổ 110 token trên tập phân tích. Đa dạng từ khóa là tỷ lệ từ duy nhất trong danh sách top 20. Perplexity tính trên tập huấn luyện, phục vụ đối chiếu bổ sung. Bảng chỉ hỗ trợ so sánh K trên tập mới vì tập cũ dùng dữ liệu khác. [Bảng chỉ số](lda_evaluation.csv) · [Chi tiết phương pháp và top 20 từ](lda_evaluation_summary.txt)

| K | Coherence Cᵥ ↑ | Đa dạng ↑ | Training perplexity ↓ |
|---:|---:|---:|---:|
| 2 | 0,285391 | 0,800000 | 263,5744 |
| 3 | 0,328328 | 0,783333 | 262,8277 |
| **4** | **0,371106** | **0,812500** | **270,9642** |
| 5 | 0,378514 | 0,740000 | 287,1326 |
| 6 | 0,378831 | 0,725000 | 299,2283 |

K = 4 có độ đa dạng cao nhất và cải thiện Cᵥ khoảng 0,043 so với K = 3, đổi lại perplexity huấn luyện cao hơn. Kiểm tra thêm cho thấy Cᵥ của K = 4 cao hơn K = 3 ở seed 7 (0,300734 so với 0,284881) và seed 123 (0,377475 so với 0,359220). Ưu thế đa dạng của K = 4 chỉ quan sát được ở seed 42 trong ba seed đã thử. K = 5 và K = 6 tăng Cᵥ tối đa khoảng 0,008 so với K = 4 nhưng giảm đa dạng. Các mẫu của K = 5–6 cho thấy học bổng và tài chính xuất hiện lặp lại trong nhiều cụm. K = 2 có perplexity thấp nhưng gom nhiều vấn đề học tập ở mức rộng hơn.

Bằng chứng nội dung chính là K = 3 gom `hồ_sơ`, `học_bổng`, `học_phí` và `tiền` trong cùng một cụm. Ở K = 4, nhóm hồ sơ nổi bật với `hồ_sơ`, `nộp`, `mext`, còn nhóm tài chính nổi bật với `tiền`, `học_phí`, `đóng`, `đủ`. `P1_0085` và `P1_0083` minh họa hồ sơ MEXT. `P4_0004` và `P1_0005` minh họa trang trải học phí bằng làm thêm và hỗ trợ tài chính. Một phần học bổng vẫn giao thoa với chủ đề chi phí. Kết hợp kết quả định lượng và bằng chứng này, **K = 4** được chọn làm mặc định của pipeline. Bốn cụm vẫn là kết quả thăm dò.

### Mẫu đại diện và mẫu ngẫu nhiên

Đối chiếu 24 ID lấy ngẫu nhiên không hoàn lại với seed 42, cùng danh sách ID cho mọi K. Đọc thêm ba mẫu xác suất cao nhất của mỗi chủ đề ở K = 3 và K = 4, cùng ví dụ đối chiếu ở K = 5 và K = 6. Tệp K = 4 xuất 15 mẫu mỗi chủ đề. Bộ dò tín hiệu cũ gắn **0/60** mẫu xác suất cao và **0/24** mẫu ngẫu nhiên K = 4. Khi đọc mẫu, các trường hợp có từ việc làm hoặc môi giới như `P2_0465` (học tiếng gắn học bổng báo), `P2_0092` (so sánh bằng senmon với bằng đại học để định hướng nghề) và `P4_0004` (làm thêm để trả học phí) vẫn có liên hệ với lộ trình học. Các tỷ lệ gắn cờ mô tả mẫu và không đo tỷ lệ sai của toàn bộ tập.

Một số dòng như `P1_0044` so sánh chương trình có thực tập với du học trực tiếp, `P2_0636` thiên về kinh nghiệm học ngoại ngữ, và `P2_1192` bình luận vui về cấp ba/anime cần người nghiên cứu xác nhận mức liên quan. Ba ID được đánh dấu trong tệp rà soát mẫu bổ sung. Nội dung nhắc đến học hoặc du học có thể thiếu chiều sâu hoặc thiếu ngữ cảnh video. [Mẫu xác suất cao của mọi K](lda_k_review_samples.csv) · [Mẫu ngẫu nhiên dùng chung](lda_k_random_review_samples.csv) · [Mẫu đại diện K = 4](lda_topic_review.csv) · [Ghi chú rà soát mẫu, chờ người xác nhận](lda_relevance_sample_review.csv)

## 5. Tên chủ đề và bình luận minh chứng ở K = 4

| Chủ đề | Số văn bản | Tên diễn giải tạm thời | Từ khóa đứng đầu và minh chứng |
|---|---:|---|---|
| Topic_0 | 27 | Chọn trường và trải nghiệm học tại APU | `apu`, `trường`, `học`, `nhật`, `sinh_viên`. `P4_0100` (0,9937) và `P4_0105` (0,9872) bàn về môi trường học và trải nghiệm tại APU. |
| Topic_1 | 39 | Hồ sơ và học bổng du học | `du_học`, `học_bổng`, `hồ_sơ`, `báo`, `nộp`. `P1_0085` (0,9751) hỏi yêu cầu hồ sơ MEXT, `P1_0083` (0,9553) hỏi quy trình MEXT theo diện trường tiến cử. |
| Topic_2 | 107 | Lộ trình ngoại ngữ, trường tiếng và bậc học tiếp theo | `học`, `nhật`, `du_học`, `trường`, `tiếng`, `đại_học`, `senmon`. `P1_0003` (0,9674) trao đổi về N2 và học tiếp sau trường tiếng, `P2_0094` (0,9009, mẫu ngẫu nhiên) hỏi về bằng cấp senmon. Đây là cụm rộng và còn chồng lấn với chủ đề chọn trường. |
| Topic_3 | 26 | Học phí, sinh hoạt phí và khả năng trang trải | `tiền`, `học_phí`, `đóng`, `đủ`, `học_bổng`. `P4_0004` (0,9837) bàn việc làm thêm và chi trả học phí, `P1_0005` (0,9746) bàn chi phí du học tự túc. |

| Nhãn trong tập mô hình | Topic_0 | Topic_1 | Topic_2 | Topic_3 |
|---|---:|---:|---:|---:|
| Motivation | 7 | 0 | 8 | 1 |
| Purpose | 3 | 0 | 21 | 1 |
| Preparation | 14 | 35 | 34 | 17 |
| Other | 3 | 4 | 44 | 7 |

[Từ khóa và trọng số theo chủ đề](lda_topics.csv) · [Chủ đề và xác suất theo ID](lda_doc_topics.csv)

## 6. Đồng bộ đầu ra, sao lưu và kiểm thử

| Hình | Tệp đã cập nhật | Dữ liệu hình |
|---|---|---|
| 1 | [Phân bố nhãn](figures/hinh_01_phan_bo_nhom_phan_loai.png) | `classified_data.csv`, 4 nhãn trong 199 dòng `relevant` |
| 2 | [TF-IDF tổng thể](figures/hinh_02_tfidf_tu_khoa_tong_the.png) | `tfidf_results.csv`, `ALL`, n = 199 |
| 3 | [TF-IDF Motivation](figures/hinh_03_tfidf_nhom_dong_luc_motivation.png) | `tfidf_results.csv`, n = 16 |
| 4 | [TF-IDF Purpose](figures/hinh_04_tfidf_nhom_muc_dich_purpose.png) | `tfidf_results.csv`, n = 25 |
| 5 | [TF-IDF Preparation](figures/hinh_05_tfidf_nhom_chuan_bi_preparation.png) | `tfidf_results.csv`, n = 100 |
| 6 | [Từ khóa LDA](figures/hinh_06_lda_tu_khoa_tung_chu_de.png) | `lda_topics.csv`, 4 chủ đề |
| 7 | [Phân bố chủ đề](figures/hinh_07_lda_phan_bo_chu_de.png) | `lda_doc_topics.csv`, 199 chủ đề chiếm ưu thế |
| 8 | [Chủ đề và nhãn](figures/hinh_08_lda_chu_de_vs_nhom_phan_loai.png) | `lda_doc_topics.csv`, tỷ lệ theo từng nhãn |
| So sánh K | [Coherence, đa dạng và perplexity](figures/lda_k_comparison.png) | `lda_evaluation.csv`, cùng 199 văn bản cho K = 2–6 |

Bản sao tất cả đầu ra trước lần sửa nằm ở `/private/tmp/ai-group-relevance-before.nzeevC`. SHA‑256 xác nhận Excel nguồn, danh sách loại trừ và bản Word tham chiếu khớp bản trước khi sửa. Các tài liệu và ghi chú lần chạy 564 văn bản vẫn dùng làm bằng chứng lịch sử. CSV Orange đã đồng bộ lại nhãn, với cấu trúc và `text_model` giữ như trước. Word Cloud nằm ngoài phạm vi đợt này.

Đã chạy thành công 6 test qua `python -m unittest discover -s tests -v`, rồi chạy `python main.py` và `python -m src.lda_evaluation` trên Excel thật. Kiểm tra đối chiếu xác nhận ID và nhãn, tất cả dòng `uncertain` vào hàng chờ, trọng số TF-IDF tính lại từ corpus, xác suất LDA, mã hóa và dữ liệu Orange, SHA‑256 của hai nhóm đầu ra và chín ảnh PNG mới. [Các test phạm vi](../tests/test_relevance.py)

## 7. Số liệu cần thay trong báo cáo Word

- **Tóm tắt, mục 3–4 và bảng tổng kết:** Giữ 1.511 dòng đầu vào, 58 dòng loại ở Cleaning và 1.453 dòng sạch. Cập nhật bốn nhãn thành **101 / 85 / 372 / 895**. Thêm bước lọc phạm vi với **199 relevant, 222 off_topic và 1.032 uncertain**. Hàng chờ duyệt gồm **1.345** dòng. Giải thích vì sao **58 dòng Other liên quan** vẫn vào tập mô hình. Thay Hình 1.
- **Mục 5:** Thay cỡ mẫu **564 → 199**. Theo nhóm, dùng **16 / 25 / 100**. Cập nhật từ khóa, trọng số và Hình 2–5 từ CSV mới.
- **Mục 6:** Thay K chính **3 → 4**. Thêm bảng K = 2–6 mới, mô tả theo bốn cụm và cập nhật phân bố **27 / 39 / 107 / 26**. Thay Hình 6–8 và biểu đồ so sánh K. Chọn lại bình luận theo ID và xác suất mới vì số Topic có thể đổi giữa các mô hình.
- **Hạn chế và kết luận:** Nêu rõ 1.032 ca `uncertain` đang chờ duyệt, nhãn theo luật chưa qua kiểm định trên tập độc lập, và các cụm chỉ mô tả tập đã lọc. Số lượng chủ đề được chọn theo chỉ số và khả năng diễn giải, còn độ bao phủ của corpus phụ thuộc quy tắc lọc.
