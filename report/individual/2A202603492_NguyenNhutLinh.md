# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603492
- **Họ và tên:** Nguyễn Nhựt Linh
- **SEED** (in ở ô đầu notebook): 7333

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ok |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ok |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ok |

## 2. Kết quả chính

| Mô hình | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 (ms/tin) | p95 (ms/tin) | USD / 1k tin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. BoW + LogReg (tự code) | 0.985 | 0.776 | 0.868 | 0.857 | 5.076 | 21224.256 | 0.066 | 0.095 | 0.0 |
| 2. BoW + MLP (deep learning) | 0.987 | 0.918 | 0.951 | 0.974 | 8.284 | 8511.461 | 0.328 | 2.419 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.882 | 0.965 | 0.921 | 0.923 | 93.375 | 18.862 | 98.754 | 141.249 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.930 | 0.941 | 0.936 | 0.947 | 93.827 | 26.680 | 28.842 | 40.265 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.976 | 0.941 | 0.958 | 0.947 | 94.318 | 26.359 | 30.444 | 43.780 | 0.0 |
| 4a. LLM nhỏ zero-shot | 0.952 | 1.000 | 0.976 | 0.976 | 0.000 | 0.955 | 511.925 | 2641.739 | 0.0 |
| 4b. LLM lớn zero-shot | 0.909 | 1.000 | 0.952 | 0.952 | 0.000 | 1.037 | 513.868 | 2618.370 | 0.0 |
| 4c. LLM nhỏ few-shot | 0.952 | 1.000 | 0.976 | 0.976 | 0.000 | 0.592 | 1993.996 | 2897.728 | 0.0 |

- Hàng LLM chỉ đo trên SUB (40 tin) -> so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
- SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
- Inference đo TỪ VĂN BẢN THÔ -> NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
- train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

**3.1. `ml_f1` – Dự đoán F1 của BoW + LogReg**

- **Dự đoán:** F1 khoảng 0.90, Recall thấp hơn Precision do dữ liệu mất cân bằng và mô hình có xu hướng thiên về ham.
- **Kết quả:** F1 = 0.8684, Precision = 0.9851, Recall = 0.7765.
- **Vì sao lệch:** F1 thực tế thấp hơn dự đoán vì Recall của lớp spam thấp hơn kỳ vọng. Mô hình bỏ sót một số tin spam, dù những tin được dự đoán là spam thường khá chính xác.

**3.2. `llm_models` – So sánh LLM lớn và nhỏ**

- **Dự đoán:** Model lớn có F1 cao hơn model nhỏ khoảng 2–5 điểm phần trăm.
- **Kết quả:** LLM nhỏ đạt F1 = 0.9756, trong khi LLM lớn chỉ đạt F1 = 0.9524.
- **Vì sao lệch:** Model lớn không nhất thiết tốt hơn model nhỏ trên một tác vụ phân loại spam đơn giản. Kết quả còn phụ thuộc vào cách model diễn giải prompt và đưa ra nhãn, không chỉ kích thước model.

**3.3. `llm_fewshot` – So sánh zero-shot và few-shot**

- **Dự đoán:** Few-shot giúp F1 tăng nhẹ và chi phí mỗi tin tăng khoảng 2–3 lần do phải bổ sung 4 ví dụ vào prompt.
- **Kết quả:** F1 không thay đổi, vẫn là 0.9756, nhưng latency p50 tăng từ 511.92 ms lên 1994.00 ms.
- **Vì sao lệch:** Các ví dụ bổ sung không giúp cải thiện độ chính xác vì zero-shot đã phân loại tốt. Tuy nhiên, prompt dài hơn làm tăng thời gian xử lý đáng kể.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất về F1, từ 0.857 xuống 0.333 (giảm 0.524), tiếp theo là BoW + MLP, từ 0.974 xuống 0.400 (giảm 0.574). Xét theo mức giảm tuyệt đối, BoW + MLP tụt nhiều hơn. Cả hai mô hình gặp khó khăn nhất ở nhóm **Spam giọng tự nhiên**, với tỷ lệ dự đoán đúng lần lượt chỉ 10% và 0%. Ngoài ra, BoW + LogReg chỉ đạt 12% ở nhóm **Thao túng / lừa đảo**.

Hai case cụ thể và cơ chế gây lỗi:

1. **Spam giọng tự nhiên:** Tin nhắn spam được viết giống một cuộc hội thoại hoặc thông báo bình thường, không chứa nhiều từ khóa quảng cáo quen thuộc. BoW chủ yếu dựa vào sự xuất hiện của từ khóa nên khó nhận ra ý đồ spam khi cách diễn đạt quá tự nhiên. Điều này khiến mô hình dễ bỏ sót spam (false negative).

2. **Nói về spam:** Một tin nhắn có thể chứa các từ như "spam", "free" hoặc "win" nhưng thực tế chỉ đang thảo luận, cảnh báo hoặc phân tích về spam. BoW dễ dựa vào từ khóa mà không hiểu ngữ cảnh, dẫn đến dự đoán nhầm ham thành spam (false positive). Tuy nhiên, nhóm này cũng cho thấy BoW đạt 100% ở cả hai mô hình, nên đây là hạn chế về cơ chế chung chứ không phải nhóm mà chúng thực sự thất bại trong bảng kết quả.

Nhìn chung, các mô hình dựa trên BoW nhạy với từ khóa nhưng thiếu khả năng hiểu ngữ cảnh. LLM nhỏ có kết quả tốt nhất với F1 = 0.918 trên 50 case khó, cho thấy khả năng xử lý ngữ cảnh và các cách diễn đạt phức tạp tốt hơn.

## 5. Thử thách cá nhân

### Đề #2 – Thay đổi kích thước Vocabulary

**5.1. Cách làm**

Em thử nghiệm với 3 kích thước vocabulary: 100, 2000 và 5000 từ. Với mỗi kích thước, em xây dựng vocabulary từ tập TRAIN, vector hóa dữ liệu, sau đó huấn luyện lại Logistic Regression với `lr=0.5` và `epochs=300`. Cuối cùng, em đánh giá các mô hình trên cùng tập TEST bằng Accuracy, Precision, Recall và F1.

**5.2. Kết quả**

| Vocabulary size | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 100 | 0.9548 | 0.9310 | 0.6353 | 0.7552 |
| 2000 | 0.9742 | 0.9851 | 0.7765 | 0.8684 |
| 5000 | 0.9742 | 0.9851 | 0.7765 | 0.8684 |

**5.3. Giải thích**

Khi tăng vocabulary từ 100 lên 2000 từ, F1 tăng từ 0.7552 lên 0.8684. Nguyên nhân là vocabulary lớn hơn giúp mô hình nhận diện được nhiều từ đặc trưng của spam hơn, giảm tình trạng mất thông tin khi vector hóa và cải thiện khả năng phát hiện spam (Recall tăng từ 0.6353 lên 0.7765).

Tuy nhiên, khi tăng tiếp lên 5000 từ, kết quả không thay đổi. Có thể do 2000 từ đã chứa đủ những đặc trưng quan trọng trong bộ dữ liệu này, các từ bổ sung không mang lại nhiều thông tin hữu ích hoặc không xuất hiện đủ thường xuyên để cải thiện mô hình.

**5.4. Kết luận**

Vocabulary 2000 từ là lựa chọn phù hợp trong thử nghiệm này. Giảm xuống 100 từ làm mất nhiều thông tin quan trọng, trong khi tăng lên 5000 từ không cải thiện F1. Vì vậy, vocabulary lớn hơn không phải lúc nào cũng giúp mô hình hoạt động tốt hơn.

## 6. Phương án cho sản phẩm thật

Nhà mạng cần xử lý 10 triệu SMS/ngày, tương đương khoảng 116 tin/giây, với yêu cầu p95 dưới 50 ms/tin.

Dựa trên kết quả thực nghiệm, em chọn BoW + MLP vì đây là phương án cân bằng tốt giữa chất lượng phân loại và tốc độ xử lý.

Các số liệu chứng minh:

F1 test = 0.951, cao hơn BoW + LogReg (0.868).

p95 = 2.419 ms/tin, thấp hơn nhiều so với SLA 50 ms.

Throughput = 8.511 tin/s, cao hơn đáng kể so với nhu cầu trung bình 116 tin/s.

So với các phương án khác, MiniLM + LogReg và MiniLM + MLP head cũng đáp ứng SLA nhưng có độ trễ cao hơn. Trong khi đó, LLM nhỏ zero-shot có F1 cao (0.976 trên SUB 40) nhưng p95 lên đến 2641.739 ms, không phù hợp với yêu cầu độ trễ.

Đánh đổi: Em chấp nhận việc BoW + MLP chưa hiểu ngữ cảnh tốt như LLM và có thể gặp khó khăn với spam viết tự nhiên hoặc né từ khóa. Đổi lại, mô hình có tốc độ xử lý nhanh, chất lượng phân loại tốt và phù hợp hơn với hệ thống cần xử lý lượng SMS lớn.

Trong thực tế, cần triển khai kiểm thử tải đồng thời và dự phòng tài nguyên để đảm bảo SLA ở giờ cao điểm.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
| **Phần** | **Bạn hỏi AI gì (prompt)** | **AI trả lời/viết gì** | **Bạn đã sửa / kiểm chứng gì** |
|---|---|---|---|
| Tính tay – Sigmoid | Công thức sigmoid là gì? Tại sao khi tính sigmoid với đầu vào bằng 0 lại ra kết quả 0.5? | AI giải thích công thức sigmoid và hướng dẫn thay giá trị đầu vào vào công thức. | Tự thay số và kiểm tra lại kết quả tính tay. |
| Phần 2 – ReLU | Nếu xóa `nn.ReLU()`, mạng 2 lớp Linear liên tiếp tương đương với mô hình nào? F1 dự kiến thay đổi ra sao? | AI giải thích hai lớp Linear liên tiếp không có hàm kích hoạt vẫn tương đương một phép biến đổi tuyến tính. | Đối chiếu với kết quả thực nghiệm khi thay đổi kiến trúc mạng. |
| Phần 2 – Backpropagation | `opt.zero_grad()` trong PyTorch dùng để làm gì? Điều gì xảy ra nếu quên gọi hàm này? | AI giải thích PyTorch mặc định cộng dồn gradient qua các lần backward nếu không xóa gradient cũ. | Kiểm tra vị trí gọi `zero_grad()` trong vòng lặp training. |
| Phần 3 – NumPy / kNN | Tại sao `np.argsort(scores)[:k]` có thể trả về những vector không tương đồng nhất với query? | AI giải thích thứ tự sắp xếp mặc định của NumPy và sự khác biệt giữa lấy k giá trị nhỏ nhất và lớn nhất. | Đối chiếu với giá trị similarity và kết quả top-k thực tế. |
| Phần 4 – Chi phí LLM | Tại sao khi tính chi phí API lại phải chia tổng số token cho 1.000.000? | AI giải thích đơn giá API được tính theo mỗi một triệu token nên cần quy đổi tổng token về cùng đơn vị. | Tự kiểm tra phép tính và đối chiếu với đơn giá API. |
| ML – Vocabulary | Giúp viết code thử nghiệm với vocabulary 100, 2000 và 5000 từ. | AI đề xuất code xây dựng vocabulary, vector hóa dữ liệu, train và đánh giá F1. | Đã chạy notebook và kiểm tra kết quả thực tế. |
| Phản ánh cuối bài – Câu 1 | Dựa vào bảng so sánh F1, train, throughput, p95 và USD, chọn mô hình phù hợp cho 10 triệu SMS/ngày. | AI phân tích độ trễ, throughput, chất lượng và chi phí của từng phương pháp. | Đối chiếu với biểu đồ Pareto và kết quả thực nghiệm, kiểm tra lại p95 và throughput. |
| Phản ánh cuối bài – Câu 2 | Đề xuất hệ thống router kết hợp ML và LLM. | AI đề xuất dùng ML xử lý tin chắc chắn và chuyển tin lưỡng lự sang LLM. | Xem xét ảnh hưởng của tỷ lệ chuyển sang LLM đến độ trễ p95 toàn hệ thống. |
| Phản ánh cuối bài – Câu 3 | Giúp viết phần tự phản ánh dựa trên kết quả thực nghiệm. | AI gợi ý những dự đoán chưa chính xác và bài học rút ra. | Điều chỉnh nhận xét theo kết quả thực tế của thí nghiệm.

Qua bài thực hành, tôi hiểu rõ hơn rằng kết quả của một mô hình không chỉ phụ thuộc vào thuật toán mà còn phụ thuộc vào dữ liệu, cách biểu diễn đặc trưng và cách đánh giá. Tôi cũng hiểu được sự đánh đổi giữa chất lượng dự đoán, tốc độ xử lý và chi phí: mô hình có F1 cao chưa chắc phù hợp để triển khai nếu độ trễ quá lớn. Đặc biệt, tôi học được cách sử dụng AI như một công cụ hỗ trợ giải thích khái niệm và debug, nhưng vẫn cần tự chạy thí nghiệm, kiểm tra số liệu và điều chỉnh kết luận dựa trên kết quả thực tế.
