# Lab 21 - Evaluation Report

**Học viên**: `Bùi Văn Đạt` - `2A202600255`  
**Ngày nộp**: `2026-05-07`  
**Submission option**: `B - GitHub + Hugging Face Hub`

## 1. Setup

- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, lấy ngẫu nhiên `200` mẫu
- **Train / Eval split**: `180 / 20`
- **max_seq_length**: `1024` (`p95 = 562`, làm tròn lên và cap theo notebook T4)
- **GPU**: Tesla T4 16 GB trên Google Colab
- **Training cost estimate**: khoảng `$0.07` cho tổng `11.8` phút train, với giả định `$0.35/hour`
- **Hugging Face adapter link**: https://huggingface.co/Datbv/lab21-qwen2.5-3b-r16/
- **GitHub repo**: `<Điền link GitHub repo>`

Mình chọn `Qwen2.5-3B` bản 4-bit của Unsloth vì đây là cấu hình phù hợp với Colab T4, đủ nhẹ để chạy QLoRA nhưng vẫn giữ được chất lượng tốt cho bài toán instruction-following tiếng Việt. Dataset Vietnamese Alpaca được chọn vì đã đúng tinh thần Alpaca format, có câu lệnh và câu trả lời tương đối rõ ràng, giúp quan sát được tác động của fine-tuning lên phong cách trả lời và độ bám format. Với quy mô 200 mẫu, notebook có thể hoàn thành 3 rank experiment trong thời gian ngắn mà vẫn cho ra chênh lệch đủ rõ về perplexity, VRAM và số trainable parameters.

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|------|------------------|------------------|----------------|-----------|------------|
| 8 | 1,843,200 | 3.85 | 7.22 | 1.5577 | 4.7479 |
| 16 | 3,686,400 | 4.24 | 6.62 | 1.5161 | 4.5544 |
| 64 | 14,745,600 | 3.72 | 8.00 | 1.4768 | 4.3790 |

Nhận xét nhanh:

- Khi tăng rank từ `8 -> 16 -> 64`, số trainable parameters tăng rất mạnh, đặc biệt ở `r=64` lớn gấp 4 lần `r=16` và gấp 8 lần `r=8`.
- Về chất lượng định lượng, perplexity giảm dần khi tăng rank: `4.7479 -> 4.5544 -> 4.3790`, nghĩa là model học tốt hơn trên tập eval khi rank lớn hơn.
- Tuy nhiên, mức cải thiện từ `r=16` lên `r=64` nhỏ hơn nhiều so với mức tăng trainable parameters. Điều này cho thấy đã bắt đầu xuất hiện dấu hiệu diminishing returns.
- Peak VRAM của `r=16` trong lần chạy này thấp hơn `r=8`, khả năng do dao động runtime và memory reuse trong Colab. Xu hướng tổng thể vẫn là `r=64` tốn tài nguyên nhất.

## 3. Loss Curve Analysis

Notebook T4 này theo thiết kế đã tắt `eval-during-training` để tiết kiệm VRAM, vì vậy trong quá trình train chỉ theo dõi được train loss curve thay vì có cả eval loss theo từng step. Dựa trên kết quả cuối kỳ, cả ba rank đều hội tụ ổn định và không có dấu hiệu thất bại huấn luyện. Eval loss sau train giảm dần khi tăng rank, cho thấy mô hình tận dụng được thêm capacity của LoRA adapter.

Vì không có chuỗi `eval_loss` giữa quá trình train trong file export, kết luận về overfitting cần dựa vào kết quả cuối cùng kết hợp với qualitative outputs. Trong lần chạy này, chưa thấy dấu hiệu rõ ràng rằng model bị overfit nặng: perplexity của các rank đều ở mức hợp lý, và qualitative outputs vẫn giữ được khả năng trả lời instruction tương đối tự nhiên. Với bài lab chạy trên T4 và dataset 200 mẫu, `3 epochs` là một cấu hình phù hợp để cân bằng thời gian và chất lượng.

## 4. Qualitative Comparison

### Example 1

**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.  
**Base model**: Trả lời đúng ý chung, giải thích machine learning là một nhánh của AI và nhấn mạnh yếu tố học từ dữ liệu.  
**Fine-tuned r=16**: Trả lời vẫn đúng chủ đề nhưng câu chữ có vẻ thiên về diễn đạt học thuật hơn, nhấn mạnh dự đoán từ dữ liệu và thuật toán học máy.  
**Nhận xét**: `Improved nhẹ`. Câu trả lời của bản fine-tuned mạch lạc hơn ở phần mở đầu, dù khác biệt chưa quá lớn.

### Example 2

**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.  
**Base model**: Đưa ra hướng giải bằng đệ quy hoặc vòng lặp, nhưng phần kiểm tra input còn hơi lỏng và câu trả lời bị cắt ở đoạn cuối.  
**Fine-tuned r=16**: Trả lời rõ ràng hơn, có xử lý input âm bằng `ValueError`, dùng vòng lặp và cấu trúc code sạch hơn.  
**Nhận xét**: `Improved`. Output fine-tuned có tính thực dụng và an toàn hơn với người dùng.

### Example 3

**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.  
**Base model**: Có cấu trúc liệt kê, nhưng diễn đạt dài và hơi lặp ý.  
**Fine-tuned r=16**: Trả lời súc tích hơn, theo dạng danh sách rõ ràng và trực tiếp hơn.  
**Nhận xét**: `Improved`. Fine-tuned model bám format liệt kê tốt hơn, phù hợp với kiểu instruction dataset.

### Example 4

**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.  
**Base model**: Mô tả đúng ở mức tổng quan rằng QLoRA là biến thể lượng tử hóa để tiết kiệm tài nguyên.  
**Fine-tuned r=16**: Có vẻ diễn giải tự tin nhưng lại dùng mở rộng tên gọi không chính xác cho LoRA, chuyển hướng sang ngôn ngữ “regularization” thay vì “low-rank adaptation”.  
**Nhận xét**: `Degraded`. Đây là ví dụ quan trọng cho thấy fine-tuning không tự động cải thiện factual precision ở mọi trường hợp.

### Example 5

**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.  
**Base model**: Trả lời đúng khung ý, giải thích đây là ba cách khác nhau để cải thiện hiệu suất mô hình.  
**Fine-tuned r=16**: Câu trả lời vẫn đúng định hướng chung, có vẻ trôi chảy hơn nhưng chưa tạo ra khác biệt lớn về nội dung cốt lõi.  
**Nhận xét**: `Same to slight improvement`. Fine-tuned model có giọng văn đều hơn, nhưng lợi ích chính nằm ở format hơn là thêm kiến thức mới.

## 5. Conclusion về Rank Trade-off

Trong thí nghiệm này, `r=16` là lựa chọn có ROI tốt nhất nếu xét theo góc nhìn triển khai thực tế trên GPU nhỏ như Tesla T4. Rank `8` là phương án rẻ nhất về số trainable parameters, nhưng perplexity cao nhất trong ba cấu hình, cho thấy năng lực biểu diễn còn hơi hạn chế. Rank `64` cho kết quả perplexity tốt nhất (`4.3790`), nhưng phải trả giá bằng số trainable parameters tăng rất mạnh lên `14.7M`, tức gấp 4 lần `r=16`. So với mức tăng capacity đó, phần cải thiện từ `r=16` xuống `r=64` chỉ còn khoảng `0.175` perplexity, tương đối nhỏ. Điều này cho thấy lợi ích bổ sung bắt đầu giảm dần khi rank tăng cao hơn mức trung bình.

Nếu mục tiêu là làm lab, báo cáo trade-off rõ ràng và vẫn giữ khả năng chạy ổn định trên Colab T4, `r=16` là cấu hình cân bằng nhất. Nó tốt hơn `r=8` khá rõ về perplexity, trong khi chưa phình to mạnh như `r=64`. Nếu đem triển khai production cho một use case tương tự, mình sẽ ưu tiên `r=16` khi chi phí hạ tầng, tốc độ lặp và độ ổn định quan trọng; chỉ chọn `r=64` nếu đã xác nhận rằng bài toán thật sự cần thêm capacity và lợi ích chất lượng đủ bù chi phí vận hành tăng thêm.

## 6. What I Learned

- Fine-tuning bằng LoRA/QLoRA hữu ích nhất cho format, style và hành vi trả lời; nó không đảm bảo sửa triệt để lỗi kiến thức nền.
- Tăng rank luôn giúp mô hình mạnh hơn về mặt capacity, nhưng không phải lúc nào cũng mang lại mức cải thiện tương xứng với tài nguyên bỏ ra.
- Trên Colab T4, QLoRA với Unsloth là một lựa chọn rất thực tế để làm thí nghiệm end-to-end về fine-tuning LLM.


