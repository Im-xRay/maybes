# Bộ manual test mẫu (dữ liệu tổng hợp)

Đây là ví dụ cấu trúc do DataMate tạo, không lấy câu hỏi/đáp án/dữ liệu từ các
bộ train, public-test hoặc private-test.

## Các file

- `questions.jsonl`: 100 câu hỏi; 80 câu có `expected_answer`, 20 câu để `null`.
- `business/biz_demo/sales.sqlite`: bảng `sales` tổng hợp 500 dòng với các cột
  `order_id`, `product`, `quantity`, `revenue`.
- `results.jsonl`: ví dụ kết quả gồm câu trả lời, QueryPlan, evidence và trace.
- `report.json`: báo cáo tổng hợp của lượt chạy minh họa.

## Chạy với model local của bạn

Từ thư mục gốc repository, trên máy đã cài model/CUDA:

```cmd
python scripts\run_manual_test.py --questions manual_test\examples\questions.jsonl --business-data-root manual_test\examples\business --model-path "D:\path\to\local-checkpoint" --results manual_test\results.jsonl --report manual_test\report.json
```

Kết quả mới nằm ngoài `examples/` để giữ nguyên kết quả mẫu trong Git; các file
`manual_test\results.jsonl` và `manual_test\report.json` bị bỏ qua khi commit.

**Lưu ý:** `results.jsonl` và `report.json` trong thư mục này được tạo bằng
planner giả lập theo mẫu câu hỏi, không phải AI. 80/80 câu khớp chỉ xác nhận
pipeline tính toán trên dữ liệu tổng hợp; không phải kết quả hoặc độ chính xác
của model.
