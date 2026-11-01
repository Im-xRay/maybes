# Yêu cầu cuộc thi và trạng thái xác minh

**Nguyên tắc:** Không xem nội dung brief do người dùng cung cấp là văn bản chính thức cho tới khi đối chiếu được với portal, regulations, Rulebook, Evaluation và Docker guideline. Các yêu cầu dưới đây được phân biệt theo trạng thái.

## Yêu cầu xác nhận từ portal

| Yêu cầu | Trạng thái | Ghi chú |
|---|---|---|
| Thử thách tên DataMate — Talking To Your Data Through Agentic AI | VERIFIED | Tên và mô tả xuất hiện trong bundle của portal chính thức. |
| Bài toán cấp cao: hỏi bằng ngôn ngữ tự nhiên trên dữ liệu bảng đời thực; câu trả lời sai gây hại, im lặng không bị phạt theo mô tả portal | VERIFIED | Mô tả marketing/challenge trong bundle; không thay thế định nghĩa metric chi tiết. |
| Định dạng submission `.jsonl` | VERIFIED | Thuộc metadata challenge trong bundle; `data/train/train/rulebook.md` §9 mô tả dòng `id/type/answer`, nhưng chưa xác minh chéo với validator hiện hành. |
| Deadline 16/11/2026 23:59 GMT+7 | NEEDS_CONFIRMATION | Có trong nội dung timeline của bundle, nhưng phải xác nhận các điều kiện upload/submission bằng trang portal hiển thị và quy định chính thức. |

## Yêu cầu trong brief, cần đối chiếu chính thức

| Nhóm | Trạng thái | Yêu cầu được nêu trong brief |
|---|---|---|
| Mô hình | NEEDS_CONFIRMATION | Trọng số công khai trước 01/10/2026; tổng tham số inference ≤15B; xét tất cả model kể cả embedding/reranker/classifier và toàn bộ tham số MoE; kiểm tra license và khai báo model. |
| Offline inference | NEEDS_CONFIRMATION | Không gọi dịch vụ AI bên ngoài, không tải tài nguyên từ Internet lúc inference. |
| Dữ liệu | NEEDS_CONFIRMATION | Train được phép; public/private test không dùng để train, lấy đáp án hoặc ghi nhớ theo ID/business ID. `/data` read-only. |
| Phần cứng/thời gian | NEEDS_CONFIRMATION | 2× RTX 5090 32GB, RAM 512GB; tối đa 6 giờ tính cả khởi động/tải model; Blackwell yêu cầu CUDA ≥12.8. |
| Độ ổn định | NEEDS_CONFIRMATION | Hai lần chạy khác nhau không quá 0.03 PA. Cần xác minh chính xác cách đo. |
| Dữ liệu đầu vào | NEEDS_CONFIRMATION | Câu hỏi có id/business_id/question/as_of_date; dữ liệu nhiều CSV/XLSX/SQLite, README, rulebook, calendar. Chưa thấy file thực tế. |
| Đầu ra | PARTIALLY_VERIFIED | Training rulebook §9 mô tả `{"id","type","answer"}`, các dạng answer, unanswerable payload và abstain; chưa đối chiếu validator của cuộc thi. |
| Tính điểm | PARTIALLY_VERIFIED | Training rulebook §9 ghi reward mỗi câu đúng/sai/bỏ trống là `+1/-1/0` cùng quy tắc so giá trị; công thức aggregate Penalized Accuracy và xác nhận đây là evaluator hiện hành chưa có. |
| Docker/submission | NEEDS_CONFIRMATION | `predict.sh`, `/data`, `/result/submission.jsonl`, log; tuyệt đối chưa chốt image, lệnh, path hay cấu trúc trước Docker guideline. |

## Quyết định tuân thủ tạm thời

1. Không huấn luyện hoặc tối ưu trên public/private test.
2. Không gọi mạng/model API trong inference.
3. Không cho LLM tự thực thi mã tùy ý hoặc ghi vào `/data`.
4. Chỉ dùng schema/reward từ training rulebook làm tham chiếu tạm thời; không suy đoán công thức PA hoặc coi đó là xác nhận của submission validator hiện hành.
5. Trì hoãn mọi quyết định Docker cho đến khi Docker guideline chính thức được đọc và lưu lại.

## Nguồn

- Portal: https://challenge.zalo.ai/portal/datamate/
- Regulations: https://challenge.zalo.ai/regulations — truy cập được nhưng nội dung trả về là SPA shell, chưa xác minh.
- Training rulebook: `data/train/train/rulebook.md` §7-9 (nguồn nằm trong archive training đã tải; chưa đối chiếu với bộ quy định/evaluator trên portal).
- Brief do người dùng cung cấp trong phiên: nguồn tham khảo làm kế hoạch, không được gắn nhãn VERIFIED cho tới khi đối chiếu chính thức.
