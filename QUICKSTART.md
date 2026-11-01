# DataMate — chạy nhanh

1. Cài Python 3.10–3.12 và tạo virtual environment:

   ```text
   python -m venv .venv
   ```

2. Cài PyTorch khớp CUDA/GPU theo hướng dẫn PyTorch, rồi cài:

   ```text
   .venv\Scripts\python.exe -m pip install -r requirements-inference.txt
   ```

   Trên Linux, thay `.venv\Scripts\python.exe` bằng
   `.venv/bin/python`.

3. Chuẩn bị sẵn checkpoint Transformers local và business data theo layout
   `<business-data-root>/<business_id>/`. DataMate không tải model.

4. Copy `config/inference.example.json` thành
   `config/inference.local.json`, cập nhật đường dẫn đến câu hỏi JSONL, thư mục
   doanh nghiệp, checkpoint và hai file output.

5. Chạy từ repository root:

   ```text
   .venv\Scripts\python.exe scripts\run_inference.py --config config\inference.local.json
   ```

   Linux:

   ```bash
   .venv/bin/python scripts/run_inference.py --config config/inference.local.json
   ```

Run log và submission JSONL nằm ở các đường dẫn đã cấu hình. Submission chỉ là
schema tham khảo, chưa xác minh với validator cuộc thi. Inference hiện yêu cầu
CUDA; nếu máy không có GPU phù hợp, lệnh sẽ báo lỗi. Xem [INSTALL.md](INSTALL.md)
để cài chi tiết và [README.md](README.md) để biết giới hạn.
