# Cài đặt DataMate

Tài liệu này tạo môi trường Python riêng cho DataMate. Cài packages và chuẩn bị
checkpoint là bước trước khi chạy; inference không tải model hay gọi API.

## Điều kiện chung

- Python 3.10–3.12 (khuyến nghị 3.12).
- Với inference: NVIDIA GPU/driver và bản PyTorch CUDA tương thích.
- Checkpoint Transformers đã chuẩn bị cục bộ; chưa có checkpoint được chọn hoặc
  phát hành kèm repository.
- Đủ dung lượng cho dependency, dữ liệu và model.

## Windows (PowerShell)

Tại thư mục clone repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Nếu PowerShell chặn activate, có thể gọi trực tiếp
`.venv\Scripts\python.exe` thay vì đổi execution policy.

Cài bản PyTorch được chọn theo CUDA/GPU từ chỉ dẫn chính thức của PyTorch cho
máy đó. Sau đó cài inference dependencies:

```powershell
python -m pip install -r requirements-inference.txt
```

Để chạy test/phát triển thay vì inference:

```powershell
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` không cài torch/transformers. Chỉ cài
`requirements-training.txt` trên máy train riêng đã chuẩn bị CUDA; không cần
cho inference.

## Ubuntu/Linux

Cài Python 3.10–3.12 và `venv` bằng package manager của hệ điều hành nếu còn
thiếu. Ví dụ với Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Cài PyTorch theo đúng phiên bản CUDA/driver/GPU trên máy từ hướng dẫn chính
thức của PyTorch; tiếp theo:

```bash
python -m pip install -r requirements-inference.txt
```

Cho phát triển/test:

```bash
python -m pip install -r requirements-dev.txt
```

## Kiểm tra môi trường

```text
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

`torch.cuda.is_available()` phải là `True` để inference hiện tại chạy được.
Kiểm tra GPU có hỗ trợ precision đã cấu hình; mặc định là `bf16`.

Chạy test:

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

Ubuntu/Linux:

```text
PYTHONPATH=src python -m pytest -q
python -m compileall -q src tests scripts
```

## Chuẩn bị file local

1. Tạo business data root có layout `<root>/<business_id>/` và đặt các bảng
   CSV/XLSX/SQLite vào đúng thư mục business.
2. Tạo JSONL input với một object mỗi dòng, có `id`, `business_id`, `question`;
   ngày `as_of_date` nếu cần.
3. Chuẩn bị checkpoint Transformers đầy đủ trong một thư mục local. LoRA adapter
   cần được merge/export trước; code không cho phép remote custom code.
4. Copy `config/inference.example.json` thành
   `config/inference.local.json`, sửa các đường dẫn. Relative path tính từ
   thư mục `config`.
5. Đảm bảo thư mục output riêng, không nằm trong input/data root.

Chạy:

```text
python scripts/run_inference.py --config config/inference.local.json
```

Đầu ra submission là schema tham khảo, chưa xác minh với validator cuộc thi.
Nếu thiếu checkpoint, CUDA hoặc thư viện, CLI sẽ báo lỗi thay vì tải tài nguyên
từ Internet hoặc fallback sang API.

Xem [MODEL_SETUP.md](MODEL_SETUP.md) để biết tiêu chí checkpoint tương thích,
cách cấu hình `model_path` và kiểm tra model trên câu hỏi manual.

## Chạy offline

Sau khi cài dependencies và đặt model local, có thể ngắt mạng trước inference.
Backend bật chế độ Transformers/Hugging Face offline và dùng `local_files_only`.
Việc tạo môi trường/cài package và lấy checkpoint là khâu chuẩn bị riêng, có
thể cần mạng; dự án chưa chứng minh môi trường mới cài hoàn toàn offline.
