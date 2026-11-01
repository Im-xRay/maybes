# Báo cáo operationalization & deployment

## Tóm tắt trạng thái

| Hạng mục | Kết quả |
|---|---|
| Inference entrypoint | `scripts/run_inference.py`, chạy batch theo JSONL; nhận cấu hình bằng JSON hoặc flags. |
| Model | Có hướng dẫn tích hợp checkpoint local tại `MODEL_SETUP.md`; chưa chọn/đóng gói checkpoint. Qwen3-8B từng được cân nhắc, không phải model đã được xác nhận sử dụng. |
| GPU | Bắt buộc NVIDIA CUDA theo backend hiện tại; không có CPU fallback. |
| Offline inference | Thiết kế local-only trong code, dùng checkpoint local và `local_files_only`; chưa kiểm chứng bằng model thật do thiếu checkpoint/GPU. |
| Dataset/fixture | Test integration dùng SQLite fixture tổng hợp và planner stub. Không đọc public/private test, không fine-tune và không sửa training data. |
| Submission | Serializer xuất schema tham khảo training rulebook (`id/type/answer`), chưa xác nhận với validator/portal hiện hành. |
| Docker | Chưa có Dockerfile, Docker CLI không sẵn có, technical guideline chưa xác minh; không tuyên bố Docker hoạt động. |
| Clean environment | Python 3.12.6 venv mới; cài `requirements-dev.txt`; full suite 199 passed. Chưa cài inference/model dependencies. |

## Cách chạy hiện tại

Entrypoint batch thống nhất:

```text
python scripts/run_inference.py --config config/inference.local.json
```

Tạo config local bằng cách copy `config/inference.example.json`. Các path tương
đối được tính từ thư mục file config. `config/inference.local.json` được ignore
bởi Git. Bắt buộc khai báo `input`, `business_data_root`, `model_path`, `log` và
`submission_output`; có thể override bằng flags. Config cũng hỗ trợ
`coercions`, `rulebook`, `calendar`, `as_of_date`, `precision`,
`max_new_tokens` và `parameter_cap`.

Input là JSONL UTF-8, các trường bắt buộc `id`, `business_id`, `question`;
`as_of_date` là ngày ISO-8601 tùy chọn. Nếu câu hỏi không có ngày, CLI dùng
`as_of_date` mặc định của config/`--as-of-date`. Cấu trúc dữ liệu:

```text
business-data-root/
└── <business_id>/
    ├── *.csv
    ├── *.xlsx
    ├── *.sqlite
    └── README.md              # tùy chọn
```

Rulebook và calendar có thể được phát hiện cạnh business root hoặc truyền tường
minh qua config/flags. Loader hỗ trợ CSV, XLSX và SQLite; không sửa dữ liệu
nguồn. Output log và submission phải nằm ngoài input/data root. Submission chỉ
được publish nếu toàn bộ câu hỏi thành công; có lỗi thì CLI trả exit code khác
0, ghi lỗi ra stderr và không xuất submission một phần.

Các CLI replay/manual vẫn giữ nguyên:

```text
python scripts/run_manual_test.py --business-data-root <business-root> --model-path <checkpoint-local>
python scripts/replay_run.py --help
```

## Cài đặt trên máy mới

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Cài PyTorch CUDA phù hợp với GPU/driver theo hướng dẫn PyTorch.
python -m pip install -r requirements-inference.txt
Copy-Item config\inference.example.json config\inference.local.json
# Sửa các đường dẫn trong config\inference.local.json rồi chạy:
python scripts\run_inference.py --config config\inference.local.json
```

### Ubuntu/Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Cài PyTorch CUDA phù hợp với GPU/driver theo hướng dẫn PyTorch.
python -m pip install -r requirements-inference.txt
cp config/inference.example.json config/inference.local.json
# Sửa config rồi chạy:
python scripts/run_inference.py --config config/inference.local.json
```

`requirements-inference.txt` khai báo PyTorch và Transformers; bản PyTorch CUDA
cần được chọn riêng theo driver/GPU. `requirements-dev.txt` cài
`openpyxl`/pytest nhẹ, không cài torch/transformers. `requirements-training.txt`
có thêm PEFT cho máy train, không cần khi inference.

### File cần chuẩn bị

1. Repository source.
2. Checkpoint Transformers local hợp lệ, gồm tokenizer/config/weights; nếu dùng
   LoRA thì merge/export thành model local trước. Chưa có model/checkpoint được
   phân phối kèm repository.
3. NVIDIA GPU và driver/CUDA phù hợp với PyTorch, VRAM đủ cho checkpoint và
   context. Chưa có mức VRAM tối thiểu được benchmark.
4. Câu hỏi JSONL và thư mục dữ liệu doanh nghiệp theo `business_id`.
5. Tài liệu rulebook/calendar/coercion nếu pipeline cụ thể cần các context đó.

Code không tự tải checkpoint, không cho phép `trust_remote_code`, và backend
đặt Hugging Face/Transformers ở chế độ offline. Packages và checkpoint phải
được chuẩn bị trước. Với backend hiện tại, `torch.cuda.is_available()` phải
true; `bf16` mặc định chỉ dùng được nếu GPU hỗ trợ, có thể override
`--precision fp16` nếu phù hợp.

## Audit entrypoint và pipeline

- `scripts/run_inference.py`: canonical batch inference, bootstrap source path,
  đọc config/flags, kiểm tra input/output, nạp model local, xử lý từng dòng
  JSONL và xuất run log/submission. `as_of_date` trên từng câu hỏi ưu tiên hơn
  ngày mặc định.
- `JsonPlanAdapter` gọi backend model để tạo QueryPlan từ câu hỏi nguyên bản và
  schema/context. Validator kiểm tra plan; executor thực hiện trên dữ liệu
  read-only; pipeline kiểm tra answer, tạo evidence và execution trace.
- `scripts/run_manual_test.py`: câu hỏi người dùng tự viết.
- `scripts/replay_run.py`: replay QueryPlan có sẵn, không chạy lại model.
- `scripts/evaluate_predictions.py`: evaluator local để chẩn đoán.
- `scripts/profile_data.py`: profiling schema.
- `scripts/prepare_plan_training.py`, `scripts/train_query_plans.py`,
  `scripts/merge_lora_adapter.py`: chuẩn bị dữ liệu plan được duyệt, huấn luyện
  CUDA và merge LoRA; không thuộc inference.

Không có lệnh CLI hay đường chạy chat realtime trong audit. `scripts/run_inference.py`
là batch pipeline, không phải web server/bot.

## Scaffold và phần chưa chứng minh

- Question Understanding có adapter/QueryPlan validation và Transformers backend,
  nhưng chưa có model checkpoint được chọn; integration tests dùng planner stub.
- Rulebook/README/calendar được chuyển vào prompt, không phải bộ thực thi business
  rules tổng quát. Rule tag không được executor hỗ trợ bị từ chối; chưa benchmark
  planner thật để xác nhận hiểu đúng quy tắc.
- Training/LoRA scripts là code path chuẩn bị; chưa có annotation QueryPlan đã
  duyệt, chưa cài training stack và chưa chạy fine-tuning/merge.
- Submission serializer/evaluator là implementation tham khảo dựa training
  rulebook, chưa được xác nhận bởi validator/metric chính thức.
- Data loader/executor/pipeline có coverage fixture; chưa được xác nhận trên
  môi trường GPU mới hoặc qua end-to-end inference với model.
- Không có Docker packaging hay service/API serving.

## Kết quả kiểm chứng

- Python **3.12.6**.
- Tạo `.venv-deploy-check` mới, cài `requirements-dev.txt` từ đầu. Test environment
  có pytest 8.4.2 và openpyxl 3.1.5, không có torch/transformers.
- Full test suite trong môi trường mới: **199 passed**.
- `compileall -q src tests scripts`: thành công.
- `git diff --check`: thành công.
- `--help` của cả 8 CLI (inference, manual test, replay, evaluator, profiling,
  prepare training data, train và merge LoRA): thành công.
- Integration test inference chạy SQLite fixture tổng hợp, planner stub, kiểm
  tra prompt/context, as_of_date mặc định, run log và JSONL serializer.
- Đã kiểm tra code path model: local checkpoint directory, offline flags,
  `local_files_only=True`, `trust_remote_code=False`, CUDA bắt buộc.
- Không chạy câu hỏi trên data train/public/private; không fine-tune.

Đây là kiểm chứng test suite và pipeline stub, không phải cài inference hoàn
chỉnh trên máy GPU mới. Không thể đọc dữ liệu thật qua model, tạo prediction
hoặc kiểm tra submission thực vì môi trường hiện không có PyTorch/Transformers,
NVIDIA CUDA hoặc checkpoint. Môi trường clean được cài packages từ package
index trong lúc chuẩn bị; chỉ **inference** mới được thiết kế offline.

## Docker và quy định

Repository không có Dockerfile/compose; Docker CLI không được cài hoặc không
có trên PATH trong môi trường kiểm tra. `planning/08_DOCKER_AND_SUBMISSION.md`
ghi nhận challenge-specific Docker/interface guidance chưa truy cập/xác minh
được. Vì yêu cầu không được đoán base image, CUDA, mount, entrypoint, giới hạn
image hoặc output contract, milestone này không tạo Dockerfile và không chạy
Docker build/test. Docker không được chứng nhận hoạt động.

Schema submission hiện tham chiếu training rulebook và có cảnh báo trong CLI,
nhưng chưa chạy validator của organizer; không tuyên bố file output được chấp
nhận chính thức.

## Hạn chế và bước còn thiếu

- Chọn checkpoint đúng license/quy định, đặt local và xác minh tokenizer,
  parameter count, precision, VRAM, tốc độ trên GPU mục tiêu.
- Cài CUDA-matched PyTorch/Transformers trên máy GPU riêng rồi chạy ít nhất một
  smoke test với dữ liệu được phép và câu hỏi kiểm soát do người dùng cung cấp.
- Benchmark chất lượng question-to-plan và end-to-end trên validation được phép;
  hiện accuracy model là **chưa đo**.
- Xác minh schema `id/type/answer`, metric, input/output path, runtime, network
  policy và Docker requirements với technical guide/validator chính thức trước
  khi gọi là submission/cuộc thi ready.
- Chỉ sau khi có guideline mới quyết định Docker base image, artifact và
  interface; hiện chưa có cơ sở an toàn để đóng gói.
