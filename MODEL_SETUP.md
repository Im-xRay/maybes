# Tích hợp model local vào DataMate

Tài liệu này hướng dẫn nối checkpoint local vào pipeline inference. Cấu hình
mẫu Qwen2.5-0.5B-Instruct đã có trong repository; DataMate không phân phối trọng
số model. Smoke test local đã load model nhưng 0/3 câu tự tạo sinh QueryPlan hợp
lệ; không coi đây là benchmark accuracy hoặc xác nhận phù hợp điều lệ.

## Model được dùng ở bước nào?

Model chỉ chuyển câu hỏi nghiệp vụ thành `QueryPlan` JSON. DataMate kiểm tra
plan, đọc dữ liệu doanh nghiệp và để executor xác định tính toán câu trả lời.
Model không trực tiếp truy cập cơ sở dữ liệu hoặc trả kết quả cuối thay executor.

Đường chạy hiện tại:

```text
câu hỏi + schema + context
        → Transformers local checkpoint
        → QueryPlan JSON
        → validator → executor → answer/evidence/trace
```

Entrypoint là `scripts/run_inference.py`; lớp nạp model là
`src/datamate/local_inference.py`.

## Điều kiện tương thích

Checkpoint cần:

- Là model causal language model có thể nạp bằng
  `transformers.AutoModelForCausalLM` và tokenizer bằng
  `transformers.AutoTokenizer`.
- Có đủ tokenizer, config và weights trong thư mục local.
- Không phụ thuộc `trust_remote_code=True`; backend chủ động đặt
  `trust_remote_code=False`.
- Có thể sinh tiếp văn bản từ prompt. Chat template được dùng nếu tokenizer có
  khai báo; nếu không, backend gửi prompt dạng text thường.
- Vừa VRAM của GPU ở precision bạn chọn, hoặc dùng CPU với `device: "cpu"` và
  `precision: "fp32"`, đồng thời có không quá `parameter_cap` tham số (mặc định
  15 tỷ).

Model instruction/chat thường phù hợp hơn model base cho sinh JSON, nhưng đây
chỉ là tiêu chí lựa chọn cần benchmark, không phải xác nhận một model cụ thể.
Một checkpoint tương thích API vẫn có thể tạo QueryPlan không đúng ngữ nghĩa.

## 1. Chuẩn bị checkpoint

Tải/chuẩn bị checkpoint theo giấy phép, nguồn và quy định áp dụng. Khi hoàn tất,
đặt thư mục model ở nơi local; ví dụ:

```text
D:\DataMate-ZaloAI\models\my-checkpoint\
    config.json
    tokenizer.json / tokenizer.model / tokenizer_config.json
    model.safetensors hoặc shard weights và index tương ứng
```

Tên file thực tế tùy checkpoint. Nếu dùng LoRA, backend này không nạp adapter
riêng: trước tiên cần merge/export thành model đầy đủ tương thích
`AutoModelForCausalLM`. Không commit checkpoint, token truy cập hoặc thông tin
tài khoản vào repository.

Trước khi nạp runtime, backend kiểm tra `config.json`, tokenizer artifact và
weights local (kể cả các shard được khai báo trong index). Sau đó Transformers
vẫn phải đọc và xác thực checkpoint thực tế; kiểm tra tệp tồn tại không chứng
minh checkpoint hợp lệ hoặc tương thích kiến trúc.

## 2. Cài runtime

Tạo môi trường ảo rồi cài **bản PyTorch đúng với GPU/driver/CUDA của máy** nếu
dùng CUDA, theo hướng dẫn chính thức của PyTorch. Có thể chọn CPU nếu không có
đủ VRAM; inference CPU dùng `fp32` và có thể chậm đáng kể. Sau đó cài dependency
inference:

```powershell
python -m pip install -r requirements-inference.txt
```

Trên Linux, dùng cùng lệnh trong virtual environment Linux.
`requirements-inference.txt` khai báo các dải tương thích cho PyTorch,
Transformers, Tokenizers, Hugging Face Hub và psutil nhưng không chọn build CUDA
cho máy của bạn. Không dùng lệnh cài PyTorch chung một cách mù quáng nếu máy cần
build CUDA cụ thể. Chỉ cài trong virtual environment riêng, không cài/nâng/hạ
package Python toàn cục.

Kiểm tra runtime:

```powershell
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Chọn `device` bằng config/flag; backend không tự fallback khi CUDA lỗi hoặc
thiếu VRAM. CUDA mặc định dùng `bf16`, chỉ chạy nếu GPU hỗ trợ; GPU FP16-only
cần `precision: "fp16"`. CPU yêu cầu `precision: "fp32"`. Không thể suy ra VRAM
cần thiết chỉ từ tên model. Backend ước lượng bộ nhớ weights và KV cache, đo
thời gian load/generate, RAM tiến trình và peak VRAM CUDA; ước lượng không gồm
workspace/fragmentation và không thay thế smoke test trên GPU thực tế.

## 3. Khai báo đường dẫn model

Tạo config local từ mẫu:

```powershell
Copy-Item config\inference.example.json config\inference.local.json
```

Trong `config\inference.local.json`, sửa tối thiểu:

```json
{
  "input": "..\\input\\questions.jsonl",
  "business_data_root": "..\\input\\business",
  "model_path": "D:\\DataMate-ZaloAI\\models\\my-checkpoint",
  "log": "..\\output\\run.jsonl",
  "submission_output": "..\\output\\submission.jsonl",
  "device": "cuda",
  "local_files_only": true,
  "precision": "bf16",
  "trust_remote_code": false,
  "max_prompt_tokens": 16384,
  "max_new_tokens": 1024,
  "parameter_cap": 15000000000,
  "generation_timeout_seconds": null
}
```

Đây là các trường cần thay trong config mẫu; giữ lại các trường context khác
nếu cần. `model_path` có thể là đường dẫn tuyệt đối hoặc tương đối so với thư
mục chứa config. Config cá nhân `config/inference.local.json` được Git ignore.
Giữ `local_files_only: true` và `trust_remote_code: false`; CLI từ chối bật
truy cập Hub hoặc remote code.
`generation_timeout_seconds` là giới hạn tùy chọn cho mỗi lần sinh; `null`
nghĩa là không đặt giới hạn thời gian. Nếu cấu hình, backend chuyển giới hạn cho
Transformers và báo lỗi thay vì trả một QueryPlan bị cắt khi hết thời gian.
`max_prompt_tokens` giới hạn độ dài phần câu hỏi/ngữ cảnh đã token hóa; phần
token sinh tối đa được giới hạn riêng bởi `max_new_tokens`. Ước lượng KV cache
dùng tổng hai giới hạn này. Các thư mục input/output phải tồn tại phù hợp;
output không được nằm trong input/data root.

### Chuẩn hóa ô trống lặp lại theo nhóm (tùy chọn)

Mặc định ô trống vẫn là dữ liệu thiếu. Chỉ khi rulebook của business xác nhận
ô trống ở các dòng tiếp theo lặp lại giá trị của cùng một nhóm, mới khai báo
`fill_down_group` trong `coercions`. Ví dụ cấu hình một bảng có mã đơn và trạng
thái được ghi ở dòng đầu:

```json
{
  "biz_example": {
    "orders.csv::orders": {
      "order_id": {
        "kind": "identifier",
        "fill_down_group": "order_id",
        "deduplicate_group": "order_id"
      },
      "status": {
        "kind": "text",
        "fill_down_group": "order_id"
      }
    }
  }
}
```

`orders.csv::orders` phải khớp tên bảng được profiler/discovery báo ra. Cột
nhóm (`order_id`) cũng cần khai báo tự điền tiếp như ví dụ để các dòng có mã
trống được gắn với nhóm hiện hành. Khi gặp mã đơn mới, trạng thái được điền lại
từ giá trị trong nhóm mới; không tự điền qua các nhóm. Chỉ cấu hình cho những
cột mà tài liệu nguồn xác nhận quy tắc này: cấu hình sai có thể biến dữ liệu
thiếu thực sự thành dữ liệu có giá trị. Chính sách được ghi vào run log và
được replay cùng kết quả.

Nếu rulebook xác nhận các dòng giống hệt nhau trong cùng một đơn là lỗi xuất
file và chỉ được tính một lần, có thể thêm `deduplicate_group` vào chính sách
cột khóa như ví dụ. So sánh dùng toàn bộ dòng sau chuẩn hóa và fill-down; dòng
khác nội dung vẫn được giữ, và dòng của đơn khác không bị gộp. Dòng không có mã
nhóm cũng không bị tự ý loại. Để giữ bộ nhớ hữu hạn, chính sách này dừng với lỗi
nếu cần theo dõi quá 100.000 dòng khác nhau có nhóm; không cắt bớt âm thầm.

Tên thay thế cũng chỉ được hợp nhất khi khai báo tường minh trên cột `text`,
ví dụ `aliases: {"tên trong file": "tên chuẩn"}`. So khớp bỏ khoảng trắng đầu/
cuối và không phân biệt hoa thường; tên không khai báo vẫn giữ nguyên. Hãy chọn
tên chuẩn vốn có trong dữ liệu nếu nó sẽ được xuất ra câu trả lời. Backend không
tự phát hiện tên đồng nghĩa; nếu shop có mã định danh thì ưu tiên mã, không gộp
theo tên.

## 4. Chạy inference

Từ thư mục gốc repository:

```powershell
python scripts\run_inference.py --config config\inference.local.json
```

Hoặc kiểm tra CLI:

```powershell
python scripts\run_inference.py --help
```

Nếu chỉ đổi precision hoặc giới hạn sinh, có thể override bằng flags, ví dụ:

```powershell
python scripts\run_inference.py --config config\inference.local.json --precision fp16 --max-new-tokens 512
```

Có thể đặt `--max-prompt-tokens` để thay giới hạn prompt; tổng context tối đa
dự kiến là prompt limit cộng `max_new_tokens`.

CLI nạp model một lần, xử lý câu hỏi JSONL tuần tự và ghi cấu hình inference
cùng run log và submission
output. Submission hiện theo schema tham khảo training rulebook; chưa xác nhận
với validator chính thức. Nếu lỗi một câu hỏi, process trả exit code khác 0 và
không xuất submission một phần.

## 5. Kiểm tra model bằng manual questions

Tự viết câu hỏi trong `manual_test/questions.jsonl` rồi chạy:

```powershell
python scripts\run_manual_test.py --business-data-root <business-root> --model-path "D:\DataMate-ZaloAI\models\my-checkpoint"
```

`manual_test/README.md` mô tả schema câu hỏi/expected answer và file kết quả.
Chỉ tự thêm đáp án chuẩn nếu bạn có đáp án đáng tin cậy; không để đáp án chuẩn
lọt vào prompt planner. Bắt đầu với một vài câu hỏi tự tạo và kiểm tra
QueryPlan, evidence, trace cùng answer; test tự động dùng planner stub không
thay thế bước kiểm tra checkpoint thật.

## Offline, an toàn và chẩn đoán

Sau khi đã cài packages và chuẩn bị checkpoint, backend đặt
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, dùng `local_files_only=True` và
không gọi API inference. Cài dependency hoặc lấy checkpoint trước đó có thể
cần mạng; đó là bước chuẩn bị riêng. Đường chạy inference không tự tải model.

| Thông báo/tình trạng | Kiểm tra |
|---|---|
| `model_path must be a local checkpoint directory` | Đường dẫn config phải trỏ đến thư mục checkpoint đã tồn tại. |
| Thiếu PyTorch hoặc Transformers | Cài PyTorch CUDA phù hợp trước, rồi cài inference requirements. |
| CUDA không khả dụng/không đủ VRAM | Kiểm tra driver, CUDA build của PyTorch và `torch.cuda.is_available()`; nếu muốn CPU, đặt `device: "cpu"` và `precision: "fp32"` rõ ràng. |
| GPU không hỗ trợ `bf16` | Chỉ đổi sang `fp16` nếu GPU hỗ trợ và checkpoint chạy ổn với precision đó. |
| Không tìm thấy config/tokenizer/weights | Chuẩn bị đầy đủ artifact của checkpoint; backend không tải phần còn thiếu. |
| Vượt parameter cap hoặc hết VRAM | Kiểm tra đúng checkpoint, precision, VRAM; không tăng cap nếu chưa xác nhận tài nguyên. |
| QueryPlan sai hoặc JSON không hợp lệ | Xem run log/manual results. Khả năng tạo plan đúng phải benchmark riêng cho checkpoint/model prompt. |

Tài liệu này mô tả đường tích hợp trong source. Kết quả load/inference,
QueryPlan validation và giới hạn đã đo được ghi tại
[planning/21_LOCAL_MODEL_INTEGRATION_REPORT.md](planning/21_LOCAL_MODEL_INTEGRATION_REPORT.md).
