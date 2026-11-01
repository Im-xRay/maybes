# Báo cáo tích hợp model local Qwen2.5-0.5B-Instruct

**Ngày kiểm tra:** 2026-11-01

**Trạng thái:** backend và checkpoint chạy được trên CUDA; smoke test planner **không đạt** — 0/3 QueryPlan hợp lệ. Không coi checkpoint là sẵn sàng cho inference nghiệp vụ.

## 1. Kết luận

Checkpoint tại `F:\AI-Models\Qwen2.5-0.5B-Instruct` có đủ config, tokenizer và weights. Đã cài môi trường Python 3.12 riêng, load tokenizer/model và sinh output bằng Transformers local-only trên GTX 1060 3GB. Ba câu hỏi tiếng Việt tự tạo đều trả JSON có code fence kèm prose giải thích; nội dung cũng dùng sai cấu trúc/cột. Một lần sửa format tổng quát cho mỗi câu vẫn không tạo được output hợp lệ. Parser/validator đã từ chối cả ba; không QueryPlan giả nào được chuyển cho executor.

Pipeline QueryPlan → validator → Python Query Executor được giữ nguyên. Test fixture dùng planner stub xác nhận QueryPlan hợp lệ vẫn đi hết executor; smoke test model thật không đến được bước thực thi vì đầu ra model không hợp lệ.

## 2. Kiểm tra ban đầu và checkpoint

- Worktree lúc bắt đầu sạch trên branch `master`; không có thay đổi sẵn trong các file cần sửa.
- `data/public_test/` là untracked và được giữ nguyên, không đưa vào commit. Theo yêu cầu chạy Public Test sau đó, runner đã đọc bộ input công khai: lượt 1 bắt đầu chạy 1.000 câu nhưng dừng sau 157 câu do prompt vượt 2.048 tokens; lượt thử 1 câu với 8.192 tokens thất bại do CUDA OOM. Không dùng public/private test để huấn luyện, ghi nhớ ID/đáp án hoặc hardcode.
- Đã đọc `README.md`, `MODEL_SETUP.md`, `requirements-inference.txt`, `scripts/run_inference.py`, toàn bộ backend `src/datamate/local_inference.py`, `QuestionUnderstanding`, QueryPlan, parser/validator, config và test liên quan.
- Checkpoint có `config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors` và `LICENSE`; tổng kích thước các file thường trong thư mục là **999,604,126 bytes**, weights là **988,097,824 bytes**.
- Cấu hình checkpoint: `model_type=qwen2`, `Qwen2ForCausalLM`, 24 layers, 14 attention heads và 2 KV heads. Khi load, PyTorch đếm **494,032,768 tham số**.
- Đã kiểm tra file license cục bộ và model card: **Apache License 2.0**. Model card ghi 0.49B tham số; lịch sử Hugging Face có initial commit ngày **2024-09-16** và bài công bố Qwen2.5 ngày **2024-09-19**.

Nguồn: [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Hugging Face commit history](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/commits/main), [Qwen2.5 announcement](https://qwenlm.github.io/blog/qwen2.5/).

## 3. Thay đổi

**Đã sửa**

- `src/datamate/local_inference.py`: thêm lựa chọn `device=cpu|cuda` tường minh; CPU yêu cầu FP32. Không tự chuyển thiết bị khi CUDA lỗi/thiếu VRAM; lỗi CUDA OOM hướng dẫn chọn CPU. Kiểm tra BF16 loại trừ emulation; đo thời gian load/generate, RSS tiến trình, token/s và peak CUDA allocation/reservation.
- `src/datamate/question_understanding.py`: bổ sung hợp đồng kiểu/operation QueryPlan trong prompt; nếu parse/validation thất bại, thử sinh lại tối đa một lần với hướng dẫn sửa tổng quát. Nếu lần hai vẫn lỗi, ném lỗi để pipeline ghi log và dừng thực thi. Log có preview tối đa 1.000 ký tự từ output model.
- `src/datamate/plan_serialization.py`: chấp nhận code fence chỉ khi toàn bộ output là một JSON fence độc lập; không trích JSON từ prose hoặc output có nội dung thừa. Chuyển nesting quá sâu thành lỗi định dạng rõ ràng. Cho phép `base_table: null` chỉ đối với loại abstain/unanswerable theo hành vi abstention sẵn có.
- `src/datamate/query_plan.py`: type `base_table` phản ánh trường hợp null của abstention và validator từ chối base table thiếu cho plan thực thi; không thêm operation hay trường JSON mới.
- `scripts/run_inference.py`: đọc/kiểm tra device và chính sách local-only; CLI summary ghi số câu trả lời, QueryPlan hợp lệ, abstention do lỗi model và metrics runtime.
- `README.md`, `MODEL_SETUP.md`: cập nhật trạng thái CUDA/CPU, cấu hình Qwen và link báo cáo.

**Đã thêm**

- `config/inference.qwen.local.example.json`: đường dẫn model được đặt trong config, không hardcode trong logic; `device=cuda`, `precision=fp16`, prompt tối đa 2.048 tokens, output tối đa 512 tokens, `local_files_only=true`, `trust_remote_code=false`.
- Docker development: `Dockerfile.dev`, `compose.dev.yaml`, `.dockerignore`, `.env.docker.example`, `DOCKER.md` và `config/inference.docker.example.json`. Đây là cấu hình chạy local, không phải image submission đã xác nhận tuân thủ.
- `planning/21_LOCAL_MODEL_INTEGRATION_REPORT.md`: báo cáo này.
- Test cho CPU được chọn tường minh, BF16 không native, cấu hình local-only, invalid/missing/wrong-type/unsupported QueryPlan, retry có giới hạn, log safe-abstain, câu hỏi tiếng Việt nhiều điều kiện và thực thi trên SQLite fixture.

`local_files_only` chỉ chấp nhận `true`, `trust_remote_code` chỉ chấp nhận `false`; backend cũng truyền trực tiếp hai giá trị này cho Transformers. Inference đặt `HF_HUB_OFFLINE=1` và `TRANSFORMERS_OFFLINE=1`.

## 4. Runtime và cấu hình

Đã tạo `.venv` trong repository; không cài hoặc nâng/hạ package Python toàn cục. Global Python hiện có Torch CPU và Transformers 5, không được dùng cho inference vì không khớp dependency constraints.

| Thành phần trong `.venv` | Phiên bản đã kiểm tra |
|---|---:|
| Python | 3.12.6 |
| PyTorch | 2.5.1+cu124 |
| Transformers | 4.55.4 |
| Tokenizers | 0.21.4 |
| Hugging Face Hub | 0.36.2 |
| psutil | 7.2.2 |
| pytest | 8.4.2 |

`requirements-inference.txt` khai báo `torch>=2.5,<3`, `transformers>=4.46,<5`, `tokenizers>=0.20,<0.22`, `huggingface-hub>=0.26,<1` và `psutil>=5.9,<8`. `pip check` không báo dependency conflict. PyTorch CUDA được cài riêng trong `.venv` từ build cu124; cài package trước inference cần mạng, nhưng inference không lấy model hoặc gọi API.

Cấu hình mẫu Qwen dùng FP16 vì GTX 1060 không hỗ trợ BF16 native. Kiểm tra PyTorch cho thấy `is_bf16_supported()` mặc định trả true do emulation, trong khi `is_bf16_supported(including_emulation=False)` trả false; backend dùng kiểm tra native.

## 5. Kết quả load, inference và tài nguyên

Thiết bị là **NVIDIA GeForce GTX 1060 3GB**, driver 582.66, CUDA runtime của PyTorch 12.4, capability 6.1. Model được nạp FP16 từ thư mục local, không `trust_remote_code`, không gọi API và không tải checkpoint.

Smoke test dùng ba câu hỏi tiếng Việt tự tạo (đã xác nhận giữ nguyên dấu Unicode) trên fixture SQLite tạm: tổng doanh thu; doanh thu đơn đã giao trong tháng 1/2026; và tổng theo từng cửa hàng không có trong schema. Mỗi câu được sinh ban đầu và thử sửa một lần.

| Đo lường | Kết quả |
|---|---:|
| Model loading (tokenizer + weights) | 6.92 giây |
| Câu hỏi tự tạo | 3 |
| Lượt sinh model, gồm retry | 6 |
| Tokens đã sinh | 1,769 |
| Thời gian generate tổng | 42.48 giây |
| Thời gian generate trung bình/lượt | 7.08 giây |
| Tốc độ generate quan sát được | 41.64 tokens/giây |
| QueryPlan hợp lệ | **0/3** |
| Câu trả lời thành công | **0/3** |
| Abstain an toàn do output model lỗi | **3/3** |

| Bộ nhớ | Kết quả đo |
|---|---:|
| Tham số model | 494,032,768 |
| Weights + buffers trong PyTorch | 988,065,664 bytes |
| KV cache ước lượng với 2.048 + 512 tokens | 31,457,280 bytes |
| CUDA total/free trước transfer theo PyTorch | 3,221,094,400 / 2,579,759,104 bytes |
| Peak CUDA allocated/reserved theo PyTorch | 1,215,612,928 / 1,547,698,176 bytes |
| RSS tiến trình sau inference cuối | 1,091,883,008 bytes |

`RSS` là mẫu sau inference, không phải peak RAM. Peak CUDA trên là số của PyTorch; lần chạy cuối không ghi lại cặp snapshot `nvidia-smi` trước/sau. Trên Windows/WDDM, `nvidia-smi` có thể báo mức dùng tổng khác với `torch.cuda.mem_get_info()`. Model vẫn load và generate được trên lần chạy này, nhưng free VRAM thay đổi theo ứng dụng đang chạy; không suy rộng thành bảo đảm mọi lần chạy đều đủ VRAM.

Model thực tế sinh code fence, prose giải thích, tên bảng rút gọn và các object/field không thuộc schema (ví dụ `filters`/`sort` sai kiểu, aggregation sai dạng). Cả parser lẫn validator đều từ chối output; test không trích JSON từ phần văn bản và không đưa plan sai vào executor.

## 6. Kiểm thử

- `pytest -q tests`: **250 passed**.
- `python -m compileall -q src tests scripts`: PASS.
- `git diff --check`: PASS.
- Đọc config Qwen bằng `_read_inference_config` và xác nhận model path, FP16, device, local-only/trust-code flags: PASS.
- `scripts/run_inference.py --help`: PASS.
- `pip check`: PASS.
- Docker Compose config parse, Docker inference config parse và synthetic adapter plan với Qwen tokenizer trong giới hạn 2.048 tokens: PASS.
- Test SQLite fixture với QueryPlan hợp lệ do stub sinh: executor chạy, câu hỏi nhiều điều kiện trả kết quả fixture mong đợi. Đây là test plumbing, không phải kết quả model thật.
- Test model thật: load tokenizer/model PASS; sinh output PASS; parse/QueryPlan validation **FAIL có chủ đích và safe-abstain**; executor không chạy với plan sai.

Public Test đã được đọc theo yêu cầu chạy đánh giá; hai lượt đều dừng do prompt vượt giới hạn hoặc CUDA OOM, không có kết quả đánh giá. Không dùng Public Test/Private Test để huấn luyện, ghi nhớ ID/đáp án hoặc hardcode. Smoke questions và dữ liệu SQLite đều tự tạo.

## 7. License và điều lệ

Model card/file license nêu Apache-2.0; thông tin công bố năm 2024 và 494M tham số được xác nhận từ model card/config cùng lịch sử repo. 494M thấp hơn giới hạn 15B được ghi trong ghi chú yêu cầu nội bộ, nếu điều kiện đó áp dụng; đây **không phải xác nhận đủ điều kiện Zalo AI Challenge**. `planning/02_COMPETITION_REQUIREMENTS.md` ghi các điều kiện model/license cần đối chiếu với quy định chính thức; validator/portal hiện hành và cách áp dụng điều lệ cho checkpoint cụ thể chưa được xác minh trong lần tích hợp này. Vì vậy không tuyên bố model hợp lệ theo quy chế.

Không dùng API thương mại, không tải checkpoint trong inference, không hardcode business/question/answer production data và không sửa Query Executor/business rules để bù đầu ra model.

## 8. Hạn chế và bước cần thiết

1. Checkpoint đã chạy được nhưng **không đạt định dạng QueryPlan trong cả ba câu smoke**. Chưa dùng cho batch nghiệp vụ/submission; abstain an toàn là hành vi mong muốn khi output còn lỗi.
2. Retry một lần không sửa được cả code fence/prose lẫn cấu trúc sai. Không có cơ chế biến đổi output đoán mò hoặc tạo QueryPlan thay model.
3. Ba câu tự tạo chỉ xác nhận lỗi rõ ràng, không đủ để đo accuracy. Cần tập validation được phép sử dụng, viết tay/không lấy từ Public/Private Test, trước mọi kết luận chất lượng.
4. Chưa benchmark CPU thật vì CUDA load/generate được; đường CPU FP32 được kiểm tra bằng unit test và phải chọn tường minh.
5. Cần tiếp tục xác minh điều lệ chính thức, revision/hash checkpoint và hiệu năng ổn định trong môi trường chạy đích trước khi quyết định deploy.

## 9. Tối ưu prompt và Docker (cập nhật 2026-11-01)

- `JsonPlanAdapter` dùng chính tokenizer/chat template của checkpoint để đo prompt trước khi gọi model, dành trước 96 tokens cho lượt sửa lỗi. Khi vượt giới hạn cấu hình, prompt ưu tiên giữ nguyên câu hỏi, schema đầy đủ và business rules; sau đó giảm calendar theo mức ưu tiên, rồi trích đoạn rulebook/README theo giới hạn token. Mọi nội dung bị lược đều được đánh dấu; không bao giờ cắt schema hoặc business rules, và nếu riêng ngữ cảnh bắt buộc vẫn vượt giới hạn thì dừng câu hỏi an toàn thay vì gây CUDA OOM.
- Cấu hình synthetic test với Qwen tokenizer đạt 1.951/1.952 tokens trong ngữ cảnh lớn; giữ nguyên schema và business rule, giữ 12 dòng calendar ưu tiên, trích rulebook liên quan và đánh dấu README bị lược. Đây là phép đo context packing trên dữ liệu tổng hợp, không phải đánh giá độ chính xác model hay chạy Public Test.
- Tạo `Dockerfile.dev`, `compose.dev.yaml`, `.dockerignore`, `.env.docker.example`, `config/inference.docker.example.json` và `DOCKER.md`. Container development cấm network, mount input/model read-only, ghi `/result/submission.jsonl`, chạy CPU FP32 và không chứa dữ liệu test/checkpoint trong build context.
- Đã truy cập endpoint config-detail chính thức của DataMate (`title=2`). Rule §Technical requirements xác nhận: chạy offline/network disabled; `/data` read-only; output ở `/result/submission.jsonl`; model được phát hành trước 01/10/2026, license phù hợp; tổng tham số mọi model inference ≤15B; 2× RTX 5090 32GB, 512GB RAM; private test chạy tối đa 6 giờ; CUDA ≥12.8 cho Blackwell; reproducibility `|PA(uploaded)-PA(re-run)| <= 0.03`. Public submission guidance yêu cầu đóng gói code/data vào Docker và checksum/link Google Drive.
- Quy chế chính thức dẫn tới Docker guideline riêng quy định base image, nhưng nội dung guideline đó chưa được trả về từ portal/API được truy cập. Vì vậy cấu hình vừa thêm chỉ là local development, không tuyên bố là image nộp cuộc thi; cần lấy đúng base image/runtime và đóng checkpoint vào image trước submission.
- Docker Desktop 4.97.0 đã cài tại `F:\DockerDesktop\App`; Docker CLI 29.9.2 và Compose 5.6.1 chạy. `compose.dev.yaml config --quiet` PASS. Docker Engine chưa kiểm thử/build được do Windows vừa bật `VirtualMachinePlatform` và `Microsoft-Hyper-V`, đang chờ reboot.
- `docker info` chưa kết nối được Docker Engine (API trả HTTP 500), phù hợp trạng thái Windows đang chờ reboot. Build/run container chưa được kiểm chứng.
- Public DataMate rule config (`https://api-challenge.zalo.ai/fe/api/config-detail?name=rule&title=2`) xác nhận `/data` read-only và `/result/submission.jsonl`; endpoint submission guidance yêu cầu đóng gói code/data thành Docker và checksum/link Google Drive. Nội dung Docker guideline/base image được nhắc tới nhưng không có trong payload công khai này.
- Regression cuối: **250 passed**; `compileall`, `pip check`, `git diff --check`, Docker Compose parse và inference config parse PASS. Docker image build/run chưa thể kiểm tra vì Docker Engine chưa khởi động, chờ reboot Windows.
