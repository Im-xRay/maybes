# maybes - XRAY: DataMate Autonomous Agent (Zalo AI Challenge 2026)
**IOAI Gold Medal & Zalo AI Champion Architecture**

XRAY is a high-performance, deterministic tabular question-answering agent specifically designed for real-world Vietnamese business data (`.csv`, `.xlsx`, `.sqlite`) under the official constraints of Zalo AI Challenge 2026:
- **Total Model Budget:** $\le 15$ Billion parameters at inference time.
- **Metric:** Penalized Accuracy (PA = `Correct - Wrong`).
- **Dynamic Rules:** Zero hardcoding of train split rules. Fully dynamic parsing of `rulebook.md`, `calendar.csv` (Lunar/Solar holidays), and `as_of_date`.
- **Target Hardware:** 2x NVIDIA RTX 5090 (Blackwell, 64GB VRAM, 512GB RAM, CUDA $\ge 12.8$).

---

## Architecture Overview

```text
                  Input: (question, business_id, as_of_date)
                                      │
                                      ▼
         ┌────────────────────────────────────────────────────────┐
         │       XRAY Universal Normalizer & Schema Profiler      │
         │  (In-Memory DuckDB + Auto Header Offset + RapidFuzz)   │
         └────────────────────────────┬───────────────────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
[Fast-Gate: Unanswerable Pre-check]                    [Schema & Prompt Assembly]
(Subjective / Missing Column / Out of Bounds)          (DuckDB Schema + Lunar Bounds)
            │                                                   │
            ▼                                                   ▼
  Output: unanswerable                                Qwen2.5-Coder-14B-Instruct
                                                      (vLLM PagedAttention v2)
                                                                │
                                                                ▼
                                                      [Python/DuckDB Snippet]
                                                                │
                                                                ▼
                                                      [Execution Sandbox K=2]
                                                       (Timeout & Auto-Traceback)
                                                                │
                                       ┌────────────────────────┴────────────────────────┐
                                       ▼                                                 ▼
                                [Execution Success]                              [Execution Error]
                                       │                                                 │
                                       ▼                                                 ▼
                             [Sanity & Format Gate]                               [ABSTAIN GATE]
                             (Round to VND / 2-dec %)                           (Output: abstain)
                                       │
                                       ▼
                                [Final Answer]
```

---

## 5 Core Modules

1. **Module 1 (`XRAY/src/eval/evaluator.py`):**
   - Exact implementation of Rulebook §9 scoring: Correct = +1, Wrong = -1, Abstain = 0.
   - Comprehensive alias matching (product codes, alternative brand names).
   - Bipartite matching for table predictions.

2. **Module 2 (`XRAY/src/data/normalizer.py`):**
   - High-throughput ingestion of CSV, multi-sheet XLSX (with automatic header offset detection), and SQLite directly into DuckDB.
   - Automatic cleaning of messy Vietnamese numbers (`560,000`, `79k`, `569.000đ`, `1,2tr`) and dates (`1 thg 6 2025`, `15/08/2026`).
   - RapidFuzz entity indexer for products and customers.

3. **Module 3 (`XRAY/src/train/generate_trajectories.py`):**
   - Rejection Sampling / Self-Taught Reasoner (STaR) pipeline.
   - Converts `(question, answer)` pairs into verified `(Prompt, CoT + Code + Answer)` trajectories.
   - Generates SFT dataset (`trajectories_sft.jsonl`) and DPO preference pairs (`trajectories_dpo.jsonl`).

4. **Module 4 (`XRAY/src/train/train_sft_dpo.py`):**
   - Stage 1: Supervised Fine-Tuning (SFT) with LoRA on Qwen2.5-Coder.
   - Stage 2: Direct Preference Optimization (DPO) to suppress hallucinations and align with Penalized Accuracy.
   - Standalone LoRA adapter merge script for offline packaging.

5. **Module 5 (`XRAY/src/inference/agent.py` & `XRAY/predict.sh`):**
   - Batch inference pipeline optimized for 2x RTX 5090 using vLLM.
   - Multi-stage Abstain Gate: avoids guessing to protect PA score.
   - Full competition logging into `/result/logs/` and `/result/submission.jsonl`.


---

## 🛠️ Hướng Dẫn Vận Hành: Huấn Luyện (Train) & Kiểm Thử (Test)

Mọi lệnh dưới đây có thể chạy trực tiếp từ thư mục gốc của repository (`XRAY/`).

### 1. Huấn Luyện Mô Hình (Training Pipeline)

Quy trình huấn luyện gồm 4 bước khép kín từ sinh dữ liệu đến đóng gói trọng số offline:

#### Bước 1: Tự sinh dữ liệu suy luận (STaR Trajectories)
Dùng Rejection Sampling để trích xuất các đường suy luận đúng (Prompt + CoT + Code + Output) từ 3,000 câu của tập `train`:
```bash
python -m src.train.generate_trajectories \
    --questions ../train/questions.jsonl \
    --answers ../train/answers.jsonl \
    --data-root ../train/data \
    --calendar ../train/calendar.csv \
    --output-sft data/trajectories_sft.jsonl \
    --output-dpo data/trajectories_dpo.jsonl
```

#### Bước 2: Huấn luyện có giám sát (Stage 1: SFT với LoRA)
Dạy LLM (`Qwen2.5-Coder-14B-Instruct`) học cách đọc schema DuckDB và viết Python/SQL chuẩn:
```bash
python -m src.train.train_sft_dpo \
    --stage sft \
    --model-name Qwen/Qwen2.5-Coder-14B-Instruct \
    --train-file data/trajectories_sft.jsonl \
    --output-dir checkpoints/qwen_sft
```

#### Bước 3: Căn chỉnh triệt tiêu ảo giác (Stage 2: DPO Alignment)
Tối ưu trực tiếp theo luật Penalized Accuracy, phạt nặng các câu sinh sai kết quả:
```bash
python -m src.train.train_sft_dpo \
    --stage dpo \
    --model-name checkpoints/qwen_sft \
    --train-file data/trajectories_dpo.jsonl \
    --output-dir checkpoints/qwen_dpo
```

#### Bước 4: Hợp nhất trọng số (Merge LoRA) phục vụ Offline Inference
```bash
python -m src.train.train_sft_dpo \
    --stage merge \
    --base-model Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter-path checkpoints/qwen_dpo \
    --output-dir models/xray_final_14b
```

---

### 2. Kiểm Thử & Chạy Đề Thi (Test & Inference Pipeline)

#### Trường hợp A: Tự kiểm thử & Benchmark trên tập `train` (Có Ground-Truth)
Chạy suy luận và tính điểm Penalized Accuracy thực tế (+1 Correct, -1 Wrong, 0 Abstain):
```bash
# 1. Chạy suy luận trên tập train
python -m src.inference.agent \
    --input ../train/questions.jsonl \
    --output result/train_pred.jsonl \
    --log-dir result/logs \
    --data-root ../train/data \
    --rulebook ../train/rulebook.md \
    --calendar ../train/calendar.csv

# 2. Đánh giá chi tiết điểm PA và ma trận phân loại lỗi
python -m src.eval.evaluator \
    --submission result/train_pred.jsonl \
    --ground-truth ../train/answers.jsonl \
    --questions ../train/questions.jsonl
```

#### Trường hợp B: Chạy dự đoán trên tập `public_test` (Tạo file nộp bài)
Sinh file nộp bài `result/submission.jsonl` chuẩn 100% định dạng BTC:
```bash
python -m src.inference.agent \
    --input ../public_test/questions.jsonl \
    --output result/submission.jsonl \
    --log-dir result/logs \
    --data-root ../public_test/data \
    --rulebook ../public_test/rulebook.md \
    --calendar ../public_test/calendar.csv
```

#### Trường hợp C: Chạy bằng Bash Script tiêu chuẩn của BTC
```bash
bash predict.sh
```
Script sẽ tự động đọc câu hỏi từ `/data/questions.jsonl`, quét dữ liệu tại `/data/data/`, xử lý theo `rulebook.md` và xuất ra `/result/submission.jsonl` cùng log chi tiết tại `/result/logs/`.
