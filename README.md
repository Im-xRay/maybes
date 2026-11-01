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
