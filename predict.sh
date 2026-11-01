#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# XRAY Autonomous Agent - Zalo AI Challenge 2026 predict.sh
# 100% Offline execution on 2x NVIDIA RTX 5090 (Blackwell)
# Inputs:  /data/questions.jsonl, /data/rulebook.md, /data/calendar.csv, /data/data/
# Outputs: /result/submission.jsonl, /result/logs/
# ==============================================================================

echo "=== [XRAY AGENT] INITIALIZING DATAMATE INFERENCE PIPELINE ==="

# Define paths
DATA_DIR="${DATA_DIR:-/data}"
RESULT_DIR="${RESULT_DIR:-/result}"
LOG_DIR="${RESULT_DIR}/logs"
MODEL_DIR="${MODEL_DIR:-/models/qwen2.5-coder-14b-instruct}"

INPUT_FILE="${DATA_DIR}/questions.jsonl"
OUTPUT_FILE="${RESULT_DIR}/submission.jsonl"
BUSINESS_DATA_ROOT="${DATA_DIR}/data"
RULEBOOK_PATH="${DATA_DIR}/rulebook.md"
CALENDAR_PATH="${DATA_DIR}/calendar.csv"

# Fallback for local testing if /data does not exist
if [ ! -d "${DATA_DIR}" ]; then
    echo "Warning: /data directory not found, using local fallback..."
    DATA_DIR="./public_test"
    INPUT_FILE="${DATA_DIR}/questions.jsonl"
    BUSINESS_DATA_ROOT="${DATA_DIR}/data"
    RULEBOOK_PATH="${DATA_DIR}/rulebook.md"
    CALENDAR_PATH="${DATA_DIR}/calendar.csv"
    RESULT_DIR="./result"
    LOG_DIR="${RESULT_DIR}/logs"
    OUTPUT_FILE="${RESULT_DIR}/submission.jsonl"
fi

mkdir -p "${RESULT_DIR}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TP_SIZE=2

echo "Running batch inference..."
if python -c "import XRAY" 2>/dev/null; then
    python -m XRAY.src.inference.agent \
        --input "${INPUT_FILE}" \
        --output "${OUTPUT_FILE}" \
        --log-dir "${LOG_DIR}" \
        --data-root "${BUSINESS_DATA_ROOT}" \
        --rulebook "${RULEBOOK_PATH}" \
        --calendar "${CALENDAR_PATH}" \
        --model-path "${MODEL_DIR}"
else
    python -m src.inference.agent \
        --input "${INPUT_FILE}" \
        --output "${OUTPUT_FILE}" \
        --log-dir "${LOG_DIR}" \
        --data-root "${BUSINESS_DATA_ROOT}" \
        --rulebook "${RULEBOOK_PATH}" \
        --calendar "${CALENDAR_PATH}" \
        --model-path "${MODEL_DIR}"
fi

echo "=== [XRAY AGENT] INFERENCE COMPLETED SUCCESSFULLY ==="
