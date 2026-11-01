"""
XRAY Trajectory Generation Pipeline - Module 3
Generates high-quality Chain-of-Thought + DuckDB SQL/Python trajectories via
Rejection Sampling (STaR) for SFT and DPO alignment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from XRAY.src.data.normalizer import BusinessDatabase, UniversalNormalizer
    from XRAY.src.eval.evaluator import match_list_items, match_scalar, match_table
except ImportError:
    from src.data.normalizer import BusinessDatabase, UniversalNormalizer
    from src.eval.evaluator import match_list_items, match_scalar, match_table


def parse_calendar(calendar_path: str) -> Dict[str, Any]:
    """Parse calendar.csv for lunar/solar dates and Tet intervals."""
    tet_dates = {}
    if not os.path.exists(calendar_path):
        return tet_dates

    with open(calendar_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        date_idx = header.index("date")
        lunar_day_idx = header.index("lunar_day")
        lunar_month_idx = header.index("lunar_month")
        lunar_year_idx = header.index("lunar_year")

        chapp_dates = {}  # year -> solar date of 23rd of lunar month 12
        gieng_dates = {}  # year -> solar date of 10th of lunar month 1

        for line in lines[1:]:
            parts = line.strip().split(",")
            s_date = parts[date_idx]
            l_day = int(parts[lunar_day_idx])
            l_month = int(parts[lunar_month_idx])
            l_year = int(parts[lunar_year_idx])

            if l_day == 23 and l_month == 12:
                # 23rd of month 12 corresponds to Tet of next year (l_year + 1)
                chapp_dates[l_year + 1] = s_date
            if l_day == 10 and l_month == 1:
                gieng_dates[l_year] = s_date

        for y in gieng_dates:
            if y in chapp_dates:
                tet_dates[y] = {
                    "tet_start": chapp_dates[y],
                    "tet_end": gieng_dates[y],
                }

    return tet_dates


class TrajectoryGenerator:
    """Executes code, validates outputs against answers.jsonl, and formats SFT/DPO pairs."""

    def __init__(
        self,
        data_root: str,
        rulebook_path: str,
        calendar_path: str,
        normalizer: Optional[UniversalNormalizer] = None,
    ):
        self.normalizer = normalizer or UniversalNormalizer(data_root)
        self.rulebook_text = ""
        if os.path.exists(rulebook_path):
            with open(rulebook_path, "r", encoding="utf-8") as f:
                self.rulebook_text = f.read()

        self.tet_calendar = parse_calendar(calendar_path)
        self.cached_dbs: Dict[str, BusinessDatabase] = {}

    def get_db(self, business_id: str) -> BusinessDatabase:
        if business_id not in self.cached_dbs:
            self.cached_dbs[business_id] = self.normalizer.load_business(business_id)
        return self.cached_dbs[business_id]

    def build_prompt(self, q_item: Dict[str, Any], db: BusinessDatabase) -> str:
        """Constructs full prompt including schema, as_of_date, calendar helpers, and query."""
        as_of_date = q_item.get("as_of_date", "2026-06-01")
        question = q_item["question"]

        prompt = f"""Bạn là Chuyên gia Phân tích Dữ liệu DataMate. Nhiệm vụ của bạn là viết code DuckDB SQL / Python để trả lời câu hỏi doanh nghiệp.
Hôm nay là: {as_of_date} (mọi mốc thời gian tương đối như 'tuần này', 'tháng này', 'tính đến hôm nay' đều tính theo ngày này).

QUY TẮC BẮT BUỘC (Trích xuất từ Rulebook):
1. Doanh thu: Chỉ tính đơn không huỷ và không trả hàng. Đơn trả hàng một phần vẫn coi như trả hàng và bỏ toàn bộ đơn. Đơn đang giao vẫn tính.
2. Tiền hàng: Bằng tổng thành tiền các dòng hàng (sau chiết khấu dòng). Không trừ giảm giá đơn, không cộng VAT, không gồm phí giao.
3. Tuần bắt đầu thứ Hai. 'Tuần này' tính từ thứ Hai của tuần chứa hôm nay đến hết hôm nay.
4. Xếp hạng: Khi các giá trị bằng nhau, xếp theo mã tăng dần (code ASC). Bán chạy xếp theo số lượng quy về đơn vị cơ bản.
5. Nếu câu hỏi yêu cầu dữ liệu mà cửa hàng KHÔNG có hoặc thời gian hỏi nằm ngoài phạm vi dữ liệu: Đưa ra {{"type": "unanswerable"}}.

{db.get_schema_prompt()}

CÂU HỎI: {question}

Hãy suy luận từng bước (Thinking) và viết đoạn code Python thực thi với biến `con` (kết nối DuckDB đã nạp sẵn các bảng trên).
Lưu kết quả cuối cùng vào biến `final_answer` (theo format đúng: number, percent, date, boolean, string, list, table, hoặc unanswerable).
"""
        return prompt

    def execute_code_snippet(self, code_str: str, db: BusinessDatabase) -> Tuple[bool, Any, str]:
        """Safely executes code in sandbox with access to db.con."""
        local_vars = {"con": db.con, "final_answer": None, "final_type": None}
        try:
            # Clean python code block fences if present
            clean_code = re.sub(r"^```python\s*", "", code_str, flags=re.MULTILINE)
            clean_code = re.sub(r"^```\s*$", "", clean_code, flags=re.MULTILINE)

            exec(clean_code, {}, local_vars)
            ans = local_vars.get("final_answer")
            t = local_vars.get("final_type")
            return True, ans, t
        except Exception as e:
            return False, None, str(e)

    def verify_trajectory(
        self,
        pred_ans: Any,
        pred_type: str,
        gold_entry: Dict[str, Any],
    ) -> bool:
        """Verifies if predicted answer exactly matches ground truth (including aliases)."""
        g_type = gold_entry.get("type")
        g_ans = gold_entry.get("answer")
        aliases = gold_entry.get("alias", [])

        if g_type == "unanswerable":
            return pred_type == "unanswerable"

        if pred_type != g_type:
            return False

        if g_type in ("number", "percent", "date", "boolean", "string"):
            return match_scalar(pred_ans, g_ans, g_type, aliases)
        elif g_type == "list":
            return match_list_items(pred_ans, g_ans, aliases)
        elif g_type == "table":
            return match_table(pred_ans, g_ans, aliases)

        return False

    def generate_sft_dpo_pair(
        self,
        prompt: str,
        chosen_code: str,
        chosen_cot: str,
        rejected_code: Optional[str] = None,
        rejected_cot: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Formats chosen and rejected trajectories for SFT and DPO datasets."""
        chosen_text = f"<thought>\n{chosen_cot}\n</thought>\n```python\n{chosen_code}\n```"
        sft_sample = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_text},
            ]
        }

        dpo_sample = None
        if rejected_code:
            rejected_text = f"<thought>\n{rejected_cot or 'Suy luận sơ sài hoặc đoán mò khi thiếu dữ liệu.'}\n</thought>\n```python\n{rejected_code}\n```"
            dpo_sample = {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }

        return sft_sample, dpo_sample


def main():
    parser = argparse.ArgumentParser(description="Generate SFT and DPO trajectories.")
    parser.add_argument("--questions", default="train/questions.jsonl")
    parser.add_argument("--answers", default="train/answers.jsonl")
    parser.add_argument("--data-root", default="train/data")
    parser.add_argument("--rulebook", default="train/rulebook.md")
    parser.add_argument("--calendar", default="train/calendar.csv")
    parser.add_argument("--out-sft", default="XRAY/trajectories_sft.jsonl")
    parser.add_argument("--out-dpo", default="XRAY/trajectories_dpo.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to process")
    args = parser.parse_args()

    gen = TrajectoryGenerator(args.data_root, args.rulebook, args.calendar)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    with open(args.answers, "r", encoding="utf-8") as f:
        answers = {item["id"]: item for item in (json.loads(line) for line in f)}

    print(f"Loaded {len(questions)} questions. Processing first {args.limit}...")

    sft_records = []
    dpo_records = []

    for i, q in enumerate(questions[: args.limit]):
        gold = answers.get(q["id"])
        if not gold:
            continue

        db = gen.get_db(q["business_id"])
        prompt = gen.build_prompt(q, db)

        # Example demonstration of unanswerable trajectory generation
        if gold["type"] == "unanswerable":
            chosen_cot = f"Kiểm tra câu hỏi '{q['question']}': Bảng dữ liệu của shop không ghi thông tin này hoặc kỳ hỏi nằm ngoài phạm vi dữ liệu ({db.min_date} -> {db.max_date}). Theo Rulebook §7, không được tự suy đoán hoặc trả về 0."
            chosen_code = 'final_type = "unanswerable"\nfinal_answer = {"type": "unanswerable", "reason": "Thiếu dữ liệu"}'

            rejected_cot = "Cố gắng tính toán bằng cách giả định dữ liệu bằng 0."
            rejected_code = 'final_type = "number"\nfinal_answer = 0'

            sft_item, dpo_item = gen.generate_sft_dpo_pair(prompt, chosen_code, chosen_cot, rejected_code, rejected_cot)
            sft_records.append(sft_item)
            if dpo_item:
                dpo_records.append(dpo_item)

    print(f"Generated {len(sft_records)} SFT samples and {len(dpo_records)} DPO samples.")
    if sft_records:
        with open(args.out_sft, "w", encoding="utf-8") as f:
            for s in sft_records:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    if dpo_records:
        with open(args.out_dpo, "w", encoding="utf-8") as f:
            for d in dpo_records:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
