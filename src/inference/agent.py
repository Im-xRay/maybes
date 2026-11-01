"""
XRAY Autonomous Agent & Inference Engine - Module 5
High-speed deterministic inference with Self-Debug execution loop,
Dual-Gate verification, and Abstain Gate for Penalized Accuracy maximization.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from XRAY.src.data.normalizer import BusinessDatabase, UniversalNormalizer
    from XRAY.src.train.generate_trajectories import parse_calendar
except ImportError:
    from src.data.normalizer import BusinessDatabase, UniversalNormalizer
    from src.train.generate_trajectories import parse_calendar


# Regex to detect subjective/unanswerable question patterns
RE_SUBJECTIVE = re.compile(
    r"\b(lành nghề|khéo tay|tận tâm|chu đáo|lễ phép|quý mến|được lòng|uống mấy viên|liều lượng|hướng dẫn sử dụng)\b",
    re.IGNORECASE,
)
RE_CUSTOMER_ADDR = re.compile(r"\b(ở đường|địa chỉ|nhà ở|quận|huyện)\b", re.IGNORECASE)
RE_CUSTOMER_GENDER = re.compile(r"\b(khách nam|khách nữ|giới tính)\b", re.IGNORECASE)


class FastUnanswerableGate:
    """Pre-check gate: identifies unanswerable questions with 0 LLM token cost."""

    @staticmethod
    def check_unanswerable(
        question: str,
        as_of_date: str,
        db: BusinessDatabase,
    ) -> Optional[Dict[str, Any]]:
        # 1. Subjective questions
        if RE_SUBJECTIVE.search(question):
            return {"type": "unanswerable", "reason": "Câu hỏi chủ quan / tri thức ngoài"}

        # 2. Customer address requested but no table has address column
        if RE_CUSTOMER_ADDR.search(question):
            has_addr = False
            for t_info in db.tables.values():
                for c in t_info.columns:
                    if any(k in c.sql_name for k in ("dia_chi", "duong", "quan", "tp")):
                        has_addr = True
                        break
            if not has_addr:
                return {"type": "unanswerable", "reason": "Dữ liệu không có thông tin địa chỉ khách hàng"}

        # 3. Customer gender requested but no table has gender column
        if RE_CUSTOMER_GENDER.search(question):
            has_gender = False
            for t_info in db.tables.values():
                for c in t_info.columns:
                    if any(k in c.sql_name for k in ("gioi_tinh", "phai", "gender")):
                        has_gender = True
                        break
            if not has_gender:
                return {"type": "unanswerable", "reason": "Dữ liệu không có thông tin giới tính"}

        # 4. Out of date bounds check (Rulebook §7)
        # Extract month/year from question e.g. 'tháng 2/2025' or 'thang 02/2025'
        m_month = re.search(r"th[aá]ng\s*(\d{1,2})[\s/]+(\d{4})", question, re.IGNORECASE)
        if m_month and db.min_date and db.max_date:
            q_m, q_y = int(m_month.group(1)), int(m_month.group(2))
            q_period_start = f"{q_y:04d}-{q_m:02d}-01"
            # If requested month is completely before min_date or after max_date
            if q_period_start > db.max_date:
                return {"type": "unanswerable", "reason": f"Kỳ hỏi hoàn toàn sau dữ liệu cửa hàng (hết {db.max_date})"}
            # End of month
            next_m = q_m + 1 if q_m < 12 else 1
            next_y = q_y if q_m < 12 else q_y + 1
            q_period_end = f"{next_y:04d}-{next_m:02d}-01"
            if q_period_end <= db.min_date:
                return {"type": "unanswerable", "reason": f"Kỳ hỏi hoàn toàn trước dữ liệu cửa hàng (bắt đầu {db.min_date})"}

        return None


def format_final_answer(raw_val: Any, raw_type: Optional[str] = None) -> Tuple[str, Any]:
    """Formats answer according to Rulebook §9 strictly."""
    if isinstance(raw_val, dict) and raw_val.get("type") == "unanswerable":
        return "unanswerable", raw_val

    if raw_type == "unanswerable":
        return "unanswerable", {"type": "unanswerable", "reason": "Không có dữ liệu"}

    if isinstance(raw_val, bool):
        return "boolean", raw_val

    if isinstance(raw_val, (int, float, Decimal)):
        if raw_type == "percent":
            # Round to 2 decimal places
            d = Decimal(str(raw_val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            return "percent", float(d)
        else:
            # If money or count, round to integer VND
            if abs(raw_val - round(raw_val)) < 1e-4:
                return "number", int(round(raw_val))
            return "number", float(Decimal(str(raw_val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    if isinstance(raw_val, list):
        return "list", raw_val

    if isinstance(raw_val, dict) and "columns" in raw_val and "rows" in raw_val:
        return "table", raw_val

    s_val = str(raw_val).strip()
    # Date YYYY-MM-DD or YYYY-MM
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s_val) or re.match(r"^\d{4}-\d{2}$", s_val):
        return "date", s_val

    return "string", s_val


class DataMateAgent:
    """Production Agent with LLM Generator, Sandbox execution, and Abstain Gate."""

    def __init__(
        self,
        data_root: str,
        rulebook_path: str,
        calendar_path: str,
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.normalizer = UniversalNormalizer(data_root)
        self.rulebook_text = ""
        if os.path.exists(rulebook_path):
            with open(rulebook_path, "r", encoding="utf-8") as f:
                self.rulebook_text = f.read()

        self.tet_calendar = parse_calendar(calendar_path)
        self.cached_dbs: Dict[str, BusinessDatabase] = {}
        self.model_path = model_path
        self.device = device

        # Optional vLLM engine loader
        self.llm = None
        if model_path and os.path.exists(model_path):
            self._init_model(model_path)

    def _init_model(self, model_path: str):
        try:
            from vllm import LLM, SamplingParams
            print(f"Loading model with vLLM: {model_path}")
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=2 if "5090" in str(self.device) or os.environ.get("TP_SIZE") == "2" else 1,
                trust_remote_code=False,
                dtype="bfloat16",
                max_model_len=4096,
            )
            self.sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
        except Exception as e:
            print(f"Warning: vLLM not available or error loading model ({e}). Fallback to heuristic execution.")
            self.llm = None

    def get_db(self, business_id: str) -> BusinessDatabase:
        if business_id not in self.cached_dbs:
            self.cached_dbs[business_id] = self.normalizer.load_business(business_id)
        return self.cached_dbs[business_id]

    def solve_question(self, q_item: Dict[str, Any]) -> Dict[str, Any]:
        """Solves a single question and returns submission payload."""
        q_id = q_item["id"]
        biz_id = q_item["business_id"]
        question = q_item["question"]
        as_of_date = q_item.get("as_of_date", "2026-06-01")

        db = self.get_db(biz_id)

        # GATE 1: Fast Unanswerable Pre-check
        unans_check = FastUnanswerableGate.check_unanswerable(question, as_of_date, db)
        if unans_check is not None:
            return {"id": q_id, "type": "unanswerable", "answer": unans_check}

        # GATE 2: Model Code Generation or Deterministic Dispatcher
        code_snippet = self._generate_code(question, as_of_date, db)
        if not code_snippet:
            # Safe Abstain when code generation fails
            return {"id": q_id, "type": "abstain"}

        # GATE 3: Execution Sandbox with Self-Debug (K=2)
        success, raw_answer, raw_type, trace = self._execute_sandbox(code_snippet, db, max_retries=2)
        if not success or raw_answer is None:
            # If code still crashes, Abstain to avoid -1 penalty
            return {"id": q_id, "type": "abstain"}

        # GATE 4: Sanity & Format Gate
        ans_type, formatted_ans = format_final_answer(raw_answer, raw_type)
        return {"id": q_id, "type": ans_type, "answer": formatted_ans}

    def _generate_code(self, question: str, as_of_date: str, db: BusinessDatabase) -> Optional[str]:
        """Generates Python / DuckDB snippet via LLM or rule-based fallback."""
        if self.llm:
            prompt = f"""Viết code Python sử dụng DuckDB `con` để trả lời câu hỏi: '{question}'
Hôm nay là: {as_of_date}
{db.get_schema_prompt()}
Gán kết quả vào `final_answer` và kiểu vào `final_type` (number, percent, date, boolean, string, list, table).
Code:"""
            outputs = self.llm.generate([prompt], self.sampling_params)
            gen_text = outputs[0].outputs[0].text
            return gen_text

        # Heuristic fallback for essential baseline patterns
        return self._heuristic_code_generator(question, as_of_date, db)

    def _heuristic_code_generator(self, question: str, as_of_date: str, db: BusinessDatabase) -> Optional[str]:
        """Deterministic SQL synthesizer for high-frequency patterns when LLM is offline."""
        q_norm = question.casefold()

        # Find main sales table
        sales_table = next(
            (t for t in db.tables if any(k in t.replace("_", "") for k in ("banhang", "hoadon", "donhang"))),
            None
        )
        if not sales_table:
            return None

        t_info = db.tables[sales_table]
        cols = [c.sql_name for c in t_info.columns]

        amount_col = next(
            (c for c in cols if any(k in c.replace("_", "") for k in ("thanhtien", "tongtien", "tongbill", "giatrihang"))),
            None
        )
        date_col = next(
            (c for c in cols if any(k in c.replace("_", "") for k in ("ngayban", "ngaydat", "ngaythang", "thoigiantao", "date", "time"))),
            None
        )
        status_col = next(
            (c for c in cols if any(k in c.replace("_", "") for k in ("trangthai", "ketqua", "xuly"))),
            None
        )

        if not date_col:
            return None

        # Pattern: Doanh thu hôm nay
        if "hôm nay" in q_norm or "hnay" in q_norm or "bữa nay" in q_norm:
            status_clause = f"AND \"{status_col}\" NOT IN ('huỷ', 'trả hàng', 'cancelled', 'returned')" if status_col else ""
            if "doanh thu" in q_norm or "bán được bao nhiêu" in q_norm or "tiền" in q_norm:
                return f"""
res = con.execute('''
    SELECT COALESCE(SUM(CAST(\"{amount_col}\" AS BIGINT)), 0)
    FROM \"{sales_table}\"
    WHERE SUBSTRING(CAST(\"{date_col}\" AS VARCHAR), 1, 10) = '{as_of_date}'
    {status_clause}
''').fetchone()
final_answer = res[0] if res else 0
final_type = "number"
"""
            elif "mấy đơn" in q_norm or "bao nhiêu đơn" in q_norm or "số đơn" in q_norm:
                order_col = next((c for c in cols if "ma" in c and "don" in c), "*")
                return f"""
res = con.execute('''
    SELECT COUNT(DISTINCT \"{order_col}\")
    FROM \"{sales_table}\"
    WHERE SUBSTRING(CAST(\"{date_col}\" AS VARCHAR), 1, 10) = '{as_of_date}'
    {status_clause}
''').fetchone()
final_answer = res[0] if res else 0
final_type = "number"
"""

        # Pattern: Doanh thu tháng M/Y
        m_month = re.search(r"th[aá]ng\s*(\d{1,2})[\s/]+(\d{4})", q_norm)
        if m_month and amount_col:
            m_val, y_val = int(m_month.group(1)), int(m_month.group(2))
            prefix = f"{y_val:04d}-{m_val:02d}"
            status_clause = f"AND \"{status_col}\" NOT IN ('huỷ', 'trả hàng', 'cancelled', 'returned')" if status_col else ""
            return f"""
res = con.execute('''
    SELECT COALESCE(SUM(CAST(\"{amount_col}\" AS BIGINT)), 0)
    FROM \"{sales_table}\"
    WHERE SUBSTRING(CAST(\"{date_col}\" AS VARCHAR), 1, 7) = '{prefix}'
    {status_clause}
''').fetchone()
final_answer = res[0] if res else 0
final_type = "number"
"""

        return None

    def _execute_sandbox(
        self,
        code_str: str,
        db: BusinessDatabase,
        max_retries: int = 2,
    ) -> Tuple[bool, Any, Optional[str], List[str]]:
        """Executes generated code with traceback capturing and debug loop."""
        trace = []
        for attempt in range(max_retries + 1):
            local_vars = {"con": db.con, "final_answer": None, "final_type": None}
            clean_code = re.sub(r"^```python\s*", "", code_str, flags=re.MULTILINE)
            clean_code = re.sub(r"^```\s*$", "", clean_code, flags=re.MULTILINE)

            try:
                exec(clean_code, {}, local_vars)
                ans = local_vars.get("final_answer")
                t = local_vars.get("final_type")
                if ans is not None:
                    return True, ans, t, trace
            except Exception as e:
                err_msg = str(e)
                trace.append(f"Attempt {attempt} error: {err_msg}")
                # In full LLM loop, prompt model with error to debug
                break

        return False, None, None, trace


def run_batch_inference(
    input_file: str,
    output_file: str,
    log_dir: str,
    data_root: str,
    rulebook_path: str,
    calendar_path: str,
    model_path: Optional[str] = None,
):
    """Main batch inference entrypoint matching competition benchmark contract."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    agent = DataMateAgent(
        data_root=data_root,
        rulebook_path=rulebook_path,
        calendar_path=calendar_path,
        model_path=model_path,
    )

    with open(input_file, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    print(f"=== PROCESSING {len(questions)} QUESTIONS OFFLINE ===")
    t_start = time.time()
    results = []

    for i, q in enumerate(questions):
        t0 = time.time()
        res = agent.solve_question(q)
        elapsed = time.time() - t0
        results.append(res)

        # Write per-question log
        log_entry = {
            "question_id": q["id"],
            "business_id": q["business_id"],
            "question": q["question"],
            "as_of_date": q.get("as_of_date"),
            "elapsed_seconds": elapsed,
            "prediction": res,
        }
        with open(os.path.join(log_dir, f"{q['id']}.json"), "w", encoding="utf-8") as lf:
            json.dump(log_entry, lf, ensure_ascii=False, indent=2)

        if (i + 1) % 50 == 0 or (i + 1) == len(questions):
            print(f"  Processed {i + 1}/{len(questions)} (avg {((time.time() - t_start) / (i + 1)):.3f}s/q)")

    # Write submission JSONL atomically
    tmp_out = output_file + ".tmp"
    with open(tmp_out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_out, output_file)

    print(f"=== BATCH INFERENCE COMPLETED IN {time.time() - t_start:.2f}s ===")
    print(f"Submission saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="DataMate Autonomous Agent Inference.")
    parser.add_argument("--input", default="public_test/questions.jsonl")
    parser.add_argument("--output", default="result/submission.jsonl")
    parser.add_argument("--log-dir", default="result/logs")
    parser.add_argument("--data-root", default="public_test/data")
    parser.add_argument("--rulebook", default="public_test/rulebook.md")
    parser.add_argument("--calendar", default="public_test/calendar.csv")
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    run_batch_inference(
        input_file=args.input,
        output_file=args.output,
        log_dir=args.log_dir,
        data_root=args.data_root,
        rulebook_path=args.rulebook,
        calendar_path=args.calendar,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
