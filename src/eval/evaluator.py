"""
XRAY Evaluation Engine - Module 1
Calculates exact Penalized Accuracy (PA) compliant with Zalo AI Challenge 2026 Rulebook §9.
Reward: Correct = +1, Wrong = -1, Abstain = 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union


def normalize_text(text: Any) -> str:
    """Normalize string: strip, lowercase, collapse whitespace, drop punctuation for soft string comparison."""
    if text is None:
        return ""
    s = str(text).strip().casefold()
    # Normalize internal whitespace
    s = " ".join(s.split())
    # Remove basic trailing punctuation like . , : ;
    s = s.rstrip(".,:;")
    return s


def is_close_num(val1: Union[int, float, Decimal], val2: Union[int, float, Decimal], tolerance: float = 1e-4) -> bool:
    """Compare numbers with floating point tolerance or exact integer equality."""
    try:
        f1 = float(val1)
        f2 = float(val2)
        if math.isnan(f1) or math.isnan(f2):
            return False
        if abs(f1 - f2) <= tolerance:
            return True
        # Relative tolerance for large numbers
        if max(abs(f1), abs(f2)) > 0:
            return abs(f1 - f2) / max(abs(f1), abs(f2)) <= tolerance
        return False
    except (ValueError, TypeError):
        return False


def match_scalar(pred_val: Any, gold_val: Any, pred_type: str, aliases: Optional[List[Any]] = None) -> bool:
    """Match a scalar prediction against gold value and aliases."""
    if aliases is None:
        aliases = []

    # Check match with primary gold value
    candidates = [gold_val] + [a for a in aliases if not isinstance(a, list)]

    if pred_type in ("number", "percent"):
        for cand in candidates:
            if is_close_num(pred_val, cand):
                return True
        return False

    elif pred_type == "date":
        # YYYY-MM-DD or YYYY-MM
        p_str = str(pred_val).strip()
        for cand in candidates:
            if p_str == str(cand).strip():
                return True
        return False

    elif pred_type == "boolean":
        if isinstance(pred_val, bool) and isinstance(gold_val, bool):
            return pred_val == gold_val
        return False

    elif pred_type == "string":
        p_norm = normalize_text(pred_val)
        for cand in candidates:
            if p_norm == normalize_text(cand):
                return True
        return False

    return False


def match_list_items(pred_list: List[Any], gold_list: List[Any], aliases: Optional[List[Any]] = None) -> bool:
    """
    Match ordered list. Rulebook §9:
    - Same length.
    - Each position i in pred_list matches gold_list[i] or any alias in aliases[i].
    """
    if not isinstance(pred_list, list) or not isinstance(gold_list, list):
        return False
    if len(pred_list) != len(gold_list):
        return False

    if aliases is None:
        aliases = [[] for _ in range(len(gold_list))]

    for i in range(len(gold_list)):
        p_item = pred_list[i]
        g_item = gold_list[i]
        item_aliases = aliases[i] if i < len(aliases) else []

        # Scalar item match
        matched = False
        candidates = [g_item]
        if isinstance(item_aliases, list):
            candidates.extend(item_aliases)
        else:
            candidates.append(item_aliases)

        if isinstance(g_item, (int, float, Decimal)):
            matched = any(is_close_num(p_item, c) for c in candidates)
        else:
            p_norm = normalize_text(p_item)
            matched = any(p_norm == normalize_text(c) for c in candidates)

        if not matched:
            return False

    return True


def match_table(pred_table: Any, gold_table: Any, aliases: Optional[List[Any]] = None) -> bool:
    """
    Table matching according to Rulebook §9:
    - First column is entity (string/code), subsequent columns are numbers in query order.
    - Row order does not matter (bipartite matching).
    """
    if not isinstance(pred_table, dict) or not isinstance(gold_table, dict):
        return False

    pred_rows = pred_table.get("rows", [])
    gold_rows = gold_table.get("rows", [])

    if len(pred_rows) != len(gold_rows):
        return False

    if len(pred_rows) == 0:
        return True

    if aliases is None:
        aliases = [[] for _ in range(len(gold_rows))]

    # Build cost/match matrix between pred rows and gold rows
    n = len(gold_rows)
    match_matrix = [[False] * n for _ in range(n)]

    for i, p_row in enumerate(pred_rows):
        if not isinstance(p_row, list):
            return False
        for j, g_row in enumerate(gold_rows):
            if not isinstance(g_row, list) or len(p_row) != len(g_row):
                continue

            # First cell: entity string matching (with alias)
            p_name = normalize_text(p_row[0])
            g_name = normalize_text(g_row[0])
            g_aliases = aliases[j] if j < len(aliases) else []
            name_candidates = [g_name]
            if isinstance(g_aliases, list):
                name_candidates.extend(normalize_text(a) for a in g_aliases)
            else:
                name_candidates.append(normalize_text(g_aliases))

            if p_name not in name_candidates:
                continue

            # Remaining cells: numbers
            cells_match = True
            for col_idx in range(1, len(g_row)):
                if not is_close_num(p_row[col_idx], g_row[col_idx]):
                    cells_match = False
                    break

            if cells_match:
                match_matrix[i][j] = True

    # Maximal bipartite matching via DFS
    match_gold_to_pred = [-1] * n

    def dfs(p_idx: int, visited: List[bool]) -> bool:
        for g_idx in range(n):
            if match_matrix[p_idx][g_idx] and not visited[g_idx]:
                visited[g_idx] = True
                if match_gold_to_pred[g_idx] == -1 or dfs(match_gold_to_pred[g_idx], visited):
                    match_gold_to_pred[g_idx] = p_idx
                    return True
        return False

    matched_count = 0
    for p_idx in range(n):
        visited = [False] * n
        if dfs(p_idx, visited):
            matched_count += 1

    return matched_count == n


def evaluate_single(pred_entry: Optional[Dict[str, Any]], gold_entry: Dict[str, Any]) -> Tuple[int, str]:
    """
    Evaluates a single prediction against ground truth.
    Returns: (reward, status_detail)
    reward: +1 for correct, -1 for wrong, 0 for abstain.
    """
    if pred_entry is None:
        return 0, "abstain_missing"

    p_type = pred_entry.get("type")
    g_type = gold_entry.get("type")
    aliases = gold_entry.get("alias", [])

    # Abstain check
    if p_type == "abstain":
        return 0, "abstain"

    # Unanswerable check
    if g_type == "unanswerable":
        if p_type == "unanswerable":
            return 1, "correct_unanswerable"
        else:
            return -1, "wrong_hallucinated_unanswerable"

    if p_type == "unanswerable":
        return -1, "wrong_falsely_claimed_unanswerable"

    # Type mismatch
    if p_type != g_type:
        return -1, f"wrong_type_mismatch(pred={p_type}, gold={g_type})"

    p_ans = pred_entry.get("answer")
    g_ans = gold_entry.get("answer")

    if p_type in ("number", "percent", "date", "boolean", "string"):
        if match_scalar(p_ans, g_ans, p_type, aliases):
            return 1, f"correct_{p_type}"
        return -1, f"wrong_{p_type}_value"

    elif p_type == "list":
        if match_list_items(p_ans, g_ans, aliases):
            return 1, "correct_list"
        return -1, "wrong_list_value"

    elif p_type == "table":
        if match_table(p_ans, g_ans, aliases):
            return 1, "correct_table"
        return -1, "wrong_table_value"

    return -1, "wrong_unrecognized_format"


class LocalEvaluator:
    """Official Penalized Accuracy evaluator and error analyzer."""

    def __init__(self, ground_truth_path: str, questions_path: Optional[str] = None):
        self.ground_truth: Dict[str, Dict[str, Any]] = {}
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    self.ground_truth[item["id"]] = item

        self.questions: Dict[str, Dict[str, Any]] = {}
        if questions_path:
            with open(questions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        self.questions[item["id"]] = item

    def evaluate_predictions(self, submission_path: str) -> Dict[str, Any]:
        predictions: Dict[str, Dict[str, Any]] = {}
        with open(submission_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    predictions[item["id"]] = item

        total_questions = len(self.ground_truth)
        total_score = 0
        counts = {"correct": 0, "wrong": 0, "abstain": 0}
        type_breakdown = defaultdict(lambda: {"correct": 0, "wrong": 0, "abstain": 0, "total": 0})
        biz_breakdown = defaultdict(lambda: {"correct": 0, "wrong": 0, "abstain": 0, "score": 0, "total": 0})
        errors = []

        for q_id, gold in self.ground_truth.items():
            pred = predictions.get(q_id)
            g_type = gold.get("type", "unknown")
            biz_id = self.questions.get(q_id, {}).get("business_id", "unknown")

            reward, detail = evaluate_single(pred, gold)
            total_score += reward

            if reward == 1:
                counts["correct"] += 1
                type_breakdown[g_type]["correct"] += 1
                biz_breakdown[biz_id]["correct"] += 1
            elif reward == -1:
                counts["wrong"] += 1
                type_breakdown[g_type]["wrong"] += 1
                biz_breakdown[biz_id]["wrong"] += 1
                errors.append({
                    "id": q_id,
                    "business_id": biz_id,
                    "question": self.questions.get(q_id, {}).get("question", ""),
                    "gold_type": g_type,
                    "gold_answer": gold.get("answer"),
                    "pred_type": pred.get("type") if pred else None,
                    "pred_answer": pred.get("answer") if pred else None,
                    "detail": detail
                })
            else:
                counts["abstain"] += 1
                type_breakdown[g_type]["abstain"] += 1
                biz_breakdown[biz_id]["abstain"] += 1

            type_breakdown[g_type]["total"] += 1
            biz_breakdown[biz_id]["total"] += 1
            biz_breakdown[biz_id]["score"] += reward

        pa_score = total_score / total_questions if total_questions > 0 else 0.0

        return {
            "total_questions": total_questions,
            "total_score": total_score,
            "penalized_accuracy": pa_score,
            "counts": counts,
            "type_breakdown": dict(type_breakdown),
            "biz_breakdown": dict(biz_breakdown),
            "errors": errors[:50]  # top 50 sample errors
        }

    def print_report(self, results: Dict[str, Any]):
        total = results["total_questions"]
        pa = results["penalized_accuracy"]
        c = results["counts"]
        print("=" * 70)
        print("   XRAY PENALIZED ACCURACY (PA) BENCHMARK REPORT")
        print("=" * 70)
        print(f" Total Questions : {total}")
        print(f" Correct (+1)    : {c['correct']:5d} ({c['correct']/total*100:5.2f}%)")
        print(f" Wrong   (-1)    : {c['wrong']:5d} ({c['wrong']/total*100:5.2f}%)")
        print(f" Abstain ( 0)    : {c['abstain']:5d} ({c['abstain']/total*100:5.2f}%)")
        print(f" Total Score     : {results['total_score']}")
        print(f" Penalized Acc   : {pa:+.4f} (Max +1.0000, Min -1.0000)")
        print("-" * 70)
        print(" TYPE BREAKDOWN:")
        for t, stats in sorted(results["type_breakdown"].items()):
            t_total = stats["total"]
            t_score = stats["correct"] - stats["wrong"]
            t_pa = t_score / t_total if t_total > 0 else 0
            print(f"  {t:15s} [{t_total:4d}]: Correct: {stats['correct']:4d} | Wrong: {stats['wrong']:4d} | Abstain: {stats['abstain']:4d} | PA: {t_pa:+.2f}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate submission for DataMate.")
    parser.add_argument("--submission", required=True, help="Path to submission.jsonl")
    parser.add_argument("--ground-truth", default="train/answers.jsonl", help="Path to ground truth answers.jsonl")
    parser.add_argument("--questions", default="train/questions.jsonl", help="Path to questions.jsonl for context")
    parser.add_argument("--out-report", default=None, help="Save JSON report")
    args = parser.parse_args()

    evaluator = LocalEvaluator(args.ground_truth, args.questions)
    res = evaluator.evaluate_predictions(args.submission)
    evaluator.print_report(res)

    if args.out_report:
        with open(args.out_report, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
