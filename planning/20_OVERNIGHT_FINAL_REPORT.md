# DataMate — Public-test run report

**Date:** 2026-10-31
**Conclusion:** `BLOCKED` — no eligible local model/runtime is available, so real Public Test inference and answer verification could not be performed.

## A. Overview

| Item | Result |
|---|---|
| Repository / branch | `pvananh2805/DataMate-Zalo_AI`, `master` |
| HEAD at start | `132f594` |
| Worktree | Pre-existing modified/untracked files were present and preserved. At report generation, no commit or push had been made. |
| Model / version | None loaded; no checkpoint at `models\local-checkpoint` |
| Runtime / GPU | `torch` and `transformers` are not installed in `.venv`; `nvidia-smi` is unavailable. CUDA inference not tested. |
| Elapsed time | 174.12 seconds for full public business-table loading, plus a short CLI preflight; no inference duration exists. |

## B. Errors and fixes

1. **Inference could not start — HIGH.** `scripts\run_inference.py` exited with code 2 because the configured local checkpoint directory does not exist. The environment also lacks PyTorch and Transformers. This is a resource/setup blocker, not a defect that can be safely repaired by inventing answers or installing an unspecified model. No model was downloaded and no dependencies were installed.
2. **Ambiguous public table headers — MEDIUM.** Loader metadata reports 32 references with blank header columns, alongside 37 leading-metadata-row skips and 29 blank-first-row skips. The full parser pass succeeded, but these warnings make some schema interpretations uncertain. No header was guessed or edited.
3. **No code fix was made.** No source failure was reproduced; unsupported answer semantics cannot be fixed safely without a real plan, validated schema, and rulebook-backed behavior.

## C. Source changes

- No application source or public-test input files were changed.
- Added this report and `planning/19_OVERNIGHT_AUDIT.md`.
- The existing dirty worktree was left untouched. Nothing was staged, committed, or pushed.

## D. Tests and checks

| Check | Result |
|---|---|
| Full regression suite | **223 passed**, 3.08 seconds |
| `compileall` (`src`, `scripts`) | Passed |
| Public data loader/context integration | **14 businesses**, 163 tables, 1,327,686 streamed rows, zero exceptions; 174.12 seconds |
| Public question structure | **1,000 valid records**, 1,000 unique IDs, 14 matching business directories |
| Actual model inference | **Not run**; checkpoint absent |
| Model inference tests / GPU / Docker | Not run |
| Public answers / official metric | Not available or evaluated |

## E. Public Test

The public input contains 1,000 structurally valid question records and 14 corresponding business data directories. Shared `rulebook.md` and `calendar.csv` loaded for all businesses; the calendar has 1,461 records. All 163 discovered tables were streamed successfully.

No question reached model generation. Therefore the output counts are **0 answer, 0 unanswerable, 0 abstain, 0 per-question errors, and 0 output records**; this does not mean the questions were answered or the test passed. The CLI stopped before processing with:

```text
model checkpoint directory does not exist:
D:\DataMate-ZaloAI\DataMate\models\local-checkpoint
```

No `submission.jsonl` or run log was created. No resume behavior was exercised. The public source files were not modified.

## F. Answer verification

| Classification | Count |
|---|---:|
| VERIFIED | 0 |
| PARTIALLY_VERIFIED | 0 |
| UNVERIFIED predictions | 0 |
| INCORRECT | 0 |
| NOT_APPLICABLE | 0 |

There are no model predictions to verify. Structural input validation and successful data loading are not evidence that any answer is correct. The 98 loader warnings (37 leading metadata rows, 32 blank-header columns, 29 blank first rows) should be reviewed before using affected columns; no assumptions were added.

## G. Regulation and submission status

The repository's prior audit records the official 14-page Regulations review and links the official PDF at `https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf`. In this run the official portal remained a JavaScript shell and the PDF retrieval was not text-readable through the current tool. Challenge-specific model limits/cutoff, Docker/runtime instructions, exact submission schema, evaluator/metric, and six-hour limit remain **UNVERIFIED**, as documented in `planning/18_COMPREHENSIVE_AUDIT_REPORT.md`.

No model was used, so its release date, license, parameter count, total inference parameter accounting, and offline behavior are **NOT TESTED**. Docker and organizer validation were not run. No claim of regulation compliance, official score, or submission acceptance is made. Public-test questions were not used for training, annotation, or question-specific optimization; no private-test files or answers were opened.

## H. Conclusion

**`BLOCKED`** for the requested full Public Test run. The available environment supports a complete read-only public-data loader pass and the 223-test regression suite, but cannot load a model or execute inference. A valid local checkpoint, compatible runtime/hardware, and verified organizer submission/evaluation guidance are prerequisites to continue.

## Follow-up repository bug review

The later repository-wide review found and fixed three general code defects:

| Defect | Fix | Evidence |
|---|---|---|
| Empty per-group `AVG`/`MIN`/`MAX` scopes incorrectly became zero. | Return `null` for empty non-additive aggregation scopes while preserving zero for `SUM` and counts. | `tests/unit/test_query_executor.py` regression; focused executor tests passed. |
| Three Python CLI scripts could not import the source package when run directly without an externally configured `PYTHONPATH`. | Add the repository `src` directory to those entrypoints. | Direct `--help` checks passed for all eight Python scripts with `PYTHONPATH` unset. |
| Deeply nested model JSON could escape the parser as `RecursionError`, potentially aborting a batch. | Convert the recursion failure to `QueryPlanFormatError`. | `tests/unit/test_question_understanding.py` regression. |

Final follow-up checks: **228 tests passed**; `compileall` passed for `src`, `scripts`, and `tests`; `git diff --check` passed. No changes were staged, committed, or pushed. These fixes do not resolve the Public Test inference blocker: no eligible local model/runtime was available, so no predictions or answer verification were performed. The table above supersedes the earlier "No code fix was made" and 223-test baseline statements for repository code status; the original blocked Public Test result remains unchanged.

## Additional findings from the reported issue review

| Reported issue | Disposition |
|---|---|
| Full calendar overwhelms planner prompt. | Fixed by selecting only explicit date/month rows, rows within seven days of `as_of_date`, and non-empty holiday rows. Calendar rows use compact arrays and a strict 8,192-character budget. The prompt marks this as partial and instructs the planner not to infer missing calendar facts. The regression test compares a >100,000-character full calendar with the bounded context. |
| One processing error discards the whole submission. | Fixed for valid input records: a per-question processing exception is reported and produces an explicit `abstain` record; the run still exits nonzero. Pipeline errors are also run-logged. Malformed input rows, duplicate IDs, and serialization errors keep atomic fail-closed behavior so output is never silently incomplete. The organizer's schema remains unverified. |
| Numeric NaN normalization. | Fixed: float and Decimal NaN values normalize to null; infinities remain explicit errors. |
| Intermediate join limit uses result-row cap. | Fixed to enforce `max_join_rows`; a regression verifies 20 joined rows can yield one result with `max_result_rows=1`. |
| Markdown JSON fences from planner. | Fixed only for a standalone, exact JSON code block (optional `json` label). Prose and `<think>` content are not stripped or extracted and remain invalid. |
| Abstention requires a real base table. | Fixed: `base_table: null` is accepted for abstention/unanswerable plans; a non-null base table must still exist. |
| Pytest import/discovery depends on `PYTHONPATH`. | Fixed with `pytest.ini` (`pythonpath = src .`, `testpaths = tests`) and README updates. Both pytest entrypoints pass with `PYTHONPATH` unset. |
| Model loads to CPU before GPU and cache preflight uses maximum configured context. | Not changed: this environment has no PyTorch, Transformers, or CUDA device. A device-map/loading change and memory-policy revision cannot be validated here; the existing checkpoint-loading strategy and conservative KV-cache preflight remain a hardware/runtime verification item. |

After these additional changes, **238 tests passed** with `.venv\Scripts\python.exe -m pytest -q` and `PYTHONPATH` unset. `compileall` and `git diff --check` passed. Public Test inference remains blocked as previously reported.

## Prompt/context optimization follow-up

The planner now selects rulebook sections using explicit question topics (time, sales, quantity/ranking, debt, inventory) and relevant source headings instead of ranking sections on weak single-word overlap. This avoids injecting unrelated industry and inventory rules into a simple quantity-sales question. A focused regression covers a construction-materials question and verifies that the pharmacy, inventory, and answer-scoring sections are omitted while sales, time, quantity/ranking, data-layout, and unanswerable guidance remain.

Prompt construction was measured locally across all **1,000** public questions, without invoking a model: **10,377 / 17,635 / 19,855 / 20,034 characters** (minimum / median / p95 / maximum). The retained rulebook excerpt has a median of **8,385 characters** and maximum of **8,599 characters**, including the truncation notice; the selected rulebook itself is capped at 8,500 characters. This is a character measurement, not a tokenizer measurement. Compared with the earlier measured median of about 30,400 prompt characters, the median fell by about 42%; the exact token count for the user's Qwen checkpoint remains unverified because its tokenizer/checkpoint is unavailable here. No Public Test model inference or answer correctness claim is made. Selective rule retrieval remains a potential source of omitted context and must be validated with the actual checkpoint and rule-backed answer traces.
