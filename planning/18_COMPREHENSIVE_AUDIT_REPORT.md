# DataMate — Comprehensive Audit Report

**Audit date:** 2026-10-31
**Audit baseline branch / HEAD:** `master` / `8d98917`
**Scope:** Repository, existing tests, local training-package inventory already recorded in `planning/13_DATASET_ANALYSIS.md`, official Zalo AI Challenge 2026 Regulations, and the official DataMate portal frontend/API. The public-test archive remains unopened; no private-test material was accessed.

## Executive status

DataMate has a useful deterministic foundation: read-only readers for CSV/XLSX/SQLite, a declarative QueryPlan, validation, a bounded executor, answer validation, evidence/logging/replay, and a batch CLI. The complete local suite currently passes **223 tests**. Existing integration tests use synthetic fixtures and planner stubs; they are not model-accuracy evidence.

The system is **not competition-ready**. No real model has been selected, loaded, benchmarked, or trained. The local evaluator is diagnostic, not the official metric. The official Docker/runtime and detailed DataMate Evaluation/Submission Guidance have not been obtained. Most importantly, the actual business rules are passed to the planner as text; there is no general structured rulebook executor, so many answer-critical rules cannot yet be certified as applied.

The official Regulations PDF was extracted locally (14 pages) from the official Regulations page and all sections 1–12 were reviewed. It confirms data-use and annotation restrictions, an OSI license requirement for open-source software used by a model, final-solution deliverables, and that evaluation follows the metric published on the challenge portal. It does **not** establish the brief's 15B cap, model release cutoff, six-hour runtime, offline/Docker contract, or formula for Penalized Accuracy. The portal's public config-detail endpoint returned `Invalid Request` for the evaluation, rule, and submission-guidance keys found in its frontend; those technical details remain **UNVERIFIED**.

## Architecture found

```text
JSONL questions
  -> local Transformers planner adapter (QueryPlan JSON)
  -> discovered schema + QueryPlan validation
  -> read-only CSV/XLSX/SQLite loader and deterministic executor
  -> validated answer + evidence + execution trace
  -> append-only internal run log / reference JSONL serializer
```

Additional paths prepare manually annotated QueryPlans, validate business-disjoint splits, run GPU-only LoRA/full training, merge LoRA adapters, profile schemas, perform local diagnostic evaluation, and replay a logged plan. The submission serializer and local evaluator are explicitly reference/diagnostic implementations, not confirmed organizer interfaces.

## Component audit

| Component | Status | Implemented and tested evidence | Data / important limits | Compliance assessment and next work |
|---|---|---|---|---|
| Data loader | PARTIAL | Deterministic discovery and streaming readers for CSV, XLSX worksheets, and read-only SQLite; path containment and symlink checks. Unit tests cover fixtures. A read-only smoke test discovered all 339 table references across 30 training businesses and read the first row from each with zero failures. | This was a one-row smoke check, not a full scan or inference run. 115 table references carry loader warnings; header ambiguity exists in training workbooks. | No known write path to source data. Review warnings and ambiguous headers before relying on inferred schemas; test full scans/performance separately. |
| Schema profiler | PARTIAL | Bounded type/null/distinct samples, semantic tags, candidate-key heuristics, package inventory; profiler unit tests exist. | Training archive has structural profiling, not an audited relationship/join map. Candidate keys are sampled heuristics, not proof. | Keep suggestions non-authoritative; validate keys/joins and ambiguous headers against permitted training data. |
| Data normalization | PARTIAL | Explicit coercion policies for numeric/date/boolean/text, configured sparse-cell fill-down, exact grouped-row deduplication, and reviewed text aliases; unit/integration tests cover these policies and replay. | The behaviors are opt-in and only as correct as their per-business configuration; no automatic rulebook or alias discovery. | Preserve nulls by default; enable transformations only with business/rulebook evidence. |
| Business context | PARTIAL | Bounded, read-only loading of business README, rulebook, and calendar; source names/sizes/hashes recorded; fixture tests exist. | Context is sent as planner text. It is not parsed into executable rules, and prompt size limits may reject large references. | A document being present does not prove its rule was followed. Build structured rules only from verified text and test their execution. |
| Question understanding | PARTIAL | Model-independent `TextGenerationBackend`/`QueryPlanner` interfaces; JSON adapter parses and validates plans; malformed/oversized outputs fail. Unit tests use fake backends. | No checkpoint or real model inference has been tested. | Planner must remain a plan generator; preserve validation and abstention for unsupported semantics. |
| QueryPlan representation | PARTIAL | Declarative AST covers filters, joins, grouping, aggregations, calculations, time buckets, unit conversions, group exclusions, ranking, and output type. Serialization and plan tests exist. | Coverage is against synthetic plan examples, not exhaustive question-to-plan correctness. | Strong safety boundary; no SQL or executable code accepted. Test rulebook-relevant edge cases. |
| QueryPlan validator | PARTIAL | Allowlisted operations, catalog/column checks, join shape, scalar/filter/time/output constraints; rejects nonempty `business_rules`. Unit tests exist. | Validation proves structural validity, not business meaning, key uniqueness, or join cardinality correctness. | Safe rejection is preferable to an unsupported rule being claimed as applied. Add semantic checks only with evidence. |
| Query executor | PARTIAL | Deterministic in-process filters, joins, grouping, aggregate/calculation, ordering/limits, time buckets, unit conversion, and exclusion handling; unit/integration fixtures. | Some operations materialize join/group state in memory and have limits. No throughput/latency test on the full training corpus. Generic arithmetic does not itself implement cancellation, return, voucher, VAT, shipping, cost, debt, or inventory policy. | Rules not encoded by the plan/executor must not be reported as applied. Benchmark permitted real training data and add rule-specific general tests before claiming these semantics. |
| Business rules | PARTIAL | Prompt instructions describe selected safe representations (for example, group exclusions, unit conversion, Monday-based weeks); unsupported `business_rules` are rejected. | There is no structured, provenance-bearing rulebook interpreter. Rulebook coverage for returns/cancellations, voucher scope, VAT/shipping, cost, debt, inventory, and lunar dates is not certified end to end. | Core blocker. Map each applicable rule to an executable plan operation or a deterministic abstention; never rely on the model merely repeating a rule. |
| Evidence | PARTIAL | Pipeline attaches source tables/columns, row counts, and operation summaries from the validated plan; pipeline tests exercise fixtures. | Evidence is a coarse execution trace, not complete row-level/value lineage or independent proof of rule interpretation. | Improve provenance where needed; a plausible answer without sufficient evidence must not be treated as verified. |
| Logging | PARTIAL | Append-only JSONL stores question/business IDs, model label, plan, execution trace, answer, elapsed time, errors, and input metadata; tests exist. | Checkpoint/config/code/data hashes and run-level environment identity are not consistently supplied by the inference CLI. No model run has been logged. | Add reproducibility metadata before model evaluation; keep logs outside source-data roots. |
| Replay | PARTIAL | Re-executes a logged successful plan against read-only data and compares answer and trace; integration tests use fixtures. | Does not regenerate the plan or prove model determinism; requires recorded execution limits and coercions. | Useful for executor reproducibility only, not as evidence of model accuracy. |
| Evaluation | PARTIAL | Local evaluator validates records and reports exact-match diagnostics by type/business, abstention/unanswerable, invalid/missing output, and latency; unit tests exist. | This is explicitly not official Penalized Accuracy. No QueryPlan-vs-gold evaluation or model-validation score has been produced. | Do not publish local exact-match or test counts as official accuracy. Obtain organizer evaluator and its version before calculating official metrics. |
| Training-data preparation | PARTIAL | Annotation schema is manually authored `question_id`, `business_id`, `query_plan`; output joins to questions, validates plans, and splits by business. Planning prompts now include available README/rulebook/calendar like inference. Answer/gold fields are rejected from question/annotation inputs; unit tests cover leakage and context parity. | No reviewed QueryPlan annotations or actual training run exist. Existing training questions/answers have been inventoried, not converted to fabricated plan labels. Context discovery follows the local training data layout. | Train/validation business leakage is guarded. Validate annotation policy against challenge-specific guidance; reviewed labels remain prerequisite. |
| Fine-tuning / LoRA / merge | PARTIAL | CUDA-only local scripts, explicit LoRA targets, local-only model loading, parameter checks, and merge/export path exist. CLI/unit-level tests cover validation paths. | No model, CUDA runtime, or training dependencies are available here; training, merge, and exported artifacts are untested. | No training has been run. Verify model/license and the complete parameter accounting rule before using any checkpoint. |
| Local inference backend | PARTIAL | Local checkpoint preflight checks config JSON, tokenizer artifacts, local weights/indexed shards; Transformers then loads with `local_files_only=True`, `trust_remote_code=False`, and offline flags. CUDA, precision, configurable prompt/output/parameter limits, pre-transfer weights/buffers plus architecture-derived KV-cache estimate, and configurable per-generation timeout with logged settings are implemented. | No real checkpoint/GPU/Transformers smoke test. File presence does not establish model/license eligibility. Memory estimate excludes runtime workspace and may be conservative; no whole-batch deadline. Parameter count is measured for the single loaded causal LM, not an inventory of additional models. | Model-ready scaffold only. Add license/parameter manifest and batch deadline; verify on eligible hardware/checkpoint. |
| Manual testing | DONE | Runner and synthetic 100-question example bundle exist; prior run processed 100/100 with a stub planner, including 80 labeled expected answers. | Synthetic fixture and scripted plans only; it is not an AI benchmark or challenge-data score. User-owned root manual-test files remain ignored/untracked. | Keep the user's manual-test files separate from official training/test material. |
| Submission serializer | PARTIAL | Atomic JSONL writer validates answer types/IDs and rejects duplicates; tests exist. It follows the training rulebook reference shape. | No organizer validator or Docker contract was run. | Output schema and answer nuances remain unverified against official challenge guidance. |
| Docker / entrypoint | BLOCKED | No Dockerfile, Compose file, or Docker-specific entrypoint was found. Batch inference CLI exists independently. | Docker CLI is unavailable; official Docker/runtime guide is not accessible through the current portal API. No image was built or run. | Do not invent paths, mounts, base image, or resource limits. Obtain the official guide and then implement and test the exact interface. |

## Training rulebook-to-executor coverage

The rulebook below is the document packaged with the **training** archive. It is not the missing challenge Docker guide or evaluator. `PARTIAL` means the QueryPlan has primitives that can express some cases; it does not mean the planner reliably identifies or applies the rule. No end-to-end gold-question evaluation was run.

| Rulebook section / rule | Status | Executor capability and exact gap |
|---|---|---|
| §1 Revenue eligibility: exclude canceled and all returned orders, including partial return; include in-delivery orders | PARTIAL | `group_exclusions` can exclude every row sharing an order key when a matching status row is present; filters can select allowed states. The planner must select the correct reliable key/status column. No rule engine enforces this, and conflicting/missing status is not resolved. |
| §1 Revenue amount: sum line net amount; exclude order-level discount, VAT, shipping | PARTIAL | Can sum an explicitly identified source amount and apply filters. No row-level arithmetic, semantic field selection, or automatic separation of line/order discounts, tax, and delivery fee; must abstain if source columns cannot support it. |
| §1 Order period by order date; payments, payment type, customer group; cancellation rate and period comparisons | PARTIAL | Date-range filters, aggregate-specific ranges/filters, distinct counts, groupings, and ratio/difference/percent-change calculations exist. Correct order/payment semantics and required rounding are model-plan choices, not validated rulebook behavior. |
| §2 Relative dates, inclusive ranges, N-day windows, strict/inclusive thresholds | PARTIAL | Executor accepts explicit ISO date ranges and comparison operators. Relative-language resolution is left to the planner using `as_of_date`; no deterministic phrase parser or timezone/shift engine exists. |
| §2 Week/month/quarter and current/previous periods | PARTIAL | Time buckets use Monday week starts and Gregorian month/quarter/year. The model must calculate the correct range boundaries from `as_of_date`; calendar completeness and current-period semantics are not enforced by the executor. |
| §2 Lunar New Year and lunar calendar | PARTIAL | Calendar rows can be included in the prompt, but no deterministic calendar resolver converts a lunar festival phrase to exact Gregorian boundaries or validates the selected dates. |
| §2 Morning/afternoon shifts | NOT STARTED | No time-of-day extraction, timezone policy, or shift-boundary operator; a raw string filter is not a safe general substitute. |
| §3 Ranking, quantity normalization, fees/non-merchandise rows | PARTIAL | Multi-key sorting, `limit`, filters, grouping, and declared base/large-unit conversion exist. Unit lookup requires valid mappings. Service-fee exclusion, product eligibility, and required tie-break columns are not enforced from the rulebook. |
| §3 Tie-break and stable ranking | PARTIAL | Sort keys can express multiple ordering keys and stable sorting preserves source order for remaining ties. The validator does not require the rulebook tie-break, so an incomplete plan can return a plausible but nondeterministic/incorrect ranking. |
| §3 Customer identity and aliases | NOT STARTED | Equality uses source values; there is no general alias/diacritic/format normalization or rulebook-driven decision between customer ID and name. |
| §3 Average order value rounding | PARTIAL | Distinct aggregation, ratio, and decimal rounding are available. The validator does not ensure the numerator/denominator have the same valid order population or that the right business amount was selected. |
| §4 Debt balance, later payments, COD/marketplace exclusion, aging | NOT STARTED | Generic joins and aggregates exist, but no event-time balance/window computation, same-day-vs-later payment treatment, payment/refund policy, COD/marketplace exclusion, or debt-aging operator is implemented. |
| §5 Purchases, weighted-average purchase price, canceled receipts | PARTIAL | Filters, time ranges, sums, and ratios can express some weighted averages if appropriate amount/quantity fields are already present. No rule validates canceled receipt handling, receipt dates, or whether source amounts are on compatible units. |
| §5 Historical inventory, end-of-day stockout, stock value | NOT STARTED | No cumulative/window balance over stock movements, event ordering, or end-of-day snapshot operation. A simple sum of rows cannot safely replace these semantics. |
| §5 Slow-moving items and price-change effects | PARTIAL | Grouped aggregates with independent time scopes and group filters can express selected comparisons. There is no general anti-join/event-history operator or guarantee the planner identifies eligible prior/current sales. |
| §5 Cost and gross/net profit | PARTIAL | Sums and aggregate-level differences are supported if line cost/profit values are already available. No row-level multiplication of quantity × per-unit cost, weighted lot costing, or enforcement of the stated source-of-cost field. |
| §6 Industry-specific rules | NOT STARTED | No domain rule executor for cafe size suffixes, pharmacy prescription/active-ingredient/lot-expiry rules, or construction delivery-event counting. Exact text filters may express narrow cases but do not implement these rules. |
| §7 Unanswerable versus zero; partial/full date coverage | PARTIAL | Pipeline can abstain and uses observed date coverage to mark a wholly out-of-coverage requested range unanswerable. It cannot prove a column is semantically complete; missing schema often becomes a plan/CLI error rather than a validated per-question abstention. |
| §8 Sparse repeated cells and aliases | PARTIAL | Loader preserves blanks as null by default. Explicit `fill_down_group` supports confirmed repeated cells; `deduplicate_group` suppresses exact duplicate normalized rows only within a configured group; text `aliases` can map reviewed spellings to a canonical value. All are opt-in and logged/replayed. No alias discovery or customer-identity decision is implemented. |
| §9 Answer types/schema/reward | PARTIAL | Answer validator, reference JSONL serializer, and local exact-match diagnostics exist. They follow training rulebook conventions, not a verified organizer validator or official metric. |
| §10 Business vocabulary and units | PARTIAL | Unit conversion can use an explicit base/large-unit/factor mapping and rejects an unknown mapped unit. It does not discover the business mapping reliably or interpret all synonyms/abbreviations itself. |

## Official-regulation traceability

Statuses below describe evidence available during this audit, not a declaration of full compliance.

| Requirement | Status | Official evidence / actual state |
|---|---|---|
| Use of competition data | PARTIAL | Regulations §§4.1–4.3: competition data is for the stated non-commercial/academic purposes, attribution to `challenge.zalo.ai` is required, and data must not be shared outside the team during the competition. Local train archive is kept untracked; public-test ZIP has not been opened. Data-retention/deletion obligations remain for the end of the competition, and private-repository collaborators have not been verified. |
| Manual labels/predictions on validation or test | PARTIAL | Regulations §4.2 prohibits using human labels or predictions on validation/test to create the submission. Current annotation preparation accepts manually authored plans and does not generate plans from gold answers; however, it does not itself enforce dataset provenance. No validation/test annotations were created. |
| Open-source licenses | PARTIAL | Regulations §4.5 requires valid OSI licenses without commercial-use restriction for open-source libraries/code used by a model, unless written permission allows otherwise. Dependency/model licenses have not been fully inventoried; model-weight license eligibility remains unverified. |
| Originality / third-party rights | PARTIAL | Regulations §11.1 requires the submission to be original and not infringe third-party rights; entrants are responsible for claims arising from code/data use. No full provenance/license audit of dependencies, model weights, or copied assets has been completed. |
| Final package contents | PARTIAL | Regulations §5.2 requires the final solution to include training code, testing code, model, method description, and run instructions; technical details and challenge-specific time limits are to be announced. Some source/docs exist, but no eligible model, official package, or verified runner is present. |
| Evaluation metric | PARTIAL | Regulations §6.1 says submissions are ranked by the evaluation metric on the portal. The exact metric/evaluator is not stated in the Regulations PDF, and the portal API request for `evaluationMetric` returned `Invalid Request`; official PA and abstention semantics are not verified. |
| Model total parameter cap, release date, MoE accounting, quantization | UNVERIFIED | Not found in extracted Regulations §§1–10. User-provided brief says ≤15B and cutoff requirements, but the challenge-specific model rules could not be retrieved. Current backend defaults to 15B for one loaded model; this does not establish compliance. |
| Offline inference / no commercial LLM API / no Internet fetch | UNVERIFIED | Not established by the Regulations PDF or accessible challenge technical guidance. Current backend is designed for local-only loading, but offline behavior has not been tested with a real checkpoint/container. |
| Private-test restrictions, ID/business leakage, `/data` read-only | UNVERIFIED | Regulations §4.2 specifically covers human labels/predictions on validation/test; the broader prohibition and runner filesystem contract in the user brief were not confirmed in technical guidance. Current loader uses read-only readers, but no official runner test was possible. |
| Docker image, entrypoint, input/output paths, resource limits, six-hour cap | BLOCKED | Regulations §§5.1–5.2 defer technical requirements/time limits to the challenge portal or challenge description. Official Docker/submission guidance was not returned by the accessible portal endpoint. No Docker implementation or runtime claim is made. |
| Output schema and `.jsonl` details | PARTIAL | Official portal bundle identifies `.jsonl` as the submission format. Training `rulebook.md` provides a reference `id/type/answer` shape. Neither the portal validator nor exact technical guidance was available, so schema compatibility is not verified. |
| No hardcoding by question/business ID; no gold-answer leakage | DONE | Project implementation uses a general plan/execution pipeline; no ID-based answer lookup was found in the audited inference path. Dataset preparation does not derive QueryPlan labels from answer values. This is a code-path observation, not an organizer certification. |
| Reproducibility and logging | PARTIAL | Run log and replay paths exist and have fixture tests. No real model run, official reproducibility tolerance, or full version/hash manifest has been verified. Regulations do not state a numerical rerun tolerance. |
| Rule changes / organizer updates | PARTIAL | Regulations §12.1 allows the organizer to amend challenge rules, content, format, schedule, and prizes by official notice. The portal and challenge-specific guidance must be rechecked before final implementation/submission. |

### Official sources checked

- Regulations landing page: <https://challenge.zalo.ai/regulations> — official PDF extracted locally, all 14 pages/sections 1–12 reviewed; relevant engineering clauses include §§3, 4.1–4.5, 5.1–5.2, 6.1–6.2, 8.2, 11.1, and 12.1.
- DataMate portal: <https://challenge.zalo.ai/portal/datamate/> — frontend bundle exposes challenge detail keys `evaluationMetric`, `rule`, and `submissionGuidance` and `.jsonl` submission format. The corresponding public config-detail requests returned `{"error_message":"Invalid Request","data":null,"error_code":-10}`.
- Training package rulebook: `data/train/train/rulebook.md`, previously inventoried in `planning/13_DATASET_ANALYSIS.md`; it is business guidance supplied with training data, not a substitute for Regulations, official evaluator, or Docker guidance.

## Training-package inventory and data boundary

The existing training-only inventory records 3,000 paired questions/answers across 30 business packages and CSV/XLSX/SQLite files. These are aggregate structural statistics from the training archive; they are not model predictions or accuracy measurements. No public-test content was inspected in this audit and no private-test content was accessed. The training archive and extracted data are untracked and must not be staged or committed. Cite `challenge.zalo.ai` when using competition data and follow the end-of-competition deletion requirement for personal data.

## Test evidence and readiness

- **Current run:** `$env:PYTHONPATH='src'; .venv\Scripts\python.exe -m pytest -q` — **223 passed in 3.30s**.
- **Latest focused suites:** local inference/config **20 passed**; training preparation/split **10 passed**; normalization/replay/pipeline **49 passed**. These runs overlap and are not additive.
- **CLI/build checks:** all eight CLI `--help` checks, `compileall -q src tests scripts`, `git diff --check`, and UTF-8/trailing-whitespace checks passed.
- **Read-only training-data loader smoke:** 30 businesses, 339 discovered table references, one first row read from each, 115 tables with warnings, zero loader failures. No public/private test data was opened.
- **Earlier recorded deployment check:** clean dev environment full suite **199 passed**, compileall, eight CLI `--help` checks, and `git diff --check` passed; see `planning/17_DEPLOYMENT_REPORT.md`.
- **Synthetic manual test:** previously recorded 100/100 stub-planner runs; 80/80 labeled answers matched fixture expectations. Not model inference and not a challenge score.
- **Real model inference / validation accuracy / official PA:** NOT RUN.
- **Fine-tuning / merge / export with a checkpoint:** NOT RUN.
- **Docker build, offline-container test, organizer validator:** NOT RUN.

| Readiness area | Status | Reason |
|---|---|---|
| Model-ready | PARTIAL | Local checkpoint/tokenizer/weight preflight, local-only Transformers adapter, loaded-model parameter guard, configurable precision/context/output, pre-transfer weights/buffers plus KV-cache estimate, and optional per-generation timeout exist. No eligible checkpoint/GPU validation, license/parameter manifest, whole-batch deadline, complete runtime-memory estimate, or model benchmark. |
| Training-ready | PARTIAL | Human annotation schema, business-heldout split, validation, training and merge scripts exist; no reviewed annotations or executed training/export. |
| Docker-ready | BLOCKED | Official Docker/runtime guidance unavailable; no Dockerfile/entrypoint or Docker runtime verification. |
| Submission-ready | BLOCKED | Serializer exists but exact schema, evaluator, artifact contract, and runner interface are unverified. |

## Audit findings and fixes

No source code was changed during Giai đoạn A. This preserves the requested audit-before-implementation order. The material blockers discovered are:

1. **Official technical guidance is missing:** evaluation details, Docker/runtime contract, model rules, output validator, and challenge-specific limits.
2. **Business-rule execution is incomplete:** passing a rulebook to the planner does not implement its rules; returns/cancellations, vouchers, VAT/shipping, costing, debt, inventory, and lunar-calendar requirements need traceable executable behavior or abstention.
3. **No real model evidence:** the pipeline is validated with synthetic fixtures and stubs only.
4. **Resource/preflight gaps:** KV cache has a static estimate, but runtime workspace is excluded; the optional per-generation timeout is not a whole-batch deadline, and complete multi-model parameter accounting is not present.
5. **Reproducibility metadata gaps:** run-level source/config/data/checkpoint hashes and environment versions are incomplete.
6. **Code-sharing gate:** the remote GitHub repository is private and the GitHub collaborator list returned only `pvananh2805` on 2026-10-31. No other direct repository access was listed. Regulations §8.2 restricts private sharing of competition-developed source outside the team during the competition; keep this repository private and recheck access before adding collaborators or publishing code. Training data/checkpoints are not part of the audited commits.

### Milestone A.1 — real training-data loader smoke

Ran the existing loader read-only across all 30 permitted training business packages. It discovered 339 tables and read one data row from each with zero load failures. 115 table references emitted existing warnings; this confirms readability, not header correctness, complete row integrity, query-answer correctness, or performance. The public-test archive and private-test material were not opened.

### Milestone B.1 — minimum CUDA storage preflight

After completing the audit, `LocalTransformersBackend` now measures loaded model parameter and buffer storage and checks it against free CUDA memory before moving the model to `cuda:0`. If the minimum storage does not fit, loading fails with a specific error instead of attempting the transfer. The backend retains the parameter/storage and CUDA-memory values for later diagnostics.

Two focused mocked-runtime tests cover rejection before device transfer and the sufficient-minimum-memory path. The full suite then passed 201 tests; no model, GPU, training, or benchmark was used. The initial storage guard did not account for KV cache.

### Milestone C.1 — explicit sparse-cell normalization

Added opt-in `fill_down_group` to the existing per-column coercion policy. A policy can explicitly forward-fill a sparse group key and carry a selected field such as order status only while that key remains the same. Blank values are unchanged unless this policy is configured. Unknown group-column references fail explicitly. The policy is part of logged coercion metadata and replay validation, so a replay reuses the same transformation.

Unit tests cover within-group propagation, group boundary reset, and invalid group references. Integration tests confirm JSON configuration loading, returned-order exclusion, and deterministic replay. The combined targeted suite passes (42); the full suite passes (207). This is a general primitive, not an automatic rulebook parser, and must only be configured where the source rulebook confirms sparse-cell semantics.

### Milestone C.2 — exact duplicate rows within a configured group

Added opt-in `deduplicate_group` to the same explicit coercion policy. It compares the complete normalized row after configured fill-down and suppresses a row only when an identical row has already occurred for that group. Rows in different groups, ungrouped rows, and non-identical lines remain distinct. The single-group policy is validated, the work is capped at 100,000 unique grouped rows to bound memory, and reaching the cap fails explicitly rather than silently dropping deduplication. The field is recorded in run metadata and accepted by replay.

Tests cover exact duplicate removal, retaining different lines and other-group rows, preserving ungrouped rows, limit failure, JSON configuration, and replayed end-to-end sales totals. The combined targeted suite passes (42), and the complete suite passes (207), with compile, CLI, and diff checks green. This remains opt-in: it must be enabled only where the rulebook and business source confirm the grouping key and exact-duplicate semantics.

### Milestone C.3 — explicit text alias mapping

Added optional text-column `aliases` mapping to normalization. It matches configured source spellings after trimming edge whitespace and case-folding, then emits the explicitly selected canonical spelling; unmapped values are preserved. Alias policies are serialized in run metadata and accepted by replay. Non-text/identifier columns and empty/conflicting keys are rejected, so this does not merge customer identifiers or discover synonyms automatically. Tests cover case-insensitive mapping, preserving unrelated values, configuration loading, and end-to-end replay of an aliased returned status.

The targeted normalization/inference/replay/pipeline suite passes (49); the full suite passes (219), with compileall, all eight CLI help checks, and diff checks green.

### Milestone D.2 — conservative KV-cache preflight

The backend now estimates single-request KV-cache bytes from local model config (layer count, KV heads, head dimension, configured maximum prompt plus generation tokens, and selected precision). Before device transfer it requires free CUDA memory for both loaded weights/buffers and the estimate. Unsupported/missing dimensions fail closed; an exact-fit and a cache-insufficient path are covered with mocked runtime tests. The estimate assumes full-context KV storage and excludes runtime workspace, allocator fragmentation, and implementation-specific overhead; it is not a GPU benchmark and cannot certify a checkpoint will run.

The focused local-inference/CLI integration tests pass (9), and the full suite passes (209); compileall, all eight CLI help checks, and diff checks pass. No model or GPU was used.

### Milestone D.3 — opt-in per-generation timeout

Added optional `generation_timeout_seconds` to inference config and CLI, including manual-test CLI. When configured, Transformers `generate(max_time=...)` is used and elapsed wall time is checked; if a partial completion arrives after the deadline, it becomes an explicit error rather than a plausible QueryPlan. The default remains `null` because the official per-question runtime budget is not verified. Effective precision, token/parameter limits, and timeout setting are included in each inference run record. This is per generation only, not a whole-batch deadline.

The focused local-inference/config tests pass (15), and the full suite passes (215); compileall, all eight CLI help checks, config JSON parsing, and diff checks pass. No model or GPU was used.

### Milestone D.4 — local checkpoint file preflight

Before importing inference dependencies or checking CUDA, the backend validates a local `config.json` object with `model_type`, recognizes tokenizer artifacts, and verifies a supported single-file weight artifact or parses a sharded weight index and checks every referenced shard exists. Optional tokenizer JSON/config files are parsed. Missing/invalid files fail with a specific local-checkpoint error; no remote fallback is possible. Transformers remains responsible for loading actual tensors/tokenizer and can still reject malformed or incompatible contents. Synthetic filesystem tests cover invalid config and missing indexed shards; no real checkpoint was present.

The focused local-inference/config tests pass (17), and the full suite passes (217); compileall, all eight CLI help checks, config JSON parsing, and diff checks pass.

### Milestone D.5 — configurable prompt context limit

Added `max_prompt_tokens` to inference config and CLI, including manual testing. It constrains tokenized prompt/context length independently from `max_new_tokens`; the VRAM estimate uses their sum. The effective value is captured in each run log. Defaults preserve the former 16,384-token prompt cap, and invalid config limits fail before model loading.

The focused inference/local-backend tests pass (20); the full suite passes (223), with compileall, all eight CLI help checks, config JSON parsing, and diff checks green. No model or GPU was used.

### Milestone E.1 — training context parity and label-leakage guard

Training example prompts now include the available business README, shared training rulebook, calendar rows, schema, and question `as_of_date`, matching inference context assembly instead of training on schema-only prompts. The accepted annotation schema is exactly `question_id`, `business_id`, and `query_plan`; unknown fields such as `answer` are rejected. Training question records also reject answer/gold/label/target/query-plan keys. These checks do not prove that a human-authored QueryPlan itself was reviewed or that input files came from the permitted training split; those remain procedural gates.

The focused training preparation/split tests pass (10); the full suite passes (220), with compileall, all eight CLI help checks, and diff checks green. No training was run and no question/answer data was committed.

## Prioritized next work

| Priority | Work | Gate / evidence to close |
|---|---|---|
| P0 | Obtain and archive the official DataMate Rule, Evaluation, and Submission Guidance, including Docker/runtime instructions and evaluator version. | Organizer source URL/file and clause-to-test mapping; do not infer missing contract. |
| P0 | Preserve the verified private-repository access boundary. | The current direct collaborator list contains only `pvananh2805`; recheck before adding users or making the repository public. Keep training/test data and checkpoints out of Git. |
| P1 | Build a rulebook-to-executor coverage matrix from training rulebook text and add general, synthetic tests for supported semantics; return validated abstention for unsupported rules. | Each rule has provenance, an implemented operation, focused tests, and an explicit unsupported path. |
| P1 | Close remaining model/runtime controls: establish the eligible model/license/parameter manifest once official rules are available; record all inference components; add batch deadline and complete run/data/code hashes. | Do not select or download a model before verified rules. Test metadata/accounting with fixtures, then verify resource behavior on eligible hardware. |
| P1 | Validate schemas, joins, and streaming behavior against permitted training business packages without touching public/private tests. | Record selected business packages and exact checks; keep outputs outside source data and avoid fabricated labels. |
| P2 | Finish reviewed QueryPlan annotations, business-heldout validation, evaluator integration, and benchmark only after official metric is obtained. | No annotation from answer gold or validation/test; distinguish plan accuracy, answer accuracy, and official metric. |
| P2 | Implement Docker/entrypoint/submission only against the official technical guide and run the prescribed container checks. | Organizer-compatible build and batch smoke test; until then remain BLOCKED. |

## Audit conclusion

The codebase is a **tested synthetic-fixture prototype with a model-ready scaffold**, not a verified competition submission. The most valuable existing work is the constrained deterministic execution boundary, read-only loader, explicit plan validation, and honest separation of local diagnostics from official scoring. The most urgent corrections are to obtain challenge-specific technical requirements and make rule execution auditable; no claim of official compliance, model performance, Docker functionality, or submission acceptance is warranted yet.
