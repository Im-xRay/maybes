# Docker local inference (development)

This is a local CPU development runner, not the organizer's final-solution
image. The official DataMate rule requires offline inference, read-only `/data`,
and output at `/result/submission.jsonl`; it also refers to a separate Docker
guideline that specifies the required base image. That base-image/runtime
guideline is not exposed by the public portal response available to this
workspace, so this configuration must not be submitted as organizer-compliant.

## Prepare local folders

Create `F:\DataMate\input` with the challenge input layout:

```text
F:\DataMate\input\
  questions.jsonl
  rulebook.md
  calendar.csv
  data\
    <business_id>\
      ...
```

Copy `.env.docker.example` to `.env` and adjust the input/output directories.
The model directory defaults to `F:\AI-Models\Qwen2.5-0.5B-Instruct`. Keep
model files and all test data outside the Docker build context and Git.

## Build and run

```powershell
docker compose --env-file .env -f compose.dev.yaml build
docker compose --env-file .env -f compose.dev.yaml run --rm datamate
```

The image installs its dependencies at build time. The inference container has
no network, mounts `/data` and the model read-only, writes logs/results under
`/result`, and uses CPU/FP32 so it does not depend on host GPU passthrough. It
uses the local model only; it never downloads a checkpoint during inference.
The 2,048-token prompt limit is enforced with the checkpoint tokenizer: the
planner first retains the question, complete schema, and supplied business
rules, then fits lower-priority calendar rows and marked rulebook/README
excerpts. If required context alone exceeds the limit, the question safely
abstains rather than truncating schema or business rules.

This CPU configuration is intended for local plumbing checks. It is not a
performance recommendation for the competition's six-hour evaluation window.
The organizer's required CUDA base image, GPU runtime flags, and packaging of
model weights must be applied after obtaining the separate Docker guideline.
