# Feedback Input Format

`feedback` does not include a benchmark evaluator. Its diagnosis entry point reads normalized records produced by an external evaluator.

## Input

Pass either a JSONL file or a run directory. For a run directory, feedback reads `<run_dir>/normalized/records.jsonl`.

```bash
python -m evosynth.feedback.run \
  --records <records.jsonl|run_dir> \
  --capabilities <base_pool.json> \
  --gen-dir <generation_dir> \
  --out-dir <feedback_dir> \
  --base-url "$JUDGE_BASE_URL" \
  --model "$JUDGE_MODEL" \
  --api-key-file "$FEEDBACK_API_KEY_FILE"
```

All three model options must be non-empty.

## Record schema

Each line is one JSON object:

```json
{
  "record_id": "case-001",
  "benchmark": "external-eval",
  "task_id": "task-001",
  "run_id": "run-001",
  "task": {"instruction": "Natural-language task description", "context": {"tools": ["search", "update"]}},
  "trajectory": [
    {"role": "user", "content": "Natural-language task description"},
    {"role": "assistant", "content": "...", "tool_calls": []},
    {"role": "tool", "content": "..."}
  ],
  "evaluation": {"status": "failed", "score": 0, "details": {"reason": "Optional failure reason"}, "errors": []}
}
```

The required fields for diagnosing a failed case are `record_id`, `task.instruction`, and `trajectory`. A case is treated as failed when `evaluation.status == "failed"`, `evaluation.success == false`, or `evaluation.score == 0` unless the status is `error` or `missing`.

Reasons under `evaluation.details` such as `reason`, `explanation`, or `feedback` are used when present. Without a reason, feedback performs a blind diagnosis.

## Tool lists

Tools are read first from `task.context.tools` or `metadata.tools`. If unavailable, provide `--tools-file`:

```json
{"external-eval": ["search", "update"], "*": ["search"]}
```

If no list is provided, feedback collects tools called in the trajectory and marks the result as inferred.
