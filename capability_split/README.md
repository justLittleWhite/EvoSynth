# Capability Split

`capability_split` takes a benchmark (a task directory, JSON/JSONL, or CSV) and uses an LLM to group tasks by transferable solving structure. It writes a JSON array of capability archetypes.

## Method

The splitter first extracts structural parts from each task, then groups those parts into archetypes. Domain, tool, and entity names are not classification boundaries by default. The output preserves required ordering, dependencies, state transitions, and verification conditions while avoiding task-specific labels or overly broad operation names.

## Usage

```bash
export CAPABILITY_SPLIT_BASE_URL=https://.../v1
export CAPABILITY_SPLIT_MODEL=your-model
export CAPABILITY_SPLIT_API_KEY_FILE=/path/to/api_key.txt

uv run python -m capability_split.run_split \
    --benchmark mcp-universe \
    --tasks <MCP_UNIVERSE_ROOT>/mcpuniverse/benchmark/configs/mcpuniverse \
    --out out/mcpuniverse_capabilities.json

uv run python -m capability_split.run_split \
    --benchmark mcp-atlas \
    --tasks <MCP_ATLAS_ROOT>/mcp_atlas_tasks.csv \
    --field PROMPT \
    --out out/mcp_atlas_capabilities.json
```

The model, base URL, and API key must be supplied through command-line options or environment variables. Sampling temperature defaults to `0.2` in `splitter.py`.

## Pipeline

1. `load_tasks` supports `description.md`, JSON/JSONL, CSV, YAML, Markdown, and text task files.
2. Stage 1 extracts structural parts independently. Successful results are appended to `{output_stem}.stage1_signatures.cache.jsonl`, allowing interrupted runs to resume.
3. Stage 2 groups parts in batches of 60, merges near-duplicate candidates, and retries uncovered parts when coverage is low.
4. Before writing the final file, internal membership IDs are removed. Validation checks archetype names, required fields, and non-empty descriptions.

## Output

```json
[
  {
    "archetype": "aggregate_report",
    "archetype_desc": "Aggregate and materialize: ...",
    "verify": "The artifact schema and computed values match the specification."
  }
]
```

Each item contains exactly `archetype`, `archetype_desc`, and `verify`.

## Files

- `prompts.py` — extraction and grouping prompts
- `splitter.py` — loading, LLM splitting, validation, and persistence
- `run_split.py` — CLI entry point
