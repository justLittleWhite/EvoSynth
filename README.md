![EvoSynth logo](assets/evosynth-logo.png)

# EvoSynth: Environment-Free Data Synthesis for Self-Evolving Tool-Use Agents

> 🎯 **EvoSynth** is a self-evolving, environment-free framework for synthesizing tool-use trajectories. It diagnoses benchmark failures, evolves a Hierarchical Capability Evolution Tree (HCET), materializes compact controlled worlds, and continuously curates targeted supervision for tool-using agents.

## ✨ Highlights

- 🌳 **HCET evolution** — Refine broad capabilities, add missing dimensions, and reallocate quotas according to benchmark failures and node lifecycle states.
- 🌐 **Environment-free materialization** — Build compact worlds from tool schemas and freeze accepted observations for state-consistent, deterministic replay.
- 🧩 **Capability abstraction** — Extract transferable solving structures from benchmark tasks.
- 🧵 **Trajectory synthesis** — Generate tasks and interaction trajectories from capabilities, MCP servers, personas, and controlled world states.
- 🔍 **Validation** — Check schemas, contracts, world state, and consistency before publishing data.
- 🧑‍🏫 **Teacher rollout** — Run a model in the tool environment and retain messages, tool calls, answers, and judge results.
- 🔁 **Feedback loop** — Diagnose external evaluation failures, analyze supply and demand, and produce the next capabilities and curriculum configuration.

## 🚀 Getting Started

### 1. Installation

Python 3.10 or newer is required:

```bash
git clone https://github.com/justLittleWhite/EvoSynth.git
cd EvoSynth
uv sync
uv pip install -e . --no-build-isolation --no-deps
```

Alternatively:

```bash
pip install -e .
```

Install the optional SFT dependency with `pip install -e '.[sft]'`.

### 2. Configure model APIs

Copy the environment template and configure model endpoints and API-key file paths:

```bash
cp .env.example .env
```

The scripts use OpenAI-compatible APIs. The synthesis and world models may use different endpoints.

### 3. Split benchmark capabilities

```bash
uv run python -m capability_split.run_split \
  --benchmark mcp-atlas \
  --tasks /path/to/tasks.csv \
  --field PROMPT \
  --out outputs/capabilities.json \
  --model "$CAPABILITY_SPLIT_MODEL" \
  --base-url "$CAPABILITY_SPLIT_BASE_URL" \
  --api-key-file "$CAPABILITY_SPLIT_API_KEY_FILE"
```

The output contains `archetype`, `archetype_desc`, and `verify`. See [capability_split/README.md](capability_split/README.md) for input formats and pipeline details.

### 4. Synthesize tasks

Prepare `configs/mcp_pool.json` and a capability file, then run:

```bash
bash scripts/generate_tasks.sh
```

Common settings can be overridden with environment variables such as `COUNT`, `WORKERS`, `SEED_START`, `SYNTHESIS_MODEL`, `EVOSYNTH_WORLD_MODEL`, and `EVOSYNTH_OUTPUT_ROOT`.

### 5. Roll out and judge

Run the teacher model over generated tasks:

```bash
bash scripts/rollout_tasks.sh
```

The rollout entry point supports `--stage infer`, `--stage judge`, and `--stage all`. Completed results are skipped by default so interrupted runs can resume. Use `scripts/debug_infer_one_task.sh` for single-task debugging.

### 6. Run feedback and fill quotas

Provide normalized external evaluation records and run:

```bash
bash scripts/run_feedback.sh
```

To generate additional tasks until per-capability quotas are met:

```bash
bash scripts/fill_quota.sh
```

See [src/evosynth/feedback/README.md](src/evosynth/feedback/README.md) for the record schema.

## 📁 Project Structure

```text
EvoSynth/
├── src/evosynth/task_gen/       # Task synthesis, world construction, and validation
├── src/evosynth/task_infer/     # Teacher rollout, rewards, and judging
├── src/evosynth/feedback/       # Failure diagnosis and supply-demand auditing
├── src/evosynth/utils/          # API, configuration, message, and path utilities
├── capability_split/              # Benchmark capability extraction and grouping
├── prompts/                       # Synthesis, inference, audit, and MCP prompts
├── configs/mcp_pool.json          # MCP server and tool pool
├── scripts/                       # End-to-end and debugging scripts
├── data/                          # Persona and other input data
└── outputs/                       # Local generated artifacts
```

## 📦 Data Artifacts

Tasks are stored as JSON bundles containing the natural-language task, metadata, world references, and validation manifest. Rollout outputs retain the full conversation and tool-call trace. The feedback module consumes standard JSONL records from external evaluators, allowing EvoSynth to run the evaluate–diagnose–evolve–synthesize–train loop across existing benchmarks.
