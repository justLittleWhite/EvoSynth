"""协议式 prompt。占位符 ``{{name}}``，用 :func:`render` 替换（不用 str.format，避免 JSON 花括号冲突）。"""
from __future__ import annotations

from typing import Mapping


def render(template: str, **values: object) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


DIAGNOSE = """ROLE
You are a capability diagnostician for an agent-training pipeline. You analyze a failed benchmark trajectory to identify which capability archetype failed and what finer-grained sub-capability the task required.

TASK
Given a benchmark task where the agent scored 0, inspect the task prompt, available tools, the agent's trajectory, and optional evaluator feedback. Identify the first critical failure point, map the task to the closest capability archetype from the pool, and abstract the specific sub-capability required by this task.

INPUT
TASK PROMPT: {{benchmark_question}}
AVAILABLE TOOLS: {{benchmark_tools}}
AGENT TRAJECTORY: {{model_trajectory}}
EVALUATOR FEEDBACK (optional, may be "N/A"): {{eval_reason}}
ARCHETYPE POOL:
{{archetype_pool_json}}

GUIDANCE
- If EVALUATOR FEEDBACK is provided, use it to anchor where the final answer went wrong, then trace backward into AGENT TRAJECTORY to find the root cause. Set blind_signal to null.
- If EVALUATOR FEEDBACK is "N/A", audit the trajectory against TASK PROMPT and AVAILABLE TOOLS using four objective signals and set blind_signal to the first one that applies:
  constraint_omission: a condition required by the task prompt is never verified through a tool or computed anywhere in the trajectory.
  tool_anomaly: a tool error is not retried, an empty result is followed by invented content, or an argument contains an identifier that never appeared in the context.
  premature_stop: AVAILABLE TOOLS include a tool that could verify or complete the missing information, yet the agent concluded early.
  synthesis_error: the observations are correct, but the final answer's calculation or condition filtering conflicts with them.
- Map the task to ARCHETYPE POOL:
  - Set archetype_fit to "exact" if the base archetype's description already captures the nature of this task and the agent simply made an execution mistake.
  - Set archetype_fit to "too_broad" if the task falls under this archetype generally, but requires a more specific sub-capability subtype (e.g., conditional aggregation, multi-attribute constraint satisfaction, cross-record reconciliation) not distinguished by the broad base description.
  - Set archetype_fit to "missing" if no archetype in the pool covers this kind of task at all.
- SUB-CAPABILITY ABSTRACTION (for structural_gap and missing_verify_rule):
  - State WHAT finer-grained capability subtype this task represents (what relational structure between records, attributes, constraints, and target output is needed), NEVER how the agent should operate step by step.
  - Strip BOTH domain nouns AND medium-specific traits (spatial/geometric terms like path, route, distance, coordinate, segment, waypoint; temporal/series terms like time-window, schedule, chronological, interval; system/automation terms like webhook, pipeline, async, endpoint, schema). Express the gap purely in terms of entities, attributes, conditions, mappings, and calculations.
  - Self-check: the wording must make complete sense if the entire task were carried out over plain relational tables of items and attributes.
  - Examples of medium-free abstraction:
    - BAD: "selecting intermediate points along a route that satisfy distance and proximity thresholds" -> GOOD: "selecting a subset of candidate items that simultaneously satisfy multiple quantitative thresholds relative to a reference anchor"
    - BAD: "reconciling time-series intervals and rate tiers across billing periods" -> GOOD: "computing a composite metric where different segments of a quantity are governed by different lookup rules"
    - BAD: "propagating repository status updates into a downstream notification channel" -> GOOD: "updating a target record so its state reflects an aggregated status derived from multiple source records"

OUTPUT FORMAT
<output>
{
  "matched_archetype": "archetype_name_or_null",
  "archetype_fit": "exact | too_broad | missing",
  "failure_stage": "tool_routing | argument_grounding | incomplete_exploration | state_recovery | computation_and_synthesis",
  "blind_signal": "constraint_omission | tool_anomaly | premature_stop | synthesis_error | null",
  "first_error_turn": 2,
  "structural_gap": "one short sentence naming the domain-agnostic sub-capability required by this task",
  "missing_verify_rule": "one short clause stating what observable relationship must hold between the tool observations and the final answer for this sub-capability"
}
</output>"""


CLUSTER_GAPS = """ROLE
You group de-contextualized capability gaps into coherent sub-capability clusters.

TASK
Each item below describes a sub-capability gap observed on benchmark failures. Group items that represent the same underlying sub-capability. Every item index must appear in exactly one cluster. Create at most {{max_clusters}} clusters; merge minor items into the closest main cluster.

INPUT
ITEMS:
{{items_json}}

OUTPUT FORMAT
<output>
{"clusters": [{"label": "short_sub_capability_tag", "members": [0, 3]}]}
</output>"""


# 与旧版的差别只有一处：不再规定「只生成一个变体」，由模型按数据决定 1..max_variants 个。
EVOLVE = """ROLE
You are the curriculum architect for a synthetic agent-training pipeline. You specialize a broad capability archetype into finer-grained sub-capabilities based on benchmark failure patterns.

TASK
Given a BASE ARCHETYPE and a set of failure observations under it, decide from the data how many distinct finer-grained sub-capabilities the observations support (between 1 and {{max_variants}}), and define one variant archetype for each. Every variant sits inside the base archetype.

INPUT
BASE ARCHETYPE:
{{base_archetype_json}}

EXISTING VARIANTS OF THIS BASE (do not duplicate):
{{existing_variants_json}}

OBSERVED FAILURE CLUSTER:
{{aggregated_structural_gaps_and_verify_rules}}

GUIDANCE
- A variant is a finer-grained capability subtype of the base archetype defined by its RELATIONAL PATTERN (how records, attributes, constraints, and outputs relate), NOT by the medium or data type of the source tasks.
- Decide the NUMBER of variants yourself from OBSERVED FAILURE CLUSTER (between 1 and {{max_variants}}). Observations that share the same underlying relational pattern become ONE variant even if they came from different domains or media (e.g., spatial routes, time schedules, financial ledgers). Observations that require a genuinely different relational structure between facts and deliverable become SEPARATE variants. Never split by surface medium, and never invent a variant to fill {{max_variants}}.
- The observations may already be grouped (CLUSTER 1, CLUSTER 2, ...). Groups are a hint, not a requirement: merge groups that share the same relational pattern, keep them separate when their relational patterns differ.
- Variants must differ clearly from each other and from EXISTING VARIANTS.
- MEDIUM-FREE ABSTRACTION (critical):
  - Even if the input observations mention medium-specific words (spatial/geometric: path, route, waypoint, distance, coordinate, segment, interpolation, proximity; temporal: schedule, time-window, chronological, period; system/infra: pipeline, webhook, repo, endpoint, channel), you MUST translate them into universal relational terms (entities, candidate sets, attributes, reference anchors, quantitative bounds, lookup rules, state transitions).
  - Plain-table self-check: before writing a variant, verify that its name, "archetype_desc", "verify", and "world_init_hint" would make 100% literal sense for a hospital administrator, a school registrar, a warehouse manager, and a financial auditor working only with plain records and attributes. If a word only makes sense on a map, a timeline, or a code repo, abstract it further.
  - Calibration examples (BAD -> GOOD):
    - BAD: "entity_discovery__path_point_selection" (selecting intermediate points along a path meeting distance constraints)
      GOOD: "entity_discovery__multi_criteria_filtering" (identifying candidate entities that simultaneously satisfy multiple quantitative bounds relative to a reference entity)
    - BAD: "derived_computation__tiered_window_calculation" (calculating metrics across sequential time windows and rate tiers)
      GOOD: "derived_computation__piecewise_rule_aggregation" (computing an aggregate metric where each record's contribution depends on which condition tier or lookup bracket its attributes fall into)
    - BAD: "cross_system_transfer__pipeline_event_propagation" (forwarding status payloads across service endpoints)
      GOOD: "cross_system_transfer__derived_state_synchronization" (updating a target record in a second system using a consolidated status derived from multiple source records)
- Match the exact tone, brevity, and style of BASE ARCHETYPE, for EACH variant:
  - "archetype": "<base_archetype>__<sub_capability_tag>" (2-3 lowercase words after the double underscore, naming the relational capability pattern).
  - "archetype_desc": 1-2 short sentences (at most 350 characters) stating what result or plan this person wants and how its parts depend on the retrieved facts. State what the task IS, never step-by-step instructions on how the agent should search or call tools.
  - "verify": 2-3 short clauses separated by semicolons (at most 450 characters) stating what must be true of the tool observations and the answer.
- Keep every variant completely domain-agnostic and medium-agnostic so it pairs naturally with any persona and MCP server.

OUTPUT FORMAT
<output>
{
  "variants": [
    {
      "archetype": "base_name__sub_capability",
      "archetype_desc": "...",
      "verify": "...",
      "world_init_hint": "one short sentence on what entity structure this sub-capability needs"
    }
  ]
}
</output>"""


NEW_ARCHETYPE = """ROLE
You are the curriculum architect for a synthetic agent-training pipeline. You design a new capability archetype for a capability dimension that no existing archetype in the pool covers.

TASK
Given a cluster of capability gaps that the current archetype pool cannot describe, define ONE new capability archetype. It must match the style and granularity of the existing pool.

INPUT
CURRENT ARCHETYPE POOL (names and descriptions):
{{archetype_pool_json}}

OBSERVED FAILURE CLUSTER (tag: {{cluster_label}}):
{{aggregated_structural_gaps_and_verify_rules}}

GUIDANCE
- Keep the same three-field schema: "archetype", "archetype_desc", and "verify".
- Name it with a short lowercase snake_case name (2-3 words, no double underscore) that is distinct from every archetype in the pool.
- Define the capability by its RELATIONAL PATTERN (how records, attributes, constraints, and outputs relate), stripping all domain nouns and medium-specific traits (spatial/geometric, temporal, or system/pipeline terms). Verify that it makes 100% literal sense over plain relational tables of records and attributes across any persona.
- Match the exact tone, brevity, and style of CURRENT ARCHETYPE POOL:
  - "archetype_desc": 1-2 short sentences (at most 350 characters) stating what kind of result, plan, or action this person wants and how it depends on retrieved facts. Never write a solving procedure.
  - "verify": 2-3 short clauses separated by semicolons (at most 450 characters) stating observable properties that must hold between the tool observations and the final answer.
- Keep it domain-agnostic and medium-agnostic so it works with any persona and MCP server.

OUTPUT FORMAT
<output>
{
  "diagnosis_type": "new_dimension",
  "recommended_sampling_multiplier": 2.0,
  "world_init_hint": "one short sentence on what entity structure this archetype needs",
  "targeted_archetype": {
    "archetype": "new_snake_case_name",
    "archetype_desc": "...",
    "verify": "..."
  }
}
</output>"""


def pool_for_prompt(entries: list[Mapping[str, object]]) -> list[dict[str, str]]:
    return [{"archetype": str(e["archetype"]), "archetype_desc": str(e["archetype_desc"]),
             "verify": str(e.get("verify") or "")} for e in entries]
