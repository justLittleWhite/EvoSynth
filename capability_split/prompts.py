"""Prompts for two-stage benchmark capability extraction.

Stage 1 decomposes each task into independent structural parts. Stage 2 groups
those parts into reusable, domain-independent capability archetypes and merges
near-duplicates across batches. A task may contribute more than one part.
"""

# ── ANDES 五原则（已与本项目对齐） ─────────────────────────────────────────

ANDES_PRINCIPLES = """\
1. Macro-level: describe capability dimensions, structural requirements and constraint \
types; never lock onto specific problems or include example questions.
2. Real differences only: differences in task structure, reasoning chain and constraint \
combination count; difficulty alone is not a capability boundary.
3. Benchmark-grounded: stay faithful to what the benchmark actually requires, and keep \
descriptions substantive rather than vague.
4. Each capability is a broad, fundamental cognitive operation with its own reasoning \
structure, NOT a narrow knowledge point. Differentiate by operation and reasoning \
structure, never by surface topic."""

# ── ANDES 失败信号（用于阶段 2 自检） ──────────────────────────────────────

ANDES_FAILURE_SIGNALS = """\
F1 Description too narrow (revolves around very few structures)
F2 Flashy surface, collapsed logic (no substantive diversity)
F3 Template lock-in (near-duplicate capabilities)
F4 Coverage collapse (benchmark capabilities left uncovered)
F5 Constraint drift (deviates from benchmark requirements)
F6 Form without substance (low training value)
F7 Description growing emptier (semantically loose)"""

# ── 阶段 1：单条任务 → 若干"部分"的结构签名 ────────────────────────────────

SIGNATURE_PROMPT = """\
You are abstracting one agent-benchmark task into structural signatures. The signatures of many
tasks will later be compared with each other, so they must carry the structure of the task and
nothing about its subject matter.

TASK:
{task}

Step 1, parts. A task may consist of several parts that can be done separately, with no
information flowing from one to another, each producing its own reasoning product. Treat
each such part separately. If the stages of the task feed one another, it is a single part.

Step 2, signature of each part. Separate the reasoning product from its enactment. Give every
data object involved in determining the reasoning product a role label (R1, R2, ...) defined
only by the function it plays (a given input, something to be inferred, a set of candidates, a
condition, a source of evidence, the reasoning product); do not assign roles to external
destinations or execution tools. Fill these fields, each as one or two sentences:
- roles: the role labels and the function of each;
- dependency: how the pieces of evidence must be connected to determine the reasoning product;
- operation: what is done with the evidence beyond obtaining it, that is, what would be missing
  if the reasoning product could simply be read off one piece of evidence;
- decision_output: the complete information determined by the operation, and what must hold
  between it and the roles;
- external_effect: what the original task does with that information afterward, or "none";
- shortcut: the simplest plausible way to skip that operation, and why its decision_output would
  be wrong.

Write each value first in your own words, then rewrite it so that no name of a real-world object,
medium (including spatial, geometric or temporal framing), kind of quantity or field of activity
remains, only role labels and functional words, and return only the rewritten version.

Return ONLY one JSON object with exactly this form:
{{"parts": [{{"roles": "...", "dependency": "...", "operation": "...", "decision_output": "...", "external_effect": "...", "shortcut": "..."}}]}}"""

# ── 阶段 2：part 签名 → archetype ──────────────────────────────────────────

SPLIT_PROMPT = """\
You are deriving reusable capability archetypes from the structural signatures of task parts in
an agent benchmark. The archetypes will be used as sampling axes for generating new synthetic
tasks and for checking the answers to them.

BENCHMARK: {benchmark}

PRINCIPLES:
{principles}

SIGNATURES ({n_parts} total; each describes one independent part of a task; the number in
brackets is the global part id):
{signatures}

{scope_note}

An archetype is a capability: it is defined by what is done to the roles, never by what the roles
are in the world. Its name, description and verify must therefore stay valid when the subject
matter, the medium or the kind of quantity is replaced by any other, and are written in
role-level words.

Group the signatures with one criterion that follows from what an archetype is for: two parts
belong to different archetypes exactly when generating a new instance would require connecting
the evidence differently (dependency) or applying a different essential operation. Subject matter,
kind of object, number of steps or conditions, difficulty, and the way the result is finally
checked are not boundaries by themselves; how a result is checked follows from the dependency and
the operation. An archetype is a reusable axis, so it must be supported by at least two
signatures unless the benchmark is too small for that.

Assign every signature id in this list to exactly one archetype. Do not invent ids that are not
listed, and do not leave ids out.

For each archetype give exactly:
- archetype: a snake_case name derived from the shared operation and dependency of its
  signatures;
- archetype_desc: one or two sentences saying what is given, what is done with it, and what is
  produced;
- verify: one sentence stating what must hold of a correct result, drawing on the shortcut fields
  of its signatures so that the shortcut fails it;
- member_part_ids: the global part ids (the bracket numbers) of every signature in this archetype.
  At least two ids.

No examples or lists of variants in the prose fields.

Before answering, check the final set against these failure signals:
{failure_signals}

Return ONLY a JSON array. Every object must contain exactly these four keys:
[{{"archetype": "...", "archetype_desc": "...", "verify": "...", "member_part_ids": [1, 2]}}]"""

# ── 阶段 2b：候选 archetype 近重复合并 ─────────────────────────────────────

MERGE_PROMPT = """\
You are merging candidate capability archetypes induced from separate batches of the same
agent benchmark. The archetypes will be used as sampling axes for generating new synthetic
tasks and for checking the answers to them.

BENCHMARK: {benchmark}

PRINCIPLES:
{principles}

CANDIDATES ({n_candidates} total). Each member_part_ids list holds global part ids already
assigned in that batch:
{candidates}

Merge candidates that share the same way of connecting evidence (dependency) and the same
essential operation. Keep them apart when generating a new instance would require connecting
the evidence differently or applying a different essential operation. Subject matter is not a
boundary.

When you merge, member_part_ids is the sorted union of the merged candidates' ids. Every id
that appears in the candidates must appear in exactly one output archetype. Do not invent ids.

For each merged archetype give exactly:
- archetype: a snake_case name derived from the shared operation and dependency;
- archetype_desc: one or two sentences saying what is given, what is done with it, and what is
  produced, still valid if the subject matter is replaced;
- verify: one sentence stating what must hold of a correct result so that the shortcut fails it;
- member_part_ids: the union of ids, at least two.

Before answering, check the final set against these failure signals:
{failure_signals}

Return ONLY a JSON array. Every object must contain exactly these four keys:
[{{"archetype": "...", "archetype_desc": "...", "verify": "...", "member_part_ids": [1, 2]}}]"""

# ── 覆盖率不足时，只把未覆盖的 part 补进已有 archetype ─────────────────────

COVER_PROMPT = """\
Some structural signatures were not assigned to any capability archetype. Attach each uncovered
signature to an existing archetype when its dependency and essential operation match, or create
a new archetype when they do not. A new archetype needs at least two member_part_ids from the
uncovered list. A leftover signature that matches an existing archetype must reuse that
archetype name instead of standing alone.

BENCHMARK: {benchmark}

EXISTING ARCHETYPES:
{archetypes}

UNCOVERED SIGNATURES ({n_parts} total; the number in brackets is the global part id):
{signatures}

member_part_ids may contain only ids from the uncovered list. Every uncovered id must appear
in exactly one returned archetype.

Return ONLY a JSON array. Every object must contain exactly these four keys:
[{{"archetype": "...", "archetype_desc": "...", "verify": "...", "member_part_ids": [1, 2]}}]"""
