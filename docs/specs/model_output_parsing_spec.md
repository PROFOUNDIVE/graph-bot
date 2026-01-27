# Model Output Parsing Spec (Reasoning-Friendly)

**Status:** Draft (v0)

This document defines a reasoning-friendly contract between:
- model raw outputs (which may include multi-line reasoning),
- parsing/extraction logic (which produces a task-specific candidate answer),
- and validators (which consume only the extracted candidate).

Motivation: for some tasks (e.g., Game of 24), the current implementation
effectively forces a single-line output by (a) prompt instructions and (b)
parsing only the first non-empty line. This spec describes an alternative that
does not require suppressing reasoning, while keeping validation deterministic.

References:
- `src/graph_bot/pipelines/stream_loop.py`
- `src/graph_bot/eval/validators.py`
- `docs/specs/amortization_io_contract.md`

## 1) Scope

In scope:
- How to extract a validator-ready answer from a raw model output that may
  contain reasoning.
- A recommended answer marker format that coexists with reasoning.
- Task-specific parsing rules for `task=game24`.

Out of scope:
- Changing validator semantics (e.g., what counts as a correct solution).
- Enforcing or discouraging chain-of-thought (CoT) at generation time.

## 2) Current Behavior (as of v0.2 stream)

Current Game24 stream behavior (implementation detail):
- Prompts instruct the model to output a single-line arithmetic expression.
- `_normalize_candidate_line()` selects the first non-empty line from
  `raw_output`.
- Precheck/validator run against that single selected line.

Practical implication: any multi-line reasoning causes the candidate to be the
first line of reasoning, not the final expression, increasing `format_error`.

## 3) Contract: Raw Output -> Candidate -> Validator

Definitions:
- `raw_output`: verbatim text returned by the model.
- `candidate`: the extracted answer string passed to the validator.

Contract rules:
1. Validators MUST receive `candidate`, not `raw_output`.
2. Parsers MUST be best-effort and deterministic: same `raw_output` yields the
   same `candidate`.
3. Parsers MUST be task-aware (at least by `task` name) because acceptable
   syntax differs per task.
4. Parsers MUST NOT require the model to suppress reasoning. Instead, they
   SHOULD provide a robust extraction method.

## 4) Recommended Answer Markers (Reasoning Allowed)

Recommended format (human-readable, easy to parse):

```text
<answer>
(8-5)*(11-2)
</answer>
```

Rules:
- The content inside the answer block is the primary extraction target.
- Reasoning may appear before/after the block.
- The extracted candidate is the first non-empty line within the block (after
  trimming whitespace).

Rationale:
- Markers decouple reasoning verbosity from extraction reliability.
- Multi-line blocks are allowed for future tasks, but Game24 typically uses one
  expression line.

## 5) Fallback Extraction (No Markers Present)

If no `<answer>...</answer>` block is present, the parser SHOULD attempt a
fallback extraction. For `task=game24`, recommended fallback strategy:

1. Split `raw_output` into trimmed lines and scan from bottom to top.
2. For each line, check whether it is a plausible Game24 expression (see
   Section 6). The first plausible line becomes `candidate`.
3. If none match, return empty candidate.

This fallback is designed to allow models that produce reasoning followed by a
final expression line without explicit markers.

## 6) Task-Specific Parsing Rules: Game24

Game24 expects an arithmetic expression that:
- Uses only digits, whitespace, parentheses, and `+ - * /`.
- Uses each provided number exactly once (as standalone number tokens; no
  digit concatenation).

Suggested checks (aligned with current implementation):
- Reject lines containing `=` or an explicit target marker like `-> 24` (some
  datasets use an arrow marker such as U+2192) to avoid "answer = 24" formats.
- Reject lines containing characters outside `[0-9\s\(\)\+\-\*/]`.
- Use the AST-based literal extractor (`extract_game24_expression_number_literals`) to
  validate number tokens.

Note: evaluation to 24 remains the validator's responsibility.

## 7) Logging Requirements

When using parsing/extraction:
- Always log:
  - `raw_output` (verbatim)
  - extracted `candidate`
  - extraction method (e.g., `answer_block`, `fallback_bottom_scan`, `empty`)

This preserves debuggability without constraining model reasoning.

## 8) Implementation Notes (Non-Normative)

Recommended code refactor:
- Replace `_normalize_candidate_line(raw_output)` with a task-aware
  `extract_candidate_answer(raw_output, task=..., allowed_numbers=...)`.
- Keep the validator API unchanged: `validator.validate(candidate, problem)`.

Potential migration strategy:
- Support both marker-based extraction and fallback scanning.
- Update prompts to *encourage* an `<answer>...</answer>` block rather than
  requiring a single-line output.
