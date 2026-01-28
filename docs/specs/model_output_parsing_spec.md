# Model Output Parsing Spec (v0.3 Programming-Based)

**Status:** Verified (v0.3)

This document defines the contract for v0.3 "programming-based" output parsing, which supports complex reasoning traces while extracting clean, machine-readable expressions for memory distillation and validation.

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

## 2) v0.3 Behavior: Programming-Based Extraction

As of v0.3, the system explicitly encourages multi-line reasoning (Chain-of-Thought) but requires a final "clean" expression for both validation and MetaGraph distillation.

Implementation detail (`stream_loop.py`):
- Prompts instruct the model to use `<answer>` tags for the final expression.
- `_normalize_candidate_line()` implements a multi-priority extraction strategy.
- `_distill_trace()` converts successful traces into a compact, normalized template format.

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

Two formats are supported, prioritizing explicit markers:

### Format A: Answer Block (Primary/Recommended)
An XML-like block. This is the most robust and preferred method. In v0.3, the system expects a **pure python expression** inside the tags (e.g., no `= 24` suffix).

```text
<answer>
(8-5)*(11-2)
</answer>
```

### Format B: GoT/CoT Style (Legacy/Alternative)
Supported for backward compatibility. The parser looks for the **last** line starting with `Output:`.

```text
Output: (10 - 4) * (13 - 9)
```

**Extraction Priority (v0.3):**
1. **Answer Block (`<answer>`)**: Extracts the first non-empty line within the tags. Strips any `=` suffix.
2. **GoT Style (`Output:`)**: Finds the *last* line starting with `Output:`. Strips prefix and any `=` suffix.
3. **Fallback (Bottom-Scan)**: Scans lines from bottom to top. The first line that passes the `_precheck_candidate` (see Section 6) is selected.

## 5) Distillation: Normalized Templates

v0.3 introduces **Distillation** to ensure the MetaGraph contains clean, reusable templates regardless of the original model's verbosity.

### Query Distillation (`_distill_query`)
Before retrieval, the user question is normalized:
- Input numbers are extracted and sorted.
- Format: `Solve 24 with <n1> <n2> <n3> <n4>`
- This ensures that `2 5 8 11` and `11 8 5 2` hit the same memory entries.

### Trace Distillation (`_distill_trace`)
After a successful validation, the trace is compressed into a 3-line template:
```text
Task: Game24
Input: 2 5 8 11
Solution: (8-5)*(11-2)
```
This distilled text is what gets stored in the `ReasoningNode.text` field for future few-shot retrieval.

## 6) Task-Specific Parsing Rules: Game24


Game24 expects a pure arithmetic expression. In v0.3, the following strict rules apply to the extracted `candidate`:

- **Strict No-Target Policy:** The candidate MUST NOT contain `=` or `->` or `→`. If these markers were present in the `raw_output`, the parser must strip them before passing to the validator.
- **Token Filtering:** Reject lines containing characters outside `[0-9\s\(\)\+\-\*/]`.
- **Literal Verification:** Use `extract_game24_expression_number_literals` to ensure only the allowed numbers are used.

Note: evaluation to 24 remains the validator's responsibility.

## 7) Logging Requirements

When using parsing/extraction:
- Always log:
  - `raw_output` (verbatim)
  - extracted `candidate`
  - extraction method (e.g., `answer_block`, `fallback_bottom_scan`, `empty`)

This preserves debuggability without constraining model reasoning.

## 8) Implementation Notes (Non-Normative)

**Recommended Code Structure:**
- Use a task-aware extraction function that implements the extraction priority in Section 4.
- `stream_loop.py` implements this by first checking for `<answer>` tags, then `Output:`, then falling back to a bottom-up line scan.

**Prompt Recommendation:**
Prompts SHOULD explicitly instruct the model to wrap the final result in `<answer>` tags. This minimizes extraction errors and allows the model to output reasoning before the final answer. In v0.3, we specifically ask for a **Python expression** that evaluates to 24.

Example prompt instruction:
> "You may show intermediate steps, but you MUST enclose the final Python expression in `<answer>` tags: `<answer>(1 + 2) * 8</answer>`"
