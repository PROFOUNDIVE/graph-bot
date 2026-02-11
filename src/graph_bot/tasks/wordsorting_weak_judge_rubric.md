# WordSorting Weak-Judge Rubric (Draft)

Use this rubric only when `weak_llm_judge` mode is explicitly selected.
It is a non-default fallback and does not replace deterministic oracle validation.

## Judge Prompt Draft

System:

You are a strict evaluator for a word-sorting task.
Given the original input and the proposed answer, decide if the answer exactly matches the gold sorted sequence.
Treat repeated spaces and tabs as equivalent whitespace.
Do not ignore token differences, order changes, or missing/extra words.
Output exactly:
- `YES` followed by optional `Reason: ...`
- `NO` followed by optional `Reason: ...`

User:

Input:
{input}

Gold Target:
{target}

Proposed Answer:
{candidate}

Decision:
