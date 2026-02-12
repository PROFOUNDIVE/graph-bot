# BoT Alignment: Meta Reasoner In Solve&Instantiate (v0.5)

## Background

BoT's intended per-query loop (as used in `~/git/buffer-of-thought-llm/`) assumes a fixed set of stages:

1. Input query
2. Distill problem (LLM, +1 call)
3. Retrieve from MetaBuffer/MetaGraph (RAG/LLM, +1 call)
4. Solve & instantiate (LLM, +1 call)
5. Distill the solution/thought template (LLM, +1 call)
6. (Optional) Similarity gate vs existing templates
7. Update MetaBuffer/MetaGraph

In this framing, Solve&Instantiate must remain a single LLM call. Adding a separate instantiation call would increase cost beyond the minimal 4-call baseline.

## Change Summary

We embed the BoT "Meta Reasoner" prompt verbatim into the *existing* solver prompt for Graph-BoT mode (non-`io`/`cot`), so instantiation is performed inside the single solve call.

Task-specific output constraints remain enforced by appending the existing task solver instructions after the Meta Reasoner text.

## Code Changes

- `src/graph_bot/tasks/game24.py`
  - For non-`io`/`cot` modes, `build_solver_prompt()` now uses:
    - `system = META_REASONER_SYSTEM + "\n\n" + got_cot_system`
    - `user` still injects `Retrieved templates/context:` via `retrieval.concatenated_context`

- `src/graph_bot/tasks/wordsorting.py`
  - For non-`io`/`cot` modes, `build_solver_prompt()` now uses:
    - `system = META_REASONER_SYSTEM + "\n\n" + cot_system`

- `src/graph_bot/tasks/mgsm.py`
  - For non-`io`/`cot` modes, `build_solver_prompt()` now prefixes its existing `system` with `META_REASONER_SYSTEM`.

## Expected Runtime Behavior

For a single query in Graph-BoT mode:

- Distill problem: handled by the distiller (LLM-based in `--distiller-mode llm`).
- Retrieve: `adapter.retrieve_paths(...)` returns `retrieval.concatenated_context`.
- Solve&Instantiate: one LLM call using a system prompt that includes the Meta Reasoner text and a user prompt that includes `Retrieved templates/context:`.
- Distill template: one LLM call in `distiller.distill_trace(...)`.

No additional instantiation call is introduced in the pipeline.

## Future Work

- Dense similarity gate (BoT "threshold" validation) prior to MetaGraph updates.
  - Current GraphRAG adapter behavior is oriented around canonical-key deduplication.
  - Adding a dense similarity threshold gate should be planned alongside the adapter's "Basic mode" documentation alignment, so this is intentionally deferred.
