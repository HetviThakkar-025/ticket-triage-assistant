# Development Notes

How this project was built with Claude Code — the decisions that shaped it, the
things that went wrong, and why the code looks the way it does.

---

## 1. Design before code

The first prompt described the full brief and ended with: *"Before writing any
code: walk me through your proposed architecture — file-by-file
responsibilities, the JWT flow, the chunking/retrieval approach, and the decision
prompt design. Wait for my confirmation."*

That constraint was worth more than any later prompt. Exploring the repo before
proposing anything surfaced three facts that changed the design:

1. **The action vocabulary was discoverable, not invented.** `data/tickets.csv`
   uses exactly 15 distinct `resolved_action` values. Reading them out of the CSV
   (`awk -F, '{print $NF}' | sort | uniq -c`) fixed the `Action` enum before a
   line of code existed. Guessing action names would have failed the evaluation
   no matter how good the reasoning was. Note the distinction from the rule in
   `DATA_NOTES.md`: the CSV defines the *output vocabulary*, but is never
   consulted at decision time — the policy files decide.

2. **The policy files are numbered rules.** Each of the six files is a list of
   4–5 atomic policy points. That made the chunking decision obvious: one
   numbered point per chunk, rather than a fixed character window that would cut
   a threshold away from its condition.

3. **The sample messages are self-contained.** Every `message` in
   `sample_test_cases.json` states the facts its decision needs, so the pipeline
   takes the message alone and the structured columns stay as an answer key.
   `POST /tickets` accepts only `message`, and `tests/evaluate.py` sends only
   `message`.

Three questions were genuinely open, so they were asked rather than assumed:
the embedding backend, 403-vs-404 for another user's ticket, and whether to run
the live evaluation. All three were answered before implementation began.

## 2. Decisions and their reasoning

**Embeddings had to be pluggable.** The machine runs Python 3.14, which
`sentence-transformers`/torch do not yet ship wheels for. Rather than quietly
substituting something, the constraint was raised with two options. The agreed
design was an `Embedder` interface with three implementations — Gemini (default),
sentence-transformers (if importable), and a pure-numpy TF-IDF fallback — chosen
by `build_embedder()` and degrading rather than crashing. The side benefit
decided the matter: the TF-IDF backend makes the whole test suite run offline
with no API key and no quota.

The implementation still needed a Python 3.12 virtualenv (`uv venv --python
3.12`) because several pinned dependencies lack 3.14 wheels.

**404, not 403, for another user's ticket.** A 403 confirms the row exists and
leaks the ID space to anyone enumerating integers. Ownership is enforced in SQL
(`WHERE t.id = ? AND t.user_id = ?`), not by fetching then comparing, so the
"not mine" and "not there" paths are the same code path and cannot drift apart.

**The prompt describes actions without restating thresholds.** `ACTION_GUIDE`
explains when each of the 15 actions applies in policy-neutral language
("damaged order where the policy requires photographs before approval") and
deliberately omits every number. The ₹2,000 and 7-day thresholds must come from
the retrieved excerpts, otherwise the RAG layer is decorative and editing a
policy file would not change behaviour.

**The missing-information rule is the load-bearing instruction.** The prompt
lists the six facts to extract, states that each is either present or `UNKNOWN`,
and forbids substituting a default. Sample case S05 ("I want to return this.")
only produces `NEEDS_MORE_INFORMATION` because of that rule — without it the
model happily assumes an unopened non-food item inside the window.

**Citations are filtered, not trusted.** `parse_decision()` intersects the
model's `sources` with the filenames that were actually retrieved, so a
hallucinated citation is dropped rather than persisted. A test covers this.

**`top_k = 8` out of 29 chunks.** On a corpus this small, recall beats precision:
several tickets are governed by rules in two files (a damaged-goods ticket also
needs the returns policy's "damaged items are handled elsewhere" rule). Retrieval
only has to avoid dropping the governing rule; the model does the filtering.

## 3. What went wrong, and what it changed

**The model names in the brief did not exist on this key.** Rather than assume,
`genai.list_models()` was called first. Neither `gemini-2.0-flash` nor
`text-embedding-004` was available on this project. Model selection then took
three rounds, all driven by what the API actually returned:

| Model | Outcome |
| --- | --- |
| `gemini-2.5-flash` | Worked — scored 5/5 — then hit its daily cap |
| `gemini-2.5-flash-lite` | HTTP 404: *"no longer available to new users"* |
| `gemini-3.5-flash` | **Chosen default.** 5/5, and newer than the brief's suggestion |

The free tier on this project allows **20 requests per day, per model**
(`quota_id: GenerateRequestsPerDayPerProjectPerModel-FreeTier`,
`quota_value: 20`). That is roughly four evaluation runs per model per day, and
it is the reason the README explains how to tell a quota failure from a
reasoning failure.

`gemini-embedding-001` is the embedding default. Both are overridable via `.env`.

**A parallel evaluation reported 26.7% accuracy — and it was measuring the wrong
thing.** A spot-check ran 30 held-out historical tickets through the pipeline
with six worker threads, and nearly all of them came back
`NEEDS_MORE_INFORMATION`. Re-running two of the failures sequentially produced
correct, confidently-reasoned answers. The cause was Gemini's free-tier
rate limit: HTTP 429 exhausted both attempts and surfaced as the safe fallback,
which is indistinguishable from a genuine "insufficient information" verdict
unless you read the confidence (`0.0`) and the reason string. The same trap
caught a later run at 20% — by then the model's *daily* quota was gone, not just
its per-minute allowance.

A broader generalisation check (held-out historical tickets, with their
structured columns restated in the message so that rule application could be
measured separately from fact extraction) could not be completed today: the
daily quota ran out partway through. Individually re-run cases were decided
correctly and confidently — for example `REJECT_OPENED_ITEM` at 0.95 and
`OPEN_SHIPPING_INVESTIGATION` at 0.90 — but no honest aggregate number can be
reported for it, so none is claimed here. The scripted evaluation over
`sample_test_cases.json` is the verified result.

That produced a real fix rather than just a corrected number. The retry had no
backoff, so a rate-limited call burned its retry immediately. `_backoff_for()`
now parses the `retry_delay { seconds: N }` that Gemini returns on a 429 and
waits for it (capped at 30s) before retrying. The behaviour is documented in the
README so nobody else reads a 429 as a reasoning failure.

**The lexical fallback needed stemming.** TF-IDF retrieval initially missed on
"delivered" vs "delivery" vs "deliver". A small suffix stripper in `_stem()`
unified them; the retrieval tests assert the governing policy is retrieved for
all six issue types using this backend.

## 4. Verification

Nothing was reported as working without being run:

- `pytest -q` → **32 passed**, fully offline.
- `python tests/evaluate.py` against a live `uvicorn` server → **5 test cases /
  Correct: 5 / Accuracy: 100.0%**, with every decision citing the correct policy
  file. Confirmed on two different models (`gemini-2.5-flash`, then the
  `gemini-3.5-flash` default).
- The full HTTP contract was exercised end to end against the running server:
  register → duplicate register (409) → bad login (401) → login → `/me` →
  `POST /tickets` → `GET /tickets` → `GET /tickets/{id}` (200 for the owner, 404
  for another user, 401 unauthenticated).
- `streamlit run streamlit_app.py` was launched headless and checked for a
  healthy response before being reported as working.
- Retrieval was inspected directly (printing top-k chunks with scores per query)
  for both the Gemini and TF-IDF backends before the decision layer was written,
  so a later failure could not be ambiguous between retrieval and reasoning.

## 5. Working practices that mattered

- **Read the data before designing.** The enum, the chunk boundary, and the
  input contract all came from ten minutes of reading the supplied files.
- **Verify environment claims.** Model availability, wheel availability, and
  library behaviour (passlib/bcrypt) were checked by running them, not assumed.
- **Treat a surprising metric as a bug in the measurement first.** The 26.7%
  result was a rate limit, and chasing it as a prompt-quality problem would have
  meant rewriting a prompt that was already correct.
- **Keep secrets and generated artefacts out of git.** `.env`, `.cache/`,
  `*.db`, and `__pycache__` are ignored; `.env.example` documents every variable.
