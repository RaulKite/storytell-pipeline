# T16 `stories` — narrative-window detection with the pipeline's own LLM

Branch: `feat/stories` (from `feat/elan-basics` — it already contains `master`, and its
coverage inventory derives from `ARTIFACT_LAYOUT`, so a new artifact key entered on master
would hit `COVERAGE_REASON_UNKNOWN` against elan-era code; branching here keeps one line
of development until the operator merges).
Requested by the operator: "usa el mismo LLM que usas para traducir para intentar
implementar dónde y cómo detectar historias en la transcripción."
This is ODD §20.3 becoming code. §20.3 said: prototype first, small. That was done —
see "Probe" below — and the schema below is shaped by what the endpoint actually said.

## Probe measurements (2026-10-01, `/tmp/probe_stories.py`, raw under `/tmp/stories-probe/`)

Endpoint: the translation endpoint itself (the configured `LITELLM_BASE_URL`, model `chat`),
key from `.env`. 2 runs × 3 real clips, temperature 0:

| clip | transcript | stories run1 | stories run2 | identical | schema-invalid |
|---|---|---|---|---|---|
| KABC (2 segs) | "hearing your voice at the Laker game / I scream so much" | 1 | 1 | yes | 0 |
| CNN (1 seg) | intro announcement | 0 + reason | 0 + reason | yes | 0 |
| La-1 (4 segs) | "Muy buena entrada… vamos a retroceder a la infancia… tenemos una foto" | 0 + reason | 0 + reason | yes | 0 |

- The prompt's "empty is the correct answer" rule **held**: 2 of 3 clips came back empty
  with a stated `no_story_reason`, unprompted by any follow-up. That is the §20.3
  requirement ("the output must be allowed to be empty") demonstrated on this endpoint.
- Temperature 0 was byte-identical across runs on all three. Not relied upon as a
  guarantee — the request digest caches instead — but it means dev loops are cheap.
- KABC's one story cited only real `segment_id`s and used real segment boundary times;
  `confidence` 0.9. The validating parser the stage needs is therefore a real rejection
  path (invented id / fake boundary ⇒ reject the batch), not decoration.
- Cost: 3.674 tokens for all six calls (~600 prompt + ~70 completion per call on these
  1-4-segment transcripts). One request per video is affordable at this scale.
- Judgment on KABC is arguably generous ("I scream so much" as story closure). That is
  the nature of the task; the design keeps the model's `why_it_is_a_story` and
  `confidence` on every row so a human can disagree traceably.

## Design decisions (and the §20.3 point each answers)

1. **New stage `stories`, one LLM request per video, transcript rendered with
   segment_id + speaker + start/end + the speaker turns beside it.** §20.3: boundaries
   are referential, so the model must see timestamps and speakers, not plain text.
   Long transcripts are windowed by `max_segments_per_request` (default 60): only
   stories whose evidence lies entirely inside one window are kept, windows overlap by
   `context_segments`, and the limit itself is recorded — a clipped forest is a known
   gap, not a silent one.
2. **Forest, not a list: `parent_id` on every row.** §20.3: stories nest; a flat
   interval list cannot hold the answer. Validation rejects a `parent_id` that is not a
   story in the same batch or an ancestor cycle.
3. **Empty is a first-class result.** Table has zero rows; the raw sidecar keeps
   `no_story_reason`; the stage logs and the summary counts empty-vs-nonempty per
   dataset, which is the "something must count how often it is empty" counter.
4. **Rejection path, not partial acceptance.** Response must be a JSON object; every
   evidence id must exist in the requested transcript; every boundary must equal a real
   segment start/end; confidence ∈ [0,1]; parent must exist and be acyclic. Any breach
   raises, retries with backoff like translation, and fails the stage loudly after
   `max_retries` — a silently dropped story is the lie, a failed stage is fixable.
5. **Digest-cache like translation**: key = prompt_version + model + temperature +
   window content (segment ids, texts, speaker ids, times). Endpoint/model/prompt change
   ⇒ new key ⇒ honest rerun. `segments_digest` (sha256 of `speech_segments.parquet`)
   rides in `config_fingerprint`, so transcript changes invalidate reuse.
6. **Same endpoint ≠ same config.** `StoriesConfig` is its own section (own prompt
   version, own batching) but defaults its `base_url`/`api_key`/`model` to the same
   `${LITELLM_*}` env, so "the same LLM as translation" is the no-config answer.
7. **Outputs**: `stories/stories.parquet` (STORIES_SCHEMA) +
   `stories/raw/stories_<window>.json` preserving the raw model text byte-identical
   (repo rule: raw next to normalized). Coverage inventory picks the new artifact keys
   up automatically; they enter as `present, not exported` with a named reason —
   exporting an ELAN tier for stories is a follow-up, not part of this unit.
8. **Mock provider** mirrors translation's: deterministic offline stories so e2e and
   unit tests never need the network.

## Tasks

- [x] S16a Probe the endpoint, choose the schema from its answers. Raw responses under
      `/tmp/stories-probe/`; probe script `/tmp/probe_stories.py`.
- [x] S16b Implement the stage. Writer: 98 new red-first tests, mutation-checked rejection
      guards. Parent closed the 8 out-of-surface drift failures the writer honestly
      reported (test_elan ×3, test_elan_stage ×3, test_readme_claims ×2: registry 44→46,
      tables 22→23, `absent` became a *real* state on disk for the two speechless datasets
      so the "no artifact is absent" guard was rewritten to assert the exact partition,
      README prose rewritten + the verbatim block regenerated by running the README's own
      snippet, collection ratchet 1857→1957). Full unit 1949 passed / 8 skipped; e2e 42
      passed; pyflakes clean.
- [x] S16d (run before the docs unit) `stories:` section appended to
      `config/config.local.yaml` (backup `/tmp/config.local.yaml.bak3`); env-resolved
      endpoint confirmed (model `chat`, key length 25, never logged). Real corpus run:
      5 endpoint calls, 2.992 tokens, ~330 ms each. KABC 1 story (`w0-s1`, 0.071–4.051 s,
      conf 0.9, "Recalling hearing a voice at a Laker game"); CNN 0 and La-1 0 with stated
      reasons; `person_demo` and `pipeline_silent` skipped with an explicit reason (no
      speech segments) — which is why the ELAN coverage inventory now legitimately shows
      `absent` on disk. Elan refresh: 23 inventoried tables, 17 exported / 2 summarised /
      4 present-not-exported (+1 absent on the two), validate ok 7/7. Hash diff: +27
      files, 0 measured tables changed — all additions are the new stories outputs, all
      changes are logs/status/manifest/provenance.
- [x] S16c Independent verification of `9f39903`: done, 4 defects found and fixed in
      `8f41689` (see §S16c below, which is the record of it, and the parent's integrity
      confirmation of the verifier's own run).
- [ ] **Still open, split out of S16c:** a descriptive README section for the stage. The
      count guards are honest; the prose tour of `stories/` that every other stage has is
      not written, and the README currently reveals the stage only through one artifact row.

### Parent spot-check of the writer's review_focus (2026-10-01)

Direct probes against `validate_stories`: invented evidence id refused; empty-without-reason
refused; parent-not-in-answer refused; start≥end refused; empty-with-reason accepted. Two
acceptances were mine expecting too much, both tested decisions: `confidence: "0.9"` coerces
because the code says a model answering `true` gives a non-number, not a 1; `start_time
0.0714` is accepted because the real source start rounds onto it at the ms grid — the
fabricated-boundary refusal is tested with genuinely fake times (4.444).

### Writer deviations the parent accepted
1. A story whose parent was dropped cross-window is dropped+counted, not a batch rejection
   (temperature 0 repeats the same answer; rejecting would retry until the video lost every
   story). Counted gap, tested.
2. Retries live in the client, tested against a real local HTTP server (5xx retried with
   backoff, 4xx not, malformed resampled) instead of a stage-level fake.
3. `_ensure_raw` restores a deleted `window_*.json` from the cache on reuse, so published
   rows never lack their raw evidence.
4. `evidence_segment_ids` is JSON-encoded in the table: no existing written schema used
   list<string>, and nested-write capability was not invented for it.

## Allowed edit surfaces (writer)

- `src/multimodal_pipeline/stages/stories.py` (new)
- `src/multimodal_pipeline/config.py`
- `src/multimodal_pipeline/schemas.py`
- `src/multimodal_pipeline/artifacts.py`
- `src/multimodal_pipeline/stages/base.py`
- `src/multimodal_pipeline/orchestrator.py`
- `src/multimodal_pipeline/stages/__init__.py`
- `src/multimodal_pipeline/elan.py` — only the coverage-reason entry for the new keys
- `tests/unit/test_stories.py` (new)
- other existing tests only where a registry-count assertion genuinely moves

## Log

- 2026-10-01 probe run, numbers above; raw responses `/tmp/stories-probe/*.json`.

## S16c: independent verification of `9f39903` and the fixes it forced (2026-10-02)

The verifier (gentle-ai-verify, read-only, /tmp-only writes) confirmed the suites (1949/8
unit, 42 e2e, pyflakes clean), the rejection guards (no escaping TypeError on any of its
malformed inputs), the corpus facts (KABC 1 row w0-s1 [0.071,4.051] conf 0.9; CNN/La-1 0
rows with reasons; both speechless datasets skipped with reason; 0 key hits across 20
stories files), raw/table agreement on KABC, and fingerprint coverage of transcript and
prompt source. It falsified two claims and found two more defects; all four were
reproduced by the parent before any fix, with the parent's own probes:

1. **HIGH — cache key ignored endpoint and prompt text** while the key's own docstring
   promised they mattered. Forced reruns after an endpoint or prompt-text change reused
   the stale window (`batches_reused: 1`, 0 client calls). Fix: `key()` now hashes the
   rendered template's SHA256 and `base_url` (as `StoriesRequest.endpoint`). Red-first
   tests including the verifier's exact one-level-up reproduction. Mutation-checked:
   removing either field kills its named test.
2. **MEDIUM — validator stripped ids, `_row` did not**: `" s1 "` validated and published
   `w0- s1 ` while its child cited `w0-s1` — a validator-guaranteed orphan. Fix: the row
   now uses the stripped entry key validation already computed. Mutation restores the
   raw field and kills two tests.
3. **MEDIUM — a cache entry whose `content` contradicts its `payload` was half-trusted**:
   the payload re-validated (no invented id could reach the table — the verifier verified
   that) but `_ensure_raw` restored the tampered response text as the raw window file.
   Fix: reuse requires `parse_stories_response(content) == payload` or refetch.
4. **LOW — a dropped story could carry a parent nobody answered**, skipping the breach
   check accepted entries got. Fix: fabricated parents breach for every entry, dropped
   included; a parent that IS in the answer but was dropped stays the counted propagation
   (deviation 1's tests still pass).

Claim 5's "all 18 stages valid" was the verifier being right and the parent's plan being
stale: the code fix changed the stories fingerprint → finalization/elan correctly went
stale *before the fix reruns*. After this commit's rerun chain (stories → elan, 5 real
endpoint calls again because the key legitimately changed), `status --plan` reports
"valid previous result" for all 18 stages × 7 datasets, and validate is ok 7/7 — the
false claim now dies against a re-measured disk instead of a remembered plan.

Side effects recorded honestly: each corpus dataset now keeps BOTH raw window files (the
pre-fix key's and the post-fix key's) — the old file is the original bytes of the answers
the first run published; deleting them would erase raw evidence, so they stay. The
speechless datasets carry empty `stories/raw/` directories (created by `ensure_dirs`
before `enabled()` skipped the stage) — cosmetic, not in any manifest.

Whole-session integrity of the verifier's own run: its first harness lost its in-memory
baseline (authorized; it declared UNCERTAIN rather than hiding). The parent took an
independent confirmation snapshot after the report: **0 differences across all 2098
files** against `/tmp/stories-verify-final-ga9b2xmu/final_snapshot.json`, so no mutation
occurred from its final snapshot to the parent's check.

Suite after fixes: 1963 collected, 1955 passed / 8 skipped; e2e 42; pyflakes clean; every
rejection fix mutation-checked (M1-M5, M5's revert kills the named test).
