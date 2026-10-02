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

Endpoint: the translation endpoint itself (`nienna-llm.inf.um.es/v1`, model `chat`),
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

- [ ] S16a Probe the endpoint, choose the schema from its answers — **done** (this file).
- [ ] S16b Implement the stage: `StoriesConfig`, `STORIES_SCHEMA`, artifact keys,
      `StoriesStage` (prompt, client reuse pattern, windowing, rejection parser,
      cache), registry + DAG wiring after `speaker_fusion`, mock client, unit tests
      (red first; mutation-check the rejection rules).
- [ ] S16c Independent verification agent (full unit + e2e, pyflakes, probes on the
      real artifacts) + README/claims updates (stage table, collection ratchet).
- [ ] S16d Operator config: add `stories:` section to `config/config.local.yaml`
      (backup first) and run real detection over the corpus; report per-clip stories
      and the empty-count.

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
