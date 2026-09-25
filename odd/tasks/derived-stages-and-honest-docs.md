# Feature: honest docs, dataset figures, and the derived stages

Branch: `master` (this clone commits work units directly to `master`; see
`odd/tasks/multimodal-video-pipeline.md` §19 for the autonomous-piloting policy).
Operator instruction, 2026-09-24: *"termina todos los bloques"* — finish the three blocks
inventoried in `multimodal-video-pipeline.md` §20 and in the §21 resume point.

## 1. Authorization

Authorized: all three blocks, as one continuous push of work units.
Not authorized (unchanged): force-push, history rewrite, remote branch deletion, merge,
anything outside this clone, hand-made operator data.

## 2. The three blocks, as inventoried and measured

**Block 1 — five measured falsehoods in README.md.** Each was verified against the tree,
not assumed:

| # | Claim | Reality | Evidence |
|---|---|---|---|
| 1 | `README.md:465` "Not yet declared" | `pyproject.toml:7` declares MIT | read both |
| 2 | `README.md:39` "The four heavy tools" | five environments; `README.md:356` says so | read both |
| 3 | `README.md:28` bare `multimodal-pipeline run` | console script of the root project; `uv run` is what works | `pyproject.toml:18` |
| 4 | `README.md:211` a stage missing prerequisites is skipped | OpenPose **fails** when enabled and the binary is missing | `stages/openpose.py:92,108` |
| 5 | `README.md:37` `inspect-environment` lists *every* skip reason | covers HF_TOKEN, translation, input dir, OpenPose binary, uv-project dirs only | `cli.py:449-464` |

Claim 5 is both a documentation defect and a capability gap: an environment that exists
but has never been `uv sync`-ed is invisible to `inspect-environment`, and so are a missing
`talknet_root` and a spaCy stage running `blank`.

Test counts in README were checked with `--collect-only` and are correct: 681 unit + 30 e2e.

**Block 2 — the six scoped capabilities** (`multimodal-video-pipeline.md` §20.1–20.6).

**Block 3 — debt with no §20 entry**: no CI at all; `tests/e2e/test_cli_smoke.py` *fails*
rather than skips on a machine without OpenPose/ffmpeg 7 (so a clean machine cannot verify
its own install); `scripts/make_fixtures.sh` uses a second OpenPose path convention and
checks for `ffmpeg` but needs the `flite` filter; `activespeaker` emits no warnings at all
(review finding R3-001); frames past the imputable tail are indistinguishable from
`no_face`; and nobody decided whether a `language_detection.status: low` should drive spaCy
model choice at all.

## 3. Design decisions taken here, with the assumption each rests on

1. **OpenPose stays loud.** A stage the operator explicitly enabled must fail rather than
   skip when its binary is gone — a silent skip would hide a broken install behind a green
   batch. The README sentence gets narrowed to name that exception instead of the stage
   getting softer. *Assumption: an operator who sets `openpose.enabled: true` wants pose
   data or an error, not a shrug.* Reversible: the alternative (skip with reason) is one
   small diff if the operator prefers it.
2. **License becomes MIT for real.** `pyproject.toml` has declared MIT all along, so the
   inconsistent artifact is the README. Adding the `LICENSE` file makes the declaration
   match what is already machine-readable. Called out explicitly because a license on a
   public repo is the operator's legal decision, and one commit reverts it.
3. **CI is CPU-only and unit-only.** A workflow that needs a GPU, OpenPose, `/opt/openpose`,
   an `HF_TOKEN` or the translation endpoint cannot run on a hosted runner, so the workflow
   runs `tests/unit` on one OS and nothing else. It is honest about being partial in its own
   name. *Pushing it means GitHub will start consuming Actions minutes for this repo* — the
   repo is already pushed to and Actions minutes are free for public repos; nothing is
   published and no new credential is used.
4. **The dense frame table gets a `frame_reason` column instead of a cleverer null.**
   `face_status` answers "was a face located"; the past-the-tail case answers "TalkNet
   produced no measurement for this tracked face". Those are two questions and one column
   cannot hold both. Bumping the frames `schema_version` is the honest cost.
5. **Language policy is config-gated, default = today's behavior.** A `low` detection
   currently drives model choice, which is what produced the confidently-wrong Catalan POS.
   The reversible move is to expose the gate and record the reason, then flip the default
   once the operator has seen it work.

## 4. Task list

Each task closes with at least one work-unit commit carrying its tests and docs.

- [ ] **T1** README: license, four→five, `uv run` in the quick start.
- [ ] **T2** README: narrow the skip-vs-fail claim, naming the OpenPose exception.
- [ ] **T3** `inspect-environment`: warn on unsynced environments, missing `talknet_root`,
      and a spaCy stage that will run `blank`; tests; README claim then holds.
- [ ] **T4** e2e suite: skip cleanly on a machine without OpenPose/ffmpeg 7 instead of
      failing, so a clean install can run the suite.
- [ ] **T5** CI: CPU-only unit workflow, explicit about what it does not cover.
- [ ] **T6** `make_fixtures.sh`: require the `flite` filter it needs, unify the OpenPose
      path convention with `openpose.root`, add a test.
- [ ] **T7** `activespeaker`: the stage's first warning logs (closes R3-001).
- [ ] **T8** dense frames: `frame_reason` distinguishing no-face / past-tail / unscored.
- [ ] **T9** spaCy model choice gated on language-detection status, default unchanged.
- [ ] **T10** §20.5 `pose_skeletons`: opt-in `--write_images`, artifact + fingerprint, live
      render of one clip.
- [ ] **T11** `scripts/make_dataset_figures.py` + committed `docs/assets/` (stage graph,
      active-speaker strip, speaker-turn strip, pose skeleton from T10).
- [ ] **T12** §20.6 README: install from nothing, what each stage decides, one worked
      example dataset, how to consume it.
- [ ] **T13** §20.1 `diarization_v2`: fuse pyannote turns with per-frame active speaker,
      v1 kept beside it.
- [ ] **T14** §20.4 `pose_normalized`: body-centred basis (dfMaker algebra) in Python,
      validated, explicit no-valid-basis state.
- [ ] **T15** §20.2 `persons`: own uv environment, Ultralytics detect+track, ids kept
      separate from TalkNet's.
- [ ] **T16** §20.3 `stories`: prototype the prompt against the live endpoint, read the
      output, then decide the schema and build the stage.
- [x] **T17** `diarization_nemotron`: a **second, parallel** diarizer (NVIDIA Nemotron 3
      Diarization) so the operator can compare two engines on the same corpus and choose.
      Added at the end of the queue on the operator's request, 2026-09-24. Done `35119ef`.

## 6. T17 — NVIDIA Nemotron 3 Diarization, as a second engine next to pyannote

Operator instruction: *"añade al final de la cola una donde lo añadimos, para tener tanto
el pyannote como este. Después de analizar los resultados decidiré cuál conviene, pero de
momento, que nos dé las dos opciones de diarización."*

So the requirement is **both engines, side by side**, and the choice is deferred to the
operator after seeing results. That single sentence decides the design: nothing may be
replaced, renamed or reused between the two.

### Verified before writing code

* **Model card and blog read** (huggingface.co/blog/nvidia/nemotron-diarization and the
  model page). Release 2026-09-23. 100M parameters, 31-layer Transformer encoder with RoPE,
  up to 8 speakers, arrival-ordered channels.
* **Not gated.** `GET /api/models/nvidia/Nemotron-3-Diarization` with the project's
  `HF_TOKEN` returns `"gated": false, "private": false`. Unlike
  `pyannote/speaker-diarization-community-1`, **no EULA click is required** — worth
  stating because the pyannote path is the number-one setup failure in this repository's
  troubleshooting table.
* **Licence is `openmdw-1.1`**, not MIT/Apache. "Ready for commercial or non-commercial
  use", but it is a different licence from the pipeline's own MIT, so the two must not be
  conflated in the README.
* **Hardware is fine here.** The blog's prose says Ampere/Hopper/Blackwell, which made this
  agent suspect the RTX 4090 was excluded (`nvidia-smi --query-gpu=compute_cap` → 8.9, Ada
  Lovelace). The **model card** lists Ada Lovelace explicitly, *GeForce RTX 4090 first*.
  The prose was the narrower claim; the card is authoritative.
* **Input is exactly what the pipeline already produces**: 16 kHz single-channel, `.wav`
  accepted. That is `audio/audio.wav`, no new conversion stage.
* **Output shape differs from pyannote in a way that matters.** `diarize()` returns strings
  `"start end speaker_id"` per segment, and **segments may overlap across channels** —
  the blog's own example has speaker_0 and speaker_1 active simultaneously. pyannote's
  exclusive `speaker_turns.parquet` cannot hold that, so a second table is mandatory rather
  than a convenience.
* **Five latency operating points**, all in 80 ms encoder frames, and the five values must
  come from one table row and be validated with `_check_streaming_parameters()`. Offline
  style (30.4 s buffer) is the accuracy point and the right default for batch video.
* **Install route is still open** — see the two probes below. transformers **5.17.0
  (latest release on PyPI) does not contain `nemotron3_diarization`**; it exists only on
  `main` (checked via the GitHub contents API). So either pin a git commit (reproducible,
  but not a versioned release) or use `nemo-toolkit[asr]` (the route the blog shows).

### Design consequences

1. New stage `diarization_nemotron`, its own uv environment, its own artifacts, its own
   table. `speech/speaker_turns.parquet` and every existing artifact keep meaning *pyannote
   only* — silently redefining them would relabel every dataset already produced.
2. `speaker_id` namespaces must not collide: pyannote emits `SPEAKER_00`, Nemotron emits
   `speaker_0`. Both are stored verbatim in their own table and the README states they are
   different id spaces that must not be joined.
3. Nemotron's model version must enter the stage fingerprint, the way the spaCy installed-
   model inventory had to (see §16 of the main feature doc): pinning a git commit of
   transformers changes the output and must invalidate the cache.
4. Skips, never fails, when the environment or model is absent — this is an optional second
   opinion, and a corpus must still process with only pyannote, and with only Nemotron.

## 7. T17 result — what was built and what it measured

Commit `35119ef` (see `git log`). The stage is `diarization_nemotron`, off by default.

### The runtime route was decided by a probe, not by the blog

Four environments were built and run against the real checkpoint. Three failed:

| route | outcome |
|---|---|
| `nemo-toolkit[asr]`, default resolution | `torch 2.14.0+cu130`; `cuda avail: False`, "driver too old (found version 12050)" |
| `nemo-toolkit[asr]` + torch 2.8.0 cu128 | CUDA fine, class imports, **checkpoint refuses to load**: `self_attention_model='rope' is not supported` |
| `transformers` 5.17.0 (latest PyPI release) | `ModuleNotFoundError: transformers.models.nemotron3_diarization` |
| `transformers` @ git `5880561a` + torch 2.8.0 cu128 | **works** — 417 weights on `cuda:0` |

So NeMo is not a preference casualty, it is genuinely broken for this checkpoint, and the
pin is an unreleased commit. Both facts are recorded in
`environments/diarization_nemotron/pyproject.toml` and in README *Why six environments*.

Two dependencies the blog does not mention turned out to be required: `librosa` (feature
extractor) and `accelerate` (`device_map`). Found by running, not reading.

### Design changes forced by reading the code rather than the blog

* **No operating-point flags.** The card's five knobs (`spkcache_len`, `fifo_len`,
  `chunk_len`, `chunk_right_context`, `spkcache_update_period`) are assigned on
  `model.sortformer_modules` — an object that only exists on the NeMo path. In the
  HuggingFace path, reading `modeling_nemotron3_diarization.py` shows `forward()` enters
  offline mode when neither `speaker_cache` nor `num_lookahead_frames` is passed, and the
  checkpoint's own resolved values (`chunk_length=340`, `chunk_right_context=40`,
  `fifo_length=40`, `speaker_cache_update_period=300`, `streaming_config.speaker_cache_length=264`)
  **already are the card's Offline Style row**. So one whole-clip call *is* the accuracy
  point, and an `--operating-point` flag would have been a control that controls nothing.
  The only real knob is `extract_speaker_dict(threshold=)`, exposed as `threshold`.
* **No `max_retries`.** `run_worker` has no retry plumbing at all (only `translation`
  implements its own), so a retry knob would be config that nothing reads. Asserted absent
  by a test.
* **No `fallback_to_cpu` copy of an existing idea** — it is genuinely new behaviour here
  (pyannote's worker only ever degrades), implemented as a pure `resolve_device()` so it is
  tested without mocking `torch.cuda`.
* **`max_speakers` is capped at 8** because the logits are `(1, frames, 8)`; a request for a
  ninth channel cannot produce one, so it is refused at config-parse time.

### The end-to-end test found a real defect

The worker's fallback result filename was `nemotron_diarization_worker_result.json`, but the
harness computes `{stage_name}_worker_result.json` and never passes `--result-json`. Every
real run therefore failed with "worker produced no result JSON" **after diarizing
successfully**. Not predicted, not guessable from the unit tests (which never invoke the
worker as a subprocess), found the first time the committed worker was run end to end.
Fixed, and `test_result_json_default_matches_the_harness_convention` now compares the two
names so it cannot drift back.

### Measured on the real corpus (7 videos, `config/config.local.yaml`)

`run` → 7 completed / 0 failed in 42 s; `validate --json` → `ok: true`; a following
`status --plan` reports `diarization_nemotron valid previous result` for all seven, so the
fingerprint stabilises.

| clip | pyannote turns / speakers | Nemotron turns / speakers | Nemotron overlapping |
|---|---|---|---|
| KABC Kimmel | 1 / 1 | 3 / **2** | 3 |
| CNN Arctic Melt | 1 / 1 | 1 / 1 | 0 |
| La1 Telediario | 2 / 1 | 4 / **2** | 3 |
| person_demo | 0 / 0 | 0 / 0 | 0 |
| pipeline_demo | 2 / 1 | 1 / 1 | 0 |
| pipeline_demo_ntsc | 2 / 1 | 1 / 1 | 0 |
| pipeline_silent | 0 / 0 | 0 / 0 | 0 |
| **total** | **8** | **10** | **6** |

Both engines agree on the two silent clips and on CNN. They disagree on KABC and La1, where
Nemotron claims a second speaker with real overlap — which is exactly the behaviour the
model was built for *and* exactly its over-splitting failure mode. Cost per video is
comparable: pyannote 5.3 s, Nemotron ~4 s (≈1.2–1.5 s model load + ≈0.15 s inference,
re-recorded per video because the worker is one process per video). So the choice is not a
speed decision, and this document deliberately does not recommend an engine.

### Absent-environment path exercised on real data

Pointing `uv_project` at a nonexistent directory: `run` completed all 7 videos, the stage
recorded `status=skipped` with
`reason="nemotron environment not installed at … create it and run \`uv sync --python 3.12\` there to enable the
second diarizer"` (the harness records skip reasons in `validation_result.reason`, not in a
`message` field). Config restored afterwards and the dataset re-validated `ok: true`.

### Two claims that were wrong before they shipped

* This agent first asserted the RTX 4090 was unsupported because the blog prose says
  "Ampere, Hopper, or Blackwell" and `nvidia-smi --query-gpu=compute_cap` reports 8.9 (Ada).
  The **model card** lists Ada Lovelace with GeForce RTX 4090 first. The card was
  authoritative and the suspicion was wrong.
* An earlier draft of the README claimed Nemotron was "faster by a lot" against a pyannote
cost of "minutes per corpus". Measured from `status.json`, pyannote cost 37 s across the 7
videos, and Nemotron's own `load_seconds` (1.55, 1.22) dwarf its `inference_seconds` (0.148,
  0.153). The claim was deleted rather than softened.

### Follow-up left open, deliberately

`speaker_assignment` still reads pyannote only. Making the engine switchable is a separate
decision with a real consequence: switching it silently relabels `speaker_id` in
`speech/segments.parquet` and `speech/words.parquet` for every dataset produced so far. That
warrants its own change once the operator has read the two tables.

## 5. Execution notes

- Delegation is live in this clone for read-only task-mode work (a `gentle-ai-explore` run
  mapped the install path this session). Writer launches historically failed here with
  `Select an existing worktree in the same Git clone as this session`; `git worktree list`
  now shows exactly one worktree. Try the writer, fall back inline, and say which happened.
- Verification: `uv run --with pytest pytest tests/unit tests/e2e -q -p no:randomly`.
- Live: `uv run multimodal-pipeline run -c config/config.local.yaml`, then
  `validate -c config/config.local.yaml --json`.
- RDD is `on (decided by default)` for this clone despite what `AGENTS.md` says (see §21 of
  the main feature doc). A review receipt is therefore possible; the absence of one is
  never approval.
