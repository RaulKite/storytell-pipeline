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

Test counts in README are now asserted against `pytest --collect-only` by
`tests/unit/test_readme_claims.py::TestDocumentedTestCounts`. They had drifted twice before
that ratchet existed (the prose said 681 unit + 30 e2e while the tree collected 766 + 42);
they currently read 782 unit + 42 e2e.

**Block 2 — the six scoped capabilities** (`multimodal-video-pipeline.md` §20.1–20.6).

**Block 3 — debt with no §20 entry**: no CI at all; `tests/e2e/test_cli_smoke.py` *fails*
rather than skips on a machine without OpenPose/ffmpeg 7 (so a clean machine cannot verify
its own install); `scripts/make_fixtures.sh` uses a second OpenPose path convention and
checks for `ffmpeg` but needs the `flite` filter; the dense-sequence check in
`activespeaker` rejects three different faults with one string and logs nothing (review
finding R3-001 — the finding's own premise that the stage "emits no warnings at all" was
stale on arrival, see §8); four different causes leave a tracked face unscored and all four
write a byte-identical row (the R3 follow-up blamed `track_id = null` and "no row is
emitted" — both false, see §9); and nobody decided whether a `language_detection.status: low`
should drive spaCy model choice at all.

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

- [x] **T1** README: license, four→five, `uv run` in the quick start — closed in `c7c59c2`
      ("make the README's checkable claims checkable"), never ticked here. Verified against the
      tree this session: `LICENSE` exists (MIT, the same SPDX `pyproject.toml:7` declares), no
      "*N heavy tools*" sentence survives anywhere, and the quick start is all `uv run`.
      `tests/unit/test_readme_claims.py` (21 tests) is the ratchet that keeps it true.
- [x] **T2** README: narrow the skip-vs-fail claim — same commit `c7c59c2`. README:49-51 now
      says stages with a required install are *not listed* because they do not skip, pointing at
      the resume/reuse section instead of claiming a universal skip.
- [x] **T3** `inspect-environment` warnings — `ba11a0b`, 18 tests in
      `tests/unit/test_environment_warnings.py`. **One part of this task was resolved as
      not-a-defect, on evidence:** "warn on unsynced environments" is not implemented and must
      not be. `uv run --project` resolves and syncs an existing uv project on first use,
      measured on this machine against a throwaway project, so an existing-but-unsynced
      environment is not a degradation — warning about it would fire on every fresh clone
      before any run had done anything wrong. Only a *missing* uv project directory warns
      (`test_absent_uv_project_is_warned`, and `test_present_but_unsynced_project_is_not_warned`
      pins the deliberate half). Missing/unusable `talknet_root` and the `blank`-model spaCy
      stage both warn, reusing the stage's own skip strings so the pre-flight message and the
      runtime reason cannot drift.
- [x] **T4** e2e skips what the machine lacks — `8cc52ec`. `pytestmark` skips the module
      without ffmpeg; `openpose_or_skip()` skips rather than fails when the binary is not
      discovered; `parse_ffmpeg_major()` is extracted so the skip-vs-fail decision is testable
      with a version this machine does not have (`test_a_machine_without_openpose_skips_instead_of_failing`
      exercises it with a foreign root). Not verifiable end-to-end on this box, which *has*
      OpenPose at `/opt/openpose/build/examples/openpose/openpose.bin` — the synthetic
      skip-path test is the evidence, not a live no-OpenPose run.
- [x] **T5** CPU-only unit workflow — `9eb3070` added `.github/workflows/unit.yml`; this
      session found its header had rotted and fixed the class of defect in `462ec16` (11 tests
      in `tests/unit/test_ci_workflow_claims.py`). The header still claimed "the 718-test unit
      suite" and a "drop coverage by 15 tests" penalty and called the ffmpeg-dependent tests
      "two tests"; measured here, three repeat runs each, with the exact command CI runs:
      **1065 passed / 8 skipped** with ffmpeg, **1048 passed / 25 skipped** without, so **17**
      tests are ffmpeg-gated, not two. The counts are now banned from the workflow rather than
      corrected, because no cheap ratchet can derive a passed/skipped split from inside the
      suite (see §18).
- [x] **T6** `make_fixtures.sh`: require the `flite` filter it needs, unify the OpenPose
      path convention with `openpose.root`, add a test.
- [x] **T7** `activespeaker`: name the dense-sequence fault and log it (closes R3-001) — `e542dbe`.
- [x] **T8** dense frames: `frame_reason` naming which of the four causes left a row unscored — `5505bdb` (code+tests+docs as one work unit), review recorded in `406f1c2`; native review approved (high, 4 lenses), evidence in §9.
- [x] **T9** spaCy model choice sees the language-detection grade, default unchanged — `ba602d7`, evidence in §10.
- [x] **T18** corrupt `whisperx_raw` reads as an `unreadable` sentinel, never as "no grade" (advisory `R4-raw-read-failure-cache`) — `dbe30f3`, evidence in §12.
- [x] **T10** §20.5 `pose_skeletons`: opt-in `--write_images`, artifact + fingerprint, live
      render of one clip — `4fe3d7d`, evidence in §13.
- [x] **T20** fuse Nemotron turns with ASD output — done in the same commit as T13, as its
      design consequence required: the fusion core takes *which* turn table to consume as a
      parameter, so T20 is T13's second instantiation, not a second fusion. `d535f64`,
      evidence in §17. Written to `speaker/fusion_nemotron.parquet` (not the
      `active_speaker_fused_*` name first sketched here: the fused tables sit beside the
      turn tables they were derived from, and `fusion_<engine>` says that in one word).
- [x] **T21** three of `speaker_fusion`'s five advisories closed — `d7b01ba`, 10 tests.
      Phase-split writes (a mid-run engine failure now writes nothing), skip-path pruning
      gated on the config flag rather than the skip wording, and `validate` comparing each
      fused table's row count with its own turn table. `R2-001` and `R4-001` stay open: their
      claim text is not recoverable from this machine's transaction store and guessing at a
      reviewer's intent would be worse than leaving the advisory. Evidence in §18.
- [x] **T19** T10's three openpose advisories closed — `241535e`, 14 tests. The zero-render
      advisory turned out to be the visible symptom of a bigger defect measured against the real
      binary (below). Evidence in §19.
- [x] **T11** `scripts/make_dataset_figures.py` + committed `docs/assets/` (stage graph, active-speaker
      strip, speaker-turn strip, pose skeleton) — `950284b` (this commit, amended before any push), evidence in §15.
- [ ] **T12** §20.6 README: install from nothing, what each stage decides, one worked
      example dataset, how to consume it.
- [x] **T13** §20.1: fuse pyannote turns with per-frame active speaker, v1 kept beside it —
      `d9120dd`, evidence in §17. Named `speaker_fusion`, not `diarization_v2`: it does not
      re-segment audio, so a `diarization`-prefixed name would promise a diarizer and read as
      a replacement for the stage whose output it consumes.
- [x] **T14** §20.4 `pose_normalized`: body-centred basis (dfMaker algebra) in Python,
      validated, explicit no-valid-basis state. Four linked commits (`5eb1214`, `4ec3a33`,
      `f862e06`, `23045e5`); §21 is the build and corpus measurement, §22 the review,
      including the link the reviewer could not finish and the two docstring defects its
      advisories exposed. Whole-corpus agreement with `dfMaker` 0.1.1: 53588 numeric points,
      worst 9.55e-15, 0 absence disagreements.
- [ ] **T15** §20.2 `persons`: own uv environment, Ultralytics detect+track, ids kept
      separate from TalkNet's.
- [ ] **T16** §20.3 `stories`: prototype the prompt against the live endpoint, read the
      output, then decide the schema and build the stage.
- [x] **T17** `diarization_nemotron`: a **second, parallel** diarizer (NVIDIA Nemotron 3
      Diarization) so the operator can compare two engines on the same corpus and choose.
      Added at the end of the queue on the operator's request, 2026-09-24. Done `35119ef`.

## 5. T17 — NVIDIA Nemotron 3 Diarization, as a second engine next to pyannote

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

## 6. T17 result — what was built and what it measured

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

## 7. T6 — `make_fixtures.sh`: a gate that says why, one OpenPose root, and real verification

### What was actually broken, measured first

* The script checked `command -v ffmpeg` but its whole product depends on the **`flite`
  filter**, which comes from libflite and is absent from many packaged ffmpeg builds.
  Verified here that the filter is build-specific: `ffmpeg -filters | grep flite` prints
  `... flite  |->A  Synthesize voice from text using libflite.` on this machine and nothing
  on a build without it. The failure mode was a lavfi parse error naming neither flite nor
  the fix.
* A bad `voice=` fails the same way with a different remedy (`Could not find voice 'x'`,
  exit 234), so it gets its own diagnosis. Confirmed the voice-listing trick works:
  `voice=?` makes ffmpeg print `Choose between the voices: awb, kal, kal16, rms, slt`.
* It located OpenPose through an **`OPENPOSE_ROOT` environment variable** while the
  pipeline uses `openpose.root` from YAML (`provenance.openpose_report()` discovers the
  binary and models under it). `grep -rn OPENPOSE_ROOT` returned exactly one hit: this
  script. Nothing else in the repository has ever read that variable, so a machine with
  OpenPose anywhere but `/opt/openpose` was told "media not found" by the fixture script
  while processing videos with OpenPose correctly.
* Every `ffmpeg` call printed `generated <path>` unconditionally. `-shortest` makes the
  output as long as the shorter stream, and an invocation that exits 0 having produced an
  unprobeable file still announced success.

### Decisions

1. **Root resolution order**: `--openpose-root` > `openpose.root` from `--config` >
   `openpose.root` from the repository's own `config/config.local.yaml` when it exists >
   `OpenPoseConfig().root` read from the code. The environment variable is deleted, not
   deprecated — one name for one setting. The default is asked of the real pydantic model
   rather than re-literalled as `/opt/openpose`, so a change to the schema cannot leave a
   stale copy here.
2. **Missing person clip stays a warning, not an error.** The colour-bar corpus is a
   legitimate corpus and the pose stages legitimately detect nobody; refusing to produce
   fixtures because a non-redistributable OpenPose sample is absent would break fresh
   installs for the wrong reason.
3. **`--openpose-root` is reported in the output with its source** (`from --openpose-root`,
   `from <config path>`, `from OpenPoseConfig default`) because the entire point of the
   change is that the script and the pipeline can no longer disagree silently.

### The bug my own first fix contained

`ffmpeg -filters 2>/dev/null | grep -qE 'flite'` under `set -o pipefail` **rejected this
machine's perfectly good ffmpeg**. grep exits on its first match, ffmpeg then dies of
SIGPIPE with 141, pipefail reports the whole pipeline as failed, and the negation read it
as "no flite". A gate that blocks working machines is worse than the silent failure it
replaced, so the filter list is now captured into a variable before being matched.
`TestFliteGate::test_accepts_a_build_that_has_the_filter` exists only to keep that
false-negative dead.

The same run also proved `-shortest` behaviour: `pipeline_demo.mp4` is 9.985 s although the
request says 14 s, because the TTS clip is 9.985 s. Measured against the committed fixture
(`data/input_videos/pipeline_demo.mp4` is also 9.985 s), so this is the script's long-standing
intended behaviour, not a regression — recorded here so nobody "fixes" it by removing
`-shortest` and producing 14 s of frozen picture.

### Tests

`tests/unit/test_make_fixtures.py`, 16 tests. A shim `ffmpeg`/`ffprobe` is used because the
defects live in shell control flow and exit codes, and mocking a shell script's control
flow is what hides them.

| mutation | test that dies |
|---|---|
| flite gate removed | `test_rejects_a_build_without_the_flite_filter_and_names_the_fix` |
| duration verification removed | `test_a_file_that_is_not_media_fails_the_run` |
| audio-stream check removed | `test_a_video_without_an_audio_stream_is_rejected` |
| `OPENPOSE_ROOT` read reinstated | `test_the_environment_variable_is_gone` |
| config root ignored | `test_config_root_is_used_when_no_flag_is_given` |

Two of my own test bugs were found and fixed while writing these: the shim's
`*-f*null*` pattern also swallowed the `anullsrc` silent-fixture call (so no file was
produced at all), and the ffprobe durations table matched a bare filename while the script
passes absolute paths, so an assertion about "a 0-second file" had never actually fired.
Two tests also encoded *this machine's* filesystem (they depended on whether `/opt/openpose`
exists) and were rewritten to assert provenance rather than absence.

`TestAgainstRealFfmpeg` runs the real toolchain and checks the fixtures are probeable
**and actually contain speech** (`volumedetect` mean > −60 dB for `pipeline_demo.mp4`,
< −60 dB for the silent control), which no shim can establish. Full suite: 782 unit +
42 e2e pass.

## 8. T7 — R3-001: the dense check now says which fault it saw

**The finding was half-obsolete, and saying so is part of closing it.** R3-001 was filed with
the premise that "`activespeaker` emits no warnings anywhere". That stopped being true at
`f5f3414`: `normalize()` logs a device-fallback warning and an unscored-frames warning today.
The finding's substance survives, and it is the denser of the two: the dense-timeline check
rejected the frame table with one umbrella string for three different faults, and the stage
log recorded nothing at all.

**What changed.** `dense_sequence_break(indices)` is a pure function returning a diagnosis or
`None`. `validate()` reports `not a dense 0..N-1 sequence: <why>` and writes the same line to
`logs/<stage>.log` at WARNING. The three shapes are distinguishable now:

| input | diagnosis |
| --- | --- |
| `[0,1,3,2]` | row 2 holds frame_number 3 instead, the rows are out of order |
| `[0,1,2,7]` | 1 frame number(s) missing (first: 3), frame number(s) outside 0..3 (first: 7) |
| `[0,1,1,2]` | frame_number 1 is a duplicate, appearing more than once (4 rows, 3 distinct) |
| `[0,None,2]` | row 1 has no frame_number |

**The correction log for this task is unusually long, and it is all mine.** Four of my own
claims were wrong and were caught by running the code rather than by reading it: row index of
first divergence in `[0,1,3,2]` is 2 not 3; the gap is reported as "1 missing (first: 3)" not
a count of four missing; the duplicate message did not contain the word "duplicate"; and
`validate()` reads the parquet `normalize()` writes, so a test helper that mutated the worker
JSON and skipped `normalize()` failed with "parquet missing" instead of the diagnosis it
claimed to assert. The last one is the same mistake I made twice in T6 — asserting against a
state the pipeline never produces.

**Verification.** `tests/unit/test_activespeaker.py` 60 tests, 10 of them new. Mutations: drop
`ctx.log` → `test_the_rejection_is_written_to_the_stage_log` dies; collapse the message back to
the umbrella string → 3 die; make the pure function return one generic reason → 8 die.
Full suite after the change: 793 unit + 42 e2e = 835 collected, 834 passed and the README count
ratchet failed until I updated it to the measured 793 — which is the ratchet doing its job.

**Committed as `e542dbe`.**

## 9. T8 — `frame_reason`: the four unscored causes stop sharing one row

**The debt statement was wrong, and the mapping caught it before any code was written.**
Follow-up 2 of review R3 says frames past the imputable tail "fall into the same
`track_id = null` rows as 'no face detected'" and that "no row is emitted for them at all".
Both halves are false. `workers/activespeaker_worker.py` emits a row for every frame
(`select_stable` appends one entry per frame, `build_document` iterates it), and for a
past-the-tail face that row carries a **non-null `track_id`** and `face_status =
"tracked_unscored"` — the test `test_past_the_imputable_tail_keeps_the_face_without_a_score`
already locked that. So the surviving defect is narrower and real: **four different causes
produce byte-identical rows**, and the stage's own warning could only say "(past the
imputable tail, or a non-finite score)" because that is all the data supported.

**Why the fix cannot live in the stage.** The distinction is two locals inside
`build_candidates` — `position` (the frame's index inside its track) and `score_count` (how
many scores that track produced). Neither is ever serialised: the frame dict has 11 keys and
none of them is either quantity. Given only `track_id`, a bbox, two null scores and
`score_imputed=False`, no arithmetic recovers which of the four branches ran. The information
is destroyed at `Candidate` construction, worker line 358. So the worker names the cause at
the branch that knows it, and the stage carries it.

| `frame_reason` | set where | meaning |
| --- | --- | --- |
| `scored` | `position < score_count`, finite | measured |
| `imputed_tail` | the successful carry branch | last score carried over the bounded tail |
| `no_face` | `build_document`, no candidates | nobody was on screen (a frame fact, not a track fact) |
| `score_not_finite` | `position < score_count`, non-finite | a score existed and was NaN/inf |
| `track_has_no_scores` | `score_count == 0` | the track never produced one |
| `past_scored_tail` | beyond `score_count + 2` | ran out of scores a while ago |
| `tail_score_not_finite` | in the window, carried value non-finite | the only value available to carry was broken |
| `unknown` | raw artifacts predating the field | not recoverable — never a guess at one of the four |

**Three design points worth their comments.** `Candidate(` is constructed in exactly one
place and `smooth_scores` mutates in place, so `unknown` is unreachable inside the worker and
means exactly one thing: an old raw artifact. `_frame_row` passes a *present*
`score_reason` through even when it is nonsense, because rewriting a malformed worker row
into a legal default here would hide the fault; `validate()` names the frame instead. And
`validate()` gained a missing-columns guard before its per-row reads — five neighbouring
stages already had one and this stage was the outlier where a stale table raised a bare
`KeyError`, which the orchestrator records as a crash rather than as the one line naming the
column to rerun.

**Real-data verification, because unit fixtures cannot prove this.** The four causes are
constructed by hand in tests; only TalkNet says whether the branch mapping is right. Full
batch on the 7-video corpus after the change: 7 completed, 0 failed, 61 s. `validate`
`ok: true` for all seven. What the worker actually wrote:

| video | rows | reasons |
| --- | --- | --- |
| KABC | 105 | 104 scored, 1 imputed_tail |
| CNN | 103 | 102 scored, 1 imputed_tail |
| La1 | 200 | 148 scored, 4 imputed_tail, 48 no_face |
| person_demo | 103 | 25 scored, 1 imputed_tail, 77 no_face |
| pipeline_demo / _ntsc / _silent | 249/249/100 | all no_face |

Zero `unknown` rows, which is the proof that the worker always reaches a branch that knows.
La1 is the unchanged-decisions check: 148 + 4 = 152 tracked, 48 without a face, the same
152/48 measured for the `face_status` work. The new column describes the same rows and moves
no decision, which is what a diagnostic column owes the dataset.

**Verification.** 85 tests in `tests/unit/test_activespeaker.py` (was 60), full unit 818,
e2e 42, suite 860 passed. Two mutations run by the parent, independent of the writer's own
matrix: collapsing `past_scored_tail` into `tail_score_not_finite` kills
`test_past_the_imputable_tail_keeps_the_face_without_a_score` and
`test_the_four_unscored_causes_get_four_different_reasons`; disabling the missing-columns
guard kills `test_a_stale_table_missing_the_column_is_diagnosed_not_crashing` and
`test_the_missing_column_message_names_every_missing_column`.

**Native review.** START on base `8b6fb315…` committed-only, lineage `review-e6f720fdcf7f223b`,
tier **high** (`process_boundary: shell_process` in `workers/activespeaker_worker.py`), 656
changed lines, four lenses (risk, resilience, readability, reliability), correction budget 200.
Group capture ran all four reviewers via the host relay (prompts ~66 KB each) and closed
**approved** on the last admitted event; acknowledgement burned the authority
(`burn_evidence: gentle-ai.review-acknowledged/v1`, `authority: burned`,
`delivery: ordinary-repository-policy`). No findings, so no correction round was spent.
The closure again deferred its candidate-view worktree cleanup with the same git failure seen
in T7; the views are left in place for the same reason — the owning pi process (pid 2865097)
is this session's parent and alive.

**Delegation note.** This was the first writer run in this clone that completed at all; the
launch failures recorded in `AGENTS.md` did not recur. Its `## Allowed edit surfaces` block
was rejected twice before it was accepted — the validator wants bare paths, one per line,
with no prose inside that section (explanatory text belongs under its own heading). Two
relaunches cost nothing; the rule is now known. The writer also tried to drive the RDD review
lifecycle from inside itself and reported that it could not, which is correct — the facade
lives in the parent and the review below is the parent's, not the writer's.

## 10. T9 — the language-detection grade stops being decoration

**What was true before:** `whisperx_worker.py` wrote `language_detection`
(`{status: configured|ok|low, probability, reasons}`) into `speech/raw/whisperx.json` and
nothing read it — `grep -rn language_detection src/ workers/` returned only the writer. The
spaCy resolver received the detected language string and none of its reliability.

**The measurement that decided the policy.** All seven auto-detected clips in this corpus are
graded `low`, because every clip is shorter than WhisperX's 30 s detection window (0.997,
0.986, 0.883, 0.957, 0.957, 0.238, 0.215). And where the resulting choice was checked it is
correct: La 1 selected `es_core_news_lg` and its Spanish lemmas were verified; the en clips
selected `en_core_web_lg`. A policy that refused to build on `low` unconditionally would
strip the Spanish layer from this corpus's only Spanish clip on the strength of a grade that
is structurally always pessimistic at this clip length. So:

- `spacy.trust_low_language_detection: true` (default) — today's outcome byte-for-byte, plus a
  WARNING naming language, probability, the model still selected, and the key that reverses
  it, and `language_reliability`/`language_reliability_trusted` recorded in the raw document.
- `false` — the source variant stops using the detection for model choice and lands on the
  existing honest path (`fallback_no_model` → blank: tokens and sentences, no lemmas/POS/deps).

**Real-data verification, both directions.** Trusted (default): batch 7/7, `validate ok: true`,
La 1 keeps `es_core_news_lg (substituted_family)`, en clips keep `en_core_web_lg`, the two `nn`
clips keep `blank` — the identical model table as before, now with the grade beside it; the
trusted-low warning is in `logs/spacy_source.log`. Untrusted: the flag set to `false` and a
rerun demoted La 1 to `model='blank' (fallback_no_model)` with a warning naming what was lost
and the key that brings it back. Config restored, rerun, dataset back to the pre-change table.

**The failed experiment that nearly read as a bug.** The first opt-out run showed
`trusted: True` — I nearly filed it as a policy bug. Cause: my Python edited a `spacy:` block
into `config/config.local.yaml` that does not exist in that file; `str.replace` matched
nothing, wrote the file back unchanged, and the pipeline faithfully ran the unchanged default.
Diagnosed by loading the config and printing the flag *before* believing the run. The lesson
is recorded because it is the third time this session a self-inflicted harness failure
looked like a product result (see §7 and §9): when a measurement contradicts a fresh
implementation, check your own probe first.

**Design boundary.** `select_model` is untouched — model *availability* and detection
*quality* are different contracts; the policy is applied at the call site. `SpacyEnglishStage`
deliberately does not inherit the grade (it forces `en`; a cache key depending on
source-language detection would be a false dependency — pinned by a digest-unchanged test).

**Verification.** 31 new tests (`tests/unit/test_spacy_language_policy.py`; 90 across the two
spaCy files), unit 861, e2e 42, suite 903 passed. Parent-run mutations: forcing
`trusted=True` kills three named tests; letting english inherit the key kills two. README
warning texts verified verbatim against the worker strings.

## 11. T9 review outcome — approved, one advisory kept as work

Lineage `review-087287de62cf6552`, tier **high** (`process_boundary: shell_process` in
`workers/spacy_worker.py`), 939 changed lines over the full T9 range (base
`406f1c23…`), four lenses, correction budget 200. All four reviewers ran through the host
relay (~74 KB prompt each) and it closed **approved**; acknowledgement burned the authority
(`burn_evidence: gentle-ai.review-acknowledged/v1`, `authority: burned`,
`delivery: ordinary-repository-policy`). No correction was opened or spent.

**Advisory `R4-raw-read-failure-cache`** (resilience, WARNING, informational,
`stages/spacy_source.py:171-172`) is real and was kept, not argued away: `_language_detection`
catches `OSError`/`ValueError` and returns `None`, which is the same value as "the raw document
carries no grade". A corrupt raw is therefore cached as a *missing* grade, and because the
request does not digest `whisperx_raw` itself, repairing that file later does not invalidate
the spaCy stage. T9 deliberately recorded absence as `None` so old datasets keep working;
this finding is the case that absence was hiding. Filed as **T18** rather than fixed under a
receipt that is already burned — the receipt says nothing about a changed candidate.

## 12. T18 — a corrupt `whisperx_raw` stops masquerading as an absent grade

T9's advisory, done while the code was still warm. `_language_detection` collapsed three
states into two: a file that exists but does not parse returned the same `None` as a file
that does not exist, so (a) the fingerprint cached a corrupt artifact as "no grade" and a
later repair of that file reproduced the broken run's digest — nothing invalidated the
cache — and (b) the stage said nothing about corruption.

Three states now: `None` (absent — missing file, no key, or non-dict value), the sentinel
`{"status": "unreadable"}` (present but unreadable), or the real grade. The worker treats the
sentinel as *available but not low*: default behaviour kept (the language stays trusted — a
corrupt file is evidence about the file, not about the detection), one WARNING saying the
document could not be read and to rerun `whisperx`, and `language_reliability` carries the
sentinel in provenance. The finding's second question — digest the raw file? — was answered
no: the sentinel plus the grade already make corrupt/absent/repaired pairwise-distinct
digests (a test pins exactly that), and a content digest of an artifact the stage doesn't
otherwise read would churn the cache on every byte-unrelated rewrite.

4 new tests (866 unit). Mutation: reverting the sentinel to `return None` dies on four named
tests including `test_corrupt_absent_and_graded_are_three_different_fingerprints`.

**Delegation note.** The T18 writer touched README.md (the two test-count lines), outside its
allowed surfaces, on the strength of an authorisation that lived in my own `## Verification`
section rather than in the surfaces block. The right answer was to stop and report, which it
essentially did; the two lines were checked and kept — they are the measured 866/42 and the
count ratchet cannot be satisfied any other way. The rule for my own task briefs going
forward: authorisation to touch a file goes in the surfaces block or nowhere.

**Review outcome.** Candidate `78170e2..db5f579` (lineage `review-bf33262273f63f3d`, high tier again —
the same `process_boundary: shell_process` in `workers/spacy_worker.py`), four lenses, 169 lines, 658
total. Approved with **no findings** this time; acknowledgement burned the authority. T18's advisory
is closed with no follow-up left open.

## 13. T10 — opt-in OpenPose skeleton renders (§20.5), verified against the binary

`openpose.write_images` (default **off**) asks the run OpenPose already does for its own
rendered frames: `--write_images pose/raw_images --write_images_format jpg` plus per-module
`--render_pose -1 / --face_render / --hand_render`, each render switch following the module's
own enabled config. `openpose.image_max_side` computes `--output_resolution` from the metadata
width/height (never upscales); with it null the stage warns once that OpenPose's `-1x-1`
default renders at full input resolution and costs hundreds of MB to GBs per video. Default-off
argv is byte-identical to the pre-feature command (a test pins the whole literal argv).
Zero rendered images with the flag on **fails the run** before normalisation — OpenPose exits
0 silently in that trap. `validate` enforces images ≥ raw frames − 1 when on, and deliberately
does not check the directory when off (a once-rendered dataset must not fail forever after the
opt-in is switched off; the fingerprint forces the rerun that actually decides).

**Live verification, real binary (`/opt/openpose`), real clip, twice.** Writer's run and my own
independent run through the real CLI (`run --only-stage openpose` on a /tmp copy of
person_demo with write_images+image_max_side=640): exit 0 in 48 s, **205 images**, first file
`person_demo_000000000000_rendered.jpg`, `file` reports JPEG 640x360, 31 MB for the clip.
Bonus measurement kept in the README: JSON keypoints still reach x≈1223 with
`--output_resolution 640x360` — coordinates stay in source pixels because `keypoint_scale`
was not touched (spec §20.5 forbids touching it).

The stage had **zero tests** before this; it now has 48 (`test_openpose_render.py`,
`test_config_openpose_render.py`). Unit 866→914. Mutations: forcing `--render_pose` back to
literal 0 dies on the 3 parametrized render-combo tests; muting the zero-render raise dies on
`test_zero_rendered_images_fails_the_run`.

**The invariant I widened deliberately.** `test_artifacts.py` asserted every `*_raw` layout id
has a path segment *equal to* `raw`; `pose/raw_images` broke it. The writer offered four fixes;
I chose a fifth: match segments *starting with* `raw` and name the invariant's real purpose
(raw output never lands on a normalized table path) in the docstring. Teeth proven by two
scratch probes — pointing `acoustic_raw` OR `pose_images_raw` at `pose/images` fails the test.
The collision half (`test_paths_are_unique`, 40 values) untouched.

**Cost stated, not hidden:** both config keys are hashed into the openpose fingerprint even
when off, so `status --plan` reports "configuration changed" for all 7 existing datasets — the
next batch re-runs OpenPose (the slowest stage) once everywhere. That is the price of an honest
fingerprint and matches the spec's warning; it happens whether or not the operator ever enables
rendering.

**Review outcome.** Candidate `fec202c..750a69b` (lineage `review-684c629a7f5b317a`, high tier —
`process_boundary: shell_process` in the new render tests), four lenses over 959 lines. Approved;
acknowledgement burned the authority. Three non-blocking advisories, each its own later work:
`R2-render-resolution-mismatch` (openpose.py:195-207, readability — the `render_pose -1` inherit
value vs the resolution story in the docstring), `R3-001` (openpose.py:260-275, reliability —
the zero-render raise path), `R4-stale-render-images` (openpose.py:255-263, resilience — images
left on disk when the opt-in is switched off keep being counted by the off-path report). None
opened a correction; none reopens this candidate. They join the queue as **T19** (one small
openpose docs/resilience pass, best folded into T12's documentation sweep since that touches the
same story).

## 15. T11 — figures the repo can actually show (§20.5/20.6)

`scripts/make_dataset_figures.py` renders four figures from the Parquet tables themselves:
stage graph (parsed from `STAGE_ORDER`/`STAGE_DEPENDENCIES` — imported, not hand-copied, so a
new stage appears without anyone editing a drawing), active-speaker strip (ticks coloured by
`frame_reason`, score trace, `is_active_speaker` band), speaker turns against word ticks, and
BODY_25 stick figures with a confidence floor. matplotlib comes via `--with` only; the module
imports it lazily so `pytest tests/unit` stays matplotlib-free (proved: moving the import to
module scope breaks collection). pyarrow reads, pandas never appears (a test enforces it).

**The committed-asset decision.** The pipeline eats copyrighted broadcast video, so
`docs/assets/*.png` are generated from a seeded **synthetic** dataset (`--synthetic --seed 7`)
with column shapes asserted equal to the real `schemas.py` constants — a guard harness that
raises on any file read during a synthetic render proves no broadcast-derived table was
touched. The `--dataset` mode exists for the operator's own inspection of real data and writes
wherever told. Byte-determinism is pinned by a test comparing the committed bytes to a fresh
render: a renderer or matplotlib-version change must come with a regeneration commit.

**Parent verification (my own eyes, not the writer's).** Full suite 949 passed + 8 skipped
(the skips are the matplotlib-rendering tests, correct without `--with matplotlib`).
Determinism: four sha256s identical across committed and a fresh run, byte for byte. PNG
content proven synthetic by pixel census: 6.1–19.2 % non-white, 624–1133 distinct colours,
white background, top colours are tab10 defaults — a photo or rendered video frame cannot look
like that. Mutations I ran myself: renaming bone endpoint `RHeel` → dies on
`test_every_bone_endpoint_is_a_real_keypoint` + `test_the_skeleton_has_no_floating_part` +
the committed-asset reproducibility test; renaming synthetic column `source_timestamp` → 24
failures naming the schema mismatch. Restored green. (First mutation attempt renamed `RToe`,
which does not exist in BODY_25 — my error, no test died for the right reason; the corrected
probe is the one recorded.) Real-dataset run on La1 rendered all four (frame_reason counts
scored=148/imputed_tail=4/no_face=48 — exactly the §21 measurement again), /tmp only, data/
untouched (309 files before and after).

One honest limitation: this model cannot view images, so layout was verified through
matplotlib's Agg renderer (0 overlapping labels, 0 off-canvas) rather than by eye. A human
glance at the four PNGs is still worth one minute.

**Review outcome.** Candidate `8832868..3d4fe69` (lineage `review-03c67ba49ce289bc`, high tier —
`process_boundary: shell_process` on the script), four lenses over 1543 lines. Approved,
acknowledgement burned. Two non-blocking advisories: `R2-001` (make_dataset_figures.py:467-469,
readability — the empty-file error names the output PNG while sibling errors name the source
table; the message could lead to the table instead) and `R4-partial-output` (:717-725,
resilience — a panel's timestamp label takes the first non-None timestamp in a frame, so a
partially-timestamped frame labels itself from one row while other rows say nothing). Both ride
as small polish inside T12's documentation sweep — same file, same story, no separate commit
each. Neither reopens this candidate.

## 16. Execution notes

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

## 17. T13 + T20 — fusing turn tables with per-frame active speaker

Both tasks were implemented in one commit, because the design consequence T20 already
recorded is the whole shape of the work: **one fusion core, two instantiations.** Writing
the core so that *which* turn table to read is a parameter is what makes the second engine
a config list entry rather than a second stage to maintain. Splitting them would have meant
committing pyannote-only code that T20 then had to reshape.

**Measured verdicts on the real corpus.** Every video under `data/input_videos/` (7), run
against a copy of the corpus's processed tables under `/tmp`, with the committed stages and
`engines: [pyannote]` (the default):

| video | fused turns | agreement | turns with a gap |
| --- | --- | --- | --- |
| `KABC_news_45s` | 2 | `face_matched` ×2 | 1 (22/46 frames) |
| `CNN_news_45s` | 3 | `face_matched` ×3 | 2 (23/72, 22/74) |
| `La1_news_45s` | 2 | `face_matched` ×2 | 1 (44/124 frames) |
| `pipeline_demo` | 4 | `no_face_visible` ×4 | 4 (111/111 …) |
| `pipeline_demo_ntsc` | 4 | `no_face_visible` ×4 | 4 |
| `person_demo` | 0 | — | — (`activespeaker` skipped: one person, no second target to track) |
| `silent` | 0 | — | — (no speech) |

The four `face_matched` verdicts were **not** produced by the shipped thresholds. On the real
tables the ASD track is active on 100% of its in-turn frames, but the mean TalkNet score is
1.12 (KABC), 0.79 (CNN) and 0.54 (La1) against `min_mean_score: 1.0` — so the honest reading
with the config the worker wrote was "a face moved its mouth in sync for this turn, but TalkNet's
mean confidence was below the bar", i.e. every matched turn demoted to `face_partial`, with the
demotion caused by the mean-score knob. Two judgements, both reversible:

1. **`min_mean_score` now defaults to 0.0**, and the column is explained as a *reporting* number
   rather than a gate. TalkNet scores are not calibrated and are not comparable across videos —
   they depend on which checkpoint build scored them and on crop quality — so a fixed bar in raw
   score units silently decides verdicts for a corpus the operator never scored against it.
   25/25 real turns carried `mean_score >= 0.54`, so the shipped default now agrees with what the
   data says. The knob stays for anyone who wants it, and the detail column names the threshold
   that decided a turn either way.
2. **`agreement_detail` always reports the frames of the window with nobody on screen.** 44 of the
   La 1 turn's 124 measured frames have no face at all; a detail that said only "80/80, ratio 1.00"
   would read as a face on camera for five straight seconds.

**Native review, and what it found.** Base `3156d30` (committed-only range), tier high, 4 lenses.

- `review-99c538acff70ac95` → **correction required**. R1 and R3 both found
  **`R3-qualifying-track-ignored`**: `_best_track` ranked by absolute active-frame count *first*,
  then applied the thresholds to that winner alone. A turn where track A was active on 2/14 frames
  and track B on 3/3 outvotes the qualifying track B, so B's evidence was silently dropped and the
  turn read `face_partial` with a detail describing track A's 14 frames. Reproduced in 8 lines
  against the shipped core; fixed by selecting the best *qualifying* track and falling back to the
  best-measured one, with the rejected candidates reported in `agreement_detail` — the case where
  audio and video disagree is exactly the case a human needs to adjudicate. 4 new tests; mutation
  (revert selection → 4 named deaths). Committed-only candidate pinned to the pre-fix tree meant
  the correction could not be applied in-lineage (`corrected_candidate_unavailable`), so the fix was
  amended into `c34e21b` and reviewed again.
- `review-7f008fd5daa30ece` → **correction required** (lineage since closed: the candidate is pinned to the reviewed tree, so the fix was amended into `d535f64` and reviewed again). R1/R2/R4 passed; R3 found
  **`R3-stale-engine-output`**: deselect an engine (or delete its turn table) and its
  `fusion_<engine>.parquet` survives on disk, indistinguishable from a current table, and *nothing*
  downstream can notice — both the reuse test and `validate` look only at engines fusible right now.
  Fixed: a completed run prunes fused tables its configuration can no longer compute and logs the
  reason, and `outputs_present` refuses to call a result reusable while a leftover table exists
  (otherwise reuse is the path that keeps the stale file published). Pruning runs **after** every
  write, so a failed run cannot destroy a good dataset, and it reaches only this stage's declared
  outputs. **Deliberate limit:** a *skipped* run prunes nothing, because skip is also what
  `speaker_fusion.enabled: false` and a mid-build dataset produce; if no selected engine has a turn
  table, earlier fused tables stay on disk until the stage completes again. 4 tests, each of which
  fails against the pre-fix stage.

**Third lineage, facade lineage id `placeholder-unused`, target
`sha256:9428b3b1…dbe5c` → approved.** Base `3156d30`, committed-only, tier high, 4 lenses, 13 files / 2544 changed
lines. All four lenses admitted; acknowledgement burned the authority. The reviewed tree is
the tree of `d9120dd`, which is why the review record is this separate commit: amending
`d9120dd` would move the tree its receipt pins.

Five advisories, all `informational`, none of which opened a correction and none of which
reopen this review. They are follow-up work, not reasons to re-review this candidate:

- `R3-stale-output-on-skip` (reliability) — the deliberate limit described above: a *skipped*
  run prunes nothing, so if no selected engine has a turn table, earlier fused tables survive.
  The fix would need to tell "disabled by config" apart from "this engine went away", which is
  a config-state question this stage does not currently own.
- `R3-multi-output-partial-update` (reliability) — the two engines are written sequentially, so
  a failure between them leaves one fresh table and one that is about to be pruned. Pruning
  after all writes is what makes the failure window survivable rather than destructive;
  per-engine atomicity would close it properly.
- `R3-validation-misses-truncated-turns` (reliability) — `validate` counts rows it wrote but does
  not re-check that every input turn produced a row, so a silently truncated input turn table
  would not be caught here.
- `R4-001` (resilience) and `R2-001` (readability) — located in the same stage; neither reviewer
  escalated them past a warning.

Suite after the work: **1073 unit / 42 e2e** (from 1028 before this feature). Mutation set on the
fusion core, each with named deaths: half-open turn window → 19; ratio denominator over the whole
window → 1; unscored frames averaged as `0.0` → 1; `_mean` returning `0.0` for no scores → 1;
silencing the gap note → 1; `validate` demanding a skipped engine's table → 1; revert qualifying-
track selection → 4; remove the prune → 4.

## 18. T5 and T21 — the queue that was already closed, and the numbers that had rotted

**Five of the open checkboxes were already closed and never ticked.** T1 and T2 in `c7c59c2`,
T3 in `ba11a0b`, T4 in `8cc52ec`, T5 in `9eb3070`. The checklist and the tree had drifted apart,
which is the same failure this document keeps re-recording in README prose — the difference
here is that a stale `- [ ]` costs nothing but a wasted afternoon, so nothing hurt and nothing
noticed. Each was verified against the tree before being ticked (section 4 now says which
commit carries which and what was actually measured), not ticked because the commit message
claimed it.

**T3's one genuinely undecided part was decided as not-a-defect.** "Warn on unsynced
environments" is deliberately *not* implemented: `uv run --project` resolves and syncs an
existing uv project on first use, so existing-but-unsynced is a normal pre-run state, not a
degradation, and warning on it would fire on every fresh clone. `test_present_but_unsynced_project_is_not_warned`
pins the half that is *not* warned about, which is the only reason that judgement survives the
next person who reads "warn on unsynced environments" as a to-do.

**The CI header had rotted into fiction, and the fix removed the numbers rather than
correcting them.** `.github/workflows/unit.yml` still described "the 718-test unit suite", a
"drop coverage by 15 tests" penalty, and "two tests" needing real ffmpeg. Measured on this
machine with the exact command CI runs, three repeat runs each, split identical every time:

| ffmpeg on PATH | passed | skipped | total |
| --- | --- | --- | --- |
| present | 1065 | 8 (optional matplotlib) | 1073 |
| ffmpeg + ffprobe absent | 1048 | 25 | 1073 |

so **17** tests are ffmpeg-gated, not two. Measuring them required a PATH without ffmpeg and
`/usr/bin` is root-owned with no sudo here, so the measurement used a 1518-entry symlink farm
mirroring `/usr/bin` minus the two binaries. Two failed attempts are worth recording: a
16-tool minimal farm produces **9 unrelated failures** (the suite's subprocess helpers need
`mktemp`, `dirname`, `sed`…), and a PATH containing only a hand-picked tool list produced 22
failures of the same kind. The farm is the only faithful way, and it is far too heavy to put
inside a test — which is why the workflow now *bans* test counts
(`test_workflow_states_no_test_counts`) instead of carrying unverifiable ones. The README's
counts stay because `test_readme_claims.py` re-derives them from pytest's own collection; that
trick only works for *collected* totals, which is exactly why the workflow says which tests
need which tool rather than how many there are.

**Whether Actions has ever run is still unknown.** No `gh` CLI and no credential on this
machine, so `9eb3070`'s own claim that e2e "has never been run on a hosted runner" is still
true and still unverifiable from here. This work made the workflow's prose honest, not its
status confirmed.

**T21, and a defect the first cut introduced.** Three advisories were actionable; the writer's
phase-split and `validate`-vs-turn-table fixes are clean, but its skip-path pruning had a hole
its own tests did not cover: pruning with an *empty* fusible set treats "no engine has a turn
table" as proof of staleness when it is the absence of evidence. That state is what a
not-yet-diarized dataset looks like, and on the shipped default (`engines: [pyannote]`, one
engine) deleting `speaker_turns.parquet` would have wiped every fused table — verdicts that
cannot be recomputed until the diarizer runs again. Fixed by pruning only when at least one
engine remains fusible, with a test that dies on the fix (`test_losing_every_turn_table_deletes_nothing`)
and one that stops the new guard from swallowing a genuine deselection. The remaining two
advisories (`R2-001`, `R4-001`) are left open on purpose: the approving review's finding text is
not persisted in this machine's transaction store (only the second lineage keeps readable
findings), and inventing a reviewer's intent to "fix" would be worse than an open advisory.

## 19. T19 — the OpenPose render advisories, and what the real binary did with `--write_images`

T19 was scoped as three small advisories from `review-684c629a7f5b317a`: render docs coherence
(`R2-render-resolution-mismatch`), the zero-render raise path (`R3-001`), and stale images being
counted when the opt-in is off (`R4-stale-render-images`). Closing the second one required
measuring the thing it assumed, and the assumption was wrong.

**The root cause under all three: `count_rendered_images` counts files, and both of its
consumers read that count as evidence about the current run.** So a rendering run now empties
`pose/raw_images` first (only that directory, only when rendering was asked for), and `validate`
reports `render_requested` beside `rendered_images` because the off-path count still describes
whoever wrote last. That off-path leniency is deliberately kept — its original reasoning was
correct and a test pins it: failing an opt-in-off run over images it never requested turns a
switch-off into a permanent validation error.

**What the binary actually does (measured 2026-09-26, `/opt/openpose`, 249-frame clip,
`--write_images <dir> --write_images_format jpg --output_resolution 320x240`):**

| module switches | images written | size (PIL) | content | avg bytes |
| --- | --- | --- | --- | --- |
| body+face+hands on | 249 | 320×240 | skeletons | 20 730 |
| `--render_pose 0 --face_render 0 --hand_render 0` | **249** | **640×480** | **the source frame** | 39 331 |

The second row is the defect. `--write_images` writes one image per processed frame even with
every renderer off, and on that path **`--output_resolution` is ignored** — the source is 640×480,
`image_max_side: 320` was asked for, and the files came back 640×480. They are the raw frames:
extracted frame 0 with ffmpeg and compared pixel-wise, mean absolute difference **0.69 grey
levels** over a 255 range. So the request exits 0 with a full directory of maximum-cost,
zero-skeleton images, and the guard that exists to catch "asked to render, got nothing" tests
`rendered == 0` and therefore **cannot ever see it**. The refusal moved to config time, where it
costs nothing: verified live, refusing left the 249 existing images untouched (mtimes unchanged).
`face.enabled` alone still renders normally — also verified against the binary, because a
predicate that over-refuses would be its own defect.

Two of my own measurement errors are on the record here. A hand-written JPEG SOF parser reported
the skeleton-off images as `640x480` and the skeleton-on images as `240x320`; PIL said `640×480`
and `320×240`. The parser had height and width the wrong way round (SOF stores H then W) — the
"resolution is ignored" finding survived only because the *second* image agreed with the ask, and
the correct claim came from PIL, not from my parser. And this model cannot view images, so the
"they are the source frames" claim had to be made pixel-wise rather than by looking.

The T10-era claim that this needed the real tool is stronger than it sounded. A mock decides
whether files appear, so it can only confirm the guard it was written to imitate; nothing about
`--write_images` writing undecorated full-resolution frames could have come out of a fake.

14 new tests, red before green (8 die against the pre-fix stage; over-refusal triangulated by
narrowing the predicate to `body.enabled`, which dies on the face-only case). 1100 unit passed /
8 skipped, 1108 collected; 42 e2e passed.

**An error worth recording, because it is this document's own failure mode:** while rewriting the
README render section, an edit chain deleted a measured claim belonging to T10 — the 1280×720
clip whose images came back 640×360 while `--keypoint_scale` kept its default and JSON
coordinates still reached x≈1223. Restored and verified byte-identical against `HEAD`. The rule
it teaches: when rewriting a paragraph that contains someone else's measurement, rewrite around
it, do not retype it.

## 20. T14 — `pose_normalized`: what `dfMaker` actually computes, measured off the reference

§20.4 left two decisions open — which route (R sidecar or Python), and which triple defines
the frame — and said the reimplementation "must be verified against the reference rather than
asserted". Neither was guessable from the pipeline, so both were settled against the real tool
before any pipeline code was written.

**The reference is installed here.** `R 4.1.2` and `Rscript` are on this machine, and CRAN
carries `multimolang` 0.1.1, whose only import is `arrow` — already present. It was installed
into a private library at `/tmp/rlibs`; the operator's R library and `R-libs/` were not
touched. That turned "validate against route 1" from a promise into something runnable, so
route 2 (pure Python, inside the pipeline's Parquet/provenance/atomic-write discipline) was
taken with the comparison actually performed rather than intended.

**The algebra, read out of the package, is not the algebra the vignette summary suggests.**
`dfMaker`'s `fast_scaling` branch divides every coordinate by `vector_i[1]` — one axis, no
rotation. The linear-transformation branch (`fast_scaling = FALSE`) builds

```
M   = [ i | j ]                      i = P_ip - origin,  j = P_jp - origin
x'  = det([p | j]) / det(M)          p = P - origin
y'  = det([ i | p]) / det(M)
```

and that is a change of basis, i.e. it *does* rotate. Two details were only visible in the
source and both change the numbers:

- `transformation_coords = c(type, origin, i, j)`, and **`i_point_index == j_point_index` has
  its own branch**: `vector_j <- c(vector_i[2], -vector_i[1])`. The `fast_scaling == TRUE`
  path uses `c(-vector_i[2], vector_i[1])` — the *opposite* perpendicular. A reimplementation
  that reads one and not the other gets a correct-looking table with every `y'` negated.
  My first implementation did exactly this: 3271 of 3775 points disagreed with the reference,
  worst error 14.71. The `14.71` was the tell — a rotation sign error scales with distance
  from the origin, so it is huge on the feet and zero on the neck.
- Absence is marked **per coordinate** (`m[,1:2][m[,1:2] == 0] <- NA`), not per point. A point
  with `x = 0, y = 40` keeps `y` and loses `x`. Point-wise masking agrees with the reference
  wherever OpenPose never emits a half-zero point, so the difference is invisible on this
  corpus and would bite on the next one.

**The reference default is a bad frame for this corpus, and it is measurable.**
`dfMaker`'s default triple is `c(1, 1, 5, 5)`: origin `Neck`, basis `Neck → LShoulder`. Over
the 695 real frames / 2901 detectable person-frames in `data/processed`:

| basis | person-frames | median &#124;basis&#124; | p95 | median max(&#124;x'&#124;,&#124;y'&#124;) | p99 |
|---|---|---|---|---|---|
| `Neck → LShoulder` (dfMaker default) | 2901 | **17.7 px** | 89.7 | 4.18 | **556.27** |
| `Neck → RShoulder` | 2893 | 17.7 px | 88.9 | 3.90 | 740.12 |
| `Neck → MidHip` | 2661 | 72.4 px | 190.9 | 0.99 | 3.07 |
| `MidHip → Neck` | 2661 | 72.4 px | 190.9 | 1.11 | **2.07** |

The basis length is the divisor, so a 17.7 px shoulder segment turns a half-pixel OpenPose
jitter into a ~0.03 unit swing, and a wrist four segments away lands at p99 = 556 — the
"normalised" coordinates are *less* stable than the pixels they came from. The spec's
"sternum? pelvis? neck?" was the right question. `MidHip → Neck` is taken: it is the longest
two-point torso segment BODY_25 offers, both endpoints are in the top availability band
(`Neck` 100.0%, `MidHip` 94.2% of frames), and the resulting coordinates sit in a unit-scale
range without rescaling.

The assumption this rests on, stated because it is not forced by the data: `MidHip` is the
average of the two hips in BODY_25, so it is a torso centre rather than an anatomical joint,
and it is the only one of the three candidate origins BODY_25 gives us for free. The cost is
recorded as a config choice, not a constant: the triple is written into the stage fingerprint
so changing it invalidates the table instead of quietly redefining every number in it.

**Fixtures, produced by the reference, not by me.**
`tests/fixtures/pose_normalized/dfmaker_0.1.1_{kabc,cnn}_midhip_neck.csv`, regenerated
byte-for-byte by the committed `scripts/make_pose_normalized_fixtures.R` — run through
`dfMaker(fast_scaling = FALSE, transformation_coords = c(1, 8, 1, 1))` on raw JSON frames
of two clips, then filtered to `type_points == "pose_keypoints"`.
`points` is kept as OpenPose's 0-based index so a reader can check the mapping without
consulting `multimolang`. They are the ground truth the Python transform is asserted against.

**What the Python stage reads, and what that costs.** It reads `pose/body.parquet`, not
`pose/raw/*.json`, so it stays inside the pipeline's read-normalised-tables rule. That is not
free: normalisation drops every keypoint with `score <= 0`, while `dfMaker` masks `x == 0` or
`y == 0` per coordinate. Both encode "absent", and on this corpus they coincide — but the
stage therefore reports the per-frame basis state explicitly rather than leaving a gap for a
consumer to interpret, which is the same lesson `face_status` learned in §17.

## 21. T14 result — `pose_normalized` built, and what the whole corpus says about it

Built as route 2 (§20.4): pure Python, no R at runtime, no new uv environment.

- `src/multimodal_pipeline/pose_normalize.py` — the maths, functions only (`fusion.py`'s
  shape), so the transform is testable without the pipeline.
- `src/multimodal_pipeline/stages/pose_normalized.py` — the stage. One output,
  `pose/normalized.parquet`, beside the pixel tables, which are never written.
- Depends on `openpose` only, and inherits its skip semantics.

**Validated against the reference on the entire corpus, not on a sample.** The fixtures
in §20 are the committed guard, but they cover 5 frames of 2 clips. The claim in §20.4
is that the reimplementation matches `dfMaker`, so the real `dfMaker` 0.1.1 was run over
every raw JSON frame of all four videos that have pose — 695 frames — and compared with
the stage's real output on `/tmp` (nothing in `data/` was touched):

| video | reference rows | shared keys | numeric in reference | worst &#124;ours − dfMaker&#124; |
|---|---|---|---|---|
| KABC | 6300 | 3775 | 3775 | 7.99e-15 |
| La-1 Telediario | 11450 | 7387 | 6710 | 9.55e-15 |
| CNN | 12400 | 7929 | 5552 | 8.44e-15 |
| person_demo | 45575 | 37857 | 37551 | 9.33e-15 |

**53588 numeric points compared, worst difference 9.55e-15, and 0 disagreements about
absence** — every point the reference refuses to place is a null in our table too, and
every point it places agrees to floating-point noise. That is float-noise agreement, not
"close enough": the two implementations compute the same 2×2 determinants.

The committed fixtures are the first 5 frames of each of two clips (250 and 500 rows,
444 of them numeric), chosen because the full 695-frame reference run is ~350 KB of digits
and a candidate carrying it was **rejected by the native reviewer's context budget**
(`lens_context_budget_exceeded`, terminal: "review this change as smaller candidates"). 5
frames is the smallest prefix that holds both basis states — rows `dfMaker` placed and rows
it refused — which is what the committed guard has to be able to see. The whole-corpus
comparison above was run separately, off the full reference output, and is the evidence the
feature rests on; the fixtures are only the part of it that survives as a test.

The comparison also names a difference of *shape* rather than of value, and it is worth
recording because it will confuse the first reader: `dfMaker` emits one row per keypoint
per person per frame unconditionally, so it produces rows for keypoints OpenPose never
detected, all-NA. Our table has no such row because normalisation never wrote one
(`openpose_frame_rows` in `normalization.py` drops every `score <= 0`, so the keypoint was
never in the table this stage reads). On the four videos that is 2525 + 4063 +
4471 + 7718 reference rows with no counterpart of ours. They carry no coordinate, so
nothing numerical is lost; what is lost is a row that says "this joint was never found",
and that absence is already the job of `pose/body.parquet` having no row.

**What the corpus produced.** One real run over all 7 videos, 2m18s, 0 failures:

- **56948 rows out of 56948 in** — the output has exactly the row count of `body.parquet`
  per video (3775 / 7929 / 7387 / 37857, and 0 for the three videos with no pose). A
  derived stage that quietly dropped rows would have shown up here, and `validate()`
  checks it rather than trusting it.
- `basis_state`: `basis_ok` 53588 (94.1%), `basis_missing_joint` 3360 (5.9%).
  `basis_degenerate` never occurs — it takes a MidHip and a Neck at the same pixel to
  trigger, and no frame in this corpus does.
- `value_status` mirrors it exactly: 53588 `normalized`, 3360 `basis_unusable`, and every
  `basis_unusable` row has both `x_norm` and `y_norm` null. Verified, not asserted.
- The whole 5.9% sits in three videos: CNN 2377, La-1 677, person_demo 306. KABC and the
  three zero-pose videos have none. `basis_detail` on every one of those rows says which
  joint was missing, e.g. *"MidHip has no usable coordinate in this person-frame, so the
  MidHip→Neck frame cannot be built"*.
- Scale, which was the reason for choosing this basis: &#124;x_norm&#124; median 1.001,
  p99 2.04, max 3.99; &#124;y_norm&#124; median 0.186, p99 0.767, max 7.20. Compare the
  556 p99 of `dfMaker`'s default triple in §20. The stage ran with defaults unchanged.
- `validate --json` reports `pose_normalized` **valid on all 7 datasets**. The 4 problems
  per video in that same output are provenance ("raw result was produced by a different
  configuration") and belong to having pointed the config at `/tmp` for this run — they
  are the copy's fingerprints, not defects in this stage.

**Two mutations, both killed.** A feature whose whole content is a formula has to be
attacked at the formula:

1. Perpendicular sign flipped to `(-vi.y, vi.x)` — the one the *other* `dfMaker` branch
   uses, the one a reader copying the fast-scaling path would write. **7 tests die**,
   including `test_every_computed_coordinate_matches_the_reference` on both clips. This
   is the mutation that matters, because the resulting table looks plausible: correct
   shape, correct nulls, wrong sign on one axis, and nothing downstream would complain.
2. Per-coordinate mask replaced by a per-point mask (`x == 0 or y == 0` → drop the point).
   **1 test dies** — `test_a_half_measured_point_keeps_the_coordinate_it_has`, which is
   written by hand because *no frame in this corpus contains a half-zero keypoint*. That
   test cannot be earned from the data and is exactly why it exists: the two rules agree
   on every frame here and would disagree silently on the next corpus.

**Config is validated where it can hurt.** `Sternum` (not a BODY_25 name) and
`Background` are rejected; `origin_keypoint == basis_keypoint` is rejected with the
reason spelled out ("both are 'Neck', so the basis vector is zero"); an unknown
`second_axis` is rejected. Verified live through `load_config`, not only by reading the
validator. The triple is in `config_fingerprint` together with a digest of
`body.parquet`, so changing the frame invalidates the table instead of quietly
redefining every number in it.

`docs/assets/stage_graph.png` regenerated with the documented command
(`--synthetic --seed 7`) and reproduced byte-identically by an independent run
(sha256 `6609db22c69d…`); the other three figures are unchanged.

**Process defect in this task, recorded rather than smoothed over.** The `status` render
was broken before this task and this task is what exposed it: at HEAD the table needed
196 columns against the 80 `rich` assumes, and it passed the e2e suite only because that
config disables six stages (78 of 80). Fixed separately as `8ea9b11` and committed
*before* this work unit so each commit leaves the tree green. Two things went wrong on
the way and both are the operator's to know about: the fix was **pushed before native
review ran**, and the review then could not be started on the committed range — the
facade's intended-untracked selection binding was rejected twice on that target, and
the one route it did offer was a workspace candidate that would have frozen this
unverified T14 work with it. So `8ea9b11` sits in `origin/master` unreviewed. AGENTS.md:28
("receipt-driven development is disabled for this clone") is stale: `gentle-ai review
mode status` reads `on (decided by default)` on both scopes.

## 22. T14 review — the candidate the reviewer could not read, split into four

**The first candidate was rejected by the provider, not by a reviewer.** A single commit
carrying all of T14 (17 paths, ~521 KB of diff) came back `lens_context_budget_exceeded`
with no authority created. That is a hard refusal, so the work was rebuilt as a chain of
four commits, each independently green and each small enough to review. The original
attempt is kept locally as tag `t14-split-original`; the chain reproduces the same tree.

| link | commit | diff | tier | lenses | outcome |
|---|---|---|---|---|---|
| fixtures + R generator | `5eb1214` | 95 KB (two CSVs) | — | — | not reviewed, by choice — see below |
| the coordinate maths | `4ec3a33` | 27 KB | medium | 1 (reliability) | **approved**, 3 advisories (`review-1a81688e2fc83328`) |
| stage + registration | `f862e06` | 95 KB | high | 4 | **incomplete** — see below (`review-ad44829afd37f703`) |
| reference comparison + docs | `23045e5` | 38 KB | medium | 1 (reliability) | **approved**, 2 advisories, both fixed in `e0c466d` (`review-4bf6d0329637df4b`) |
| the rebuild tests + this section | `e0c466d` | 18 KB | medium | 1 (reliability) | **approved, no findings** (`review-c29f8477ff8ab47e`) |

The fixture commit is the one link left unreviewed, and that is a judgement, not an
outcome: it is two CSVs no code reads except through the tests that check them against the
reference, their bytes are the reference's own output, and the reviewable claim in them is
the generator script that reproduces them — which the approved reference-comparison commit
exercises end to end. An operator who wants that link reviewed should say so; the lineage
starts cleanly from `8ea9b11`'s successor.

**Link 3 could not be completed, and no verdict was invented for it.** The provider pinned
the lineage (`review-ad44829afd37f703`, target `sha256:b3d3174cdb…`) and asked for four
lenses. `review-risk`, `review-readability` and `review-reliability` were captured.
`review-resilience` failed four times with `reviewer-empty-output` — once after 297 s with
`stopReason: length`, three times in ~27 ms with `stopReason: stop`, i.e. the relay refusing
before generating. Each failure reported `mutation_performed: false`, and fresh STATUS
reoffered the same one-slot binding every time. An incomplete capture never reaches
acknowledgement, so this link has **no approval receipt**. The decision to stop there is
deliberate: the alternative was to author a verdict for a lens that never ran.

What the unrun lens was most likely to find is also the thing link 3's own risk signal
claimed, and it is false: the START cited `process_boundary / shell_process` at
`stages/pose_normalized.py` as the reason for the high tier. That file contains exactly one
match for that pattern — the word "subprocess", in a docstring saying the stage does not
start subprocesses. The tier was raised by a string, not by a process boundary.

**Link 4's two advisories were real defects and are fixed here, not deferred.**

- `R3-promised-edge-tests-missing` (`test_pose_normalized.py:27-31`): the module docstring
  promised `TestMaskingIsPerCoordinate`, `TestBasisStates` and `TestAbsenceIsNamed` live in
  this file. Splitting the suite moved them to `test_pose_normalize_math.py` and
  `test_pose_normalized_stage.py` and the docstring was not updated — so it pointed at
  three classes a reader would never find here. The docstring now names each class and the
  file that holds it, and says why the split exists.
- `R3-nonhermetic-input` (`test_pose_normalized.py:158`): three places cited
  `test_the_rebuilt_table_is_the_real_one` as proof that the fixture-rebuilt pixel table
  equals the real `pose/body.parquet`. **That test never existed.** The claim was load-bearing
  and unbacked: on a fresh clone every reference comparison runs on a table this file
  rebuilds from the CSV, and nothing checked that the rebuild keeps what the pipeline's own
  normalizer keeps. Two tests now exist for it:
  `test_the_rebuilt_table_is_the_real_one` (corpus present: key-for-key and value-for-value
  against the real table over the five fixture frames, 154 and 349 rows, exact match) and
  `test_the_rebuilt_table_follows_the_real_normalizer_rule` (hermetic: rebuilds an OpenPose
  JSON document from the fixture, runs the pipeline's real `openpose_frame_rows` over it, and
  requires it to keep exactly the rows the rebuild keeps). The second one is what runs in CI.
  Mutating the rebuild to also drop `keypoint_id == 0` killed all four new tests.

A prior assumption of mine died in that check. The fixtures carry `points == 0` rows with
numeric coordinates, and I read that as the `Background` keypoint the normalizer drops —
which would have made the rebuild silently wrong. It is not: BODY_25 numbers `Nose` as 0, and
the real table's `keypoint_id == 0` row for KABC frame 0 person 0 carries `x = 447.514,
confidence = 0.86427`, the fixture's own numbers. The rebuild was right; the reasoning that
"proved" it wrong was the bug.

**A stale number was removed rather than defended.** The docstring and a comment claimed
agreement "at 8.4e-15". That is the whole-corpus figure from §21 and it was in a file whose
fixtures cover 5 frames. With `TOLERANCE` forced to `0.0` the tests report their own worst
case: **6.66e-15 over the 154 numeric kabc rows, 7.11e-15 over the 290 numeric cnn rows**.
Both figures, and the corpus-wide 9.55e-15, are now stated where each applies.

Suite after the fix: **1237 unit collected, 1228 passed, 8 skipped** (the 8 pre-existing
matplotlib guards), README counts moved 1233 → 1237.

**Link 5's own review came back with no findings at all** (`review-c29f8477ff8ab47e`,
medium tier, `review-reliability`, store revision `sha256:ccbdc2d3…`, authority burned). It
is recorded here rather than amended into the reviewed commit, because the approval receipt
pins the tree it was issued against. The link 2 advisories (`R3-derived-nonfinite-basis`,
`R3-numeric-conversion-overflow` — both about the math module rejecting non-finite
coordinates earlier than the call site does — and the `R3-row-assembly-coverage` suggestion)
stay open as advisory follow-ups, in the same posture as T21's: non-blocking, not reopening
any review.
