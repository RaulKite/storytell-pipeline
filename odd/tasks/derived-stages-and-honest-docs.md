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

- [ ] **T1** README: license, four→five, `uv run` in the quick start.
- [ ] **T2** README: narrow the skip-vs-fail claim, naming the OpenPose exception.
- [ ] **T3** `inspect-environment`: warn on unsynced environments, missing `talknet_root`,
      and a spaCy stage that will run `blank`; tests; README claim then holds.
- [ ] **T4** e2e suite: skip cleanly on a machine without OpenPose/ffmpeg 7 instead of
      failing, so a clean install can run the suite.
- [ ] **T5** CI: CPU-only unit workflow, explicit about what it does not cover.
- [x] **T6** `make_fixtures.sh`: require the `flite` filter it needs, unify the OpenPose
      path convention with `openpose.root`, add a test.
- [x] **T7** `activespeaker`: name the dense-sequence fault and log it (closes R3-001) — `e542dbe`.
- [x] **T8** dense frames: `frame_reason` naming which of the four causes left a row unscored — `5505bdb` (code+tests+docs as one work unit), review recorded in `406f1c2`; native review approved (high, 4 lenses), evidence in §9.
- [x] **T9** spaCy model choice sees the language-detection grade, default unchanged — `ba602d7`, evidence in §10.
- [x] **T18** corrupt `whisperx_raw` reads as an `unreadable` sentinel, never as "no grade" (advisory `R4-raw-read-failure-cache`) — `dbe30f3`, evidence in §12.
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

## 13. Execution notes

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
