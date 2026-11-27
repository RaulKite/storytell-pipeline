# Multimodal video pipeline — current work

This is the live ODD tracker, not an implementation diary or a parallel specification. Start
with [README](../../README.md), [Architecture](../../docs/architecture.md) and
[Datasets](../../docs/datasets.md); completed plans and review receipts stay in Git history.

## Delivered capabilities

“Built” claims no accuracy, corpus regeneration, desktop usability or hosted CI. Output guides
and the preview MP4s are contracts in [CLI](../../docs/cli.md) and [Datasets](../../docs/datasets.md).

| Stage | State | Current contract |
|---|---|---|
| `persons` | built | Detections/tracks plus an opt-in drawn preview; IDs are not unique-human counts |
| `stories` | built | Transcript-bound candidates, including legitimate empty answers |
| `pose_normalized` | built | Separate body-centred table; measured pixels preserved |
| `openpose` | built | Measured landmarks, opt-in skeleton JPGs and native MP4 |
| `activespeaker` | built | Dense selected-face tables and opt-in native annotated AVI with audio |
| `elan` | built | Up to 12 flat summary tiers, explicit omissions and nonoverlap |

## Open and deferred work

- [ ] **Operator:** check the current 12-tier export in ELAN on a laptop/desktop — labels and media
  playback; a headless preview cannot close usability.
- [ ] **Deferred:** committed non-English synthetic fixture and e2e coverage; historical Spanish
  live-run evidence is not a CI fixture.
- [ ] **Deferred:** an ASD renderer over normalized selected-face labels; native overlays stay distinct.
- [ ] **Not established:** hosted CI, corpus accuracy, desktop ELAN.
- [ ] **Deferred:** the persons preview's trim branch moves its audio clock by the trimmed amount
  and fills a leftover gap with silence, but a gap *inside* it stays a timestamp — no measured
  source shows one, and a resample there would be unverified work.

## Diarization: Nemotron only

**Decision:** Nemotron (`diarization_nemotron`) is the only diarizer, enabled by default; pyannote
and the diarization × TalkNet fusion are gone from the code, config schema, artifact registry and
ELAN tiers. Assumption: a second engine bought a comparison nobody acted on while doubling
artifacts and id namespaces, and the fusion mixes informed no labelling.

**Consequences:** `speaker_assignment` labels the transcript from
`speech/speaker_turns_nemotron.parquet` by maximum overlap and owns that column outright: a cached
turn table is used even when the diarizer was skipped, and with no table it writes
`speaker_id: null` plus `no_diarization` rather than leaving pyannote's `SPEAKER_00` behind. A
config still carrying either section fails naming the replacement; `HF_TOKEN` stays optional.

## Every stage enabled by default

**Decision:** every tool/analysis stage is on by default, in the code defaults and in
`config/config.example.yaml`, so a config naming only input/output measures everything. Only
preview rendering stays opt-in (`openpose.write_images` plus the three `write_video` switches): it
adds no analysis and each reruns its stage.

**Consequences:** `persons` flips to on and `activespeaker` is enabled in the example. `enabled()`
separates *install state* from defects: a missing uv project, worker script, OpenPose binary,
TalkNet checkout or `weights_dir` file skips naming the fix, while a present but broken tool fails
loudly as that stage's recorded failure, so the DAG finishes and `status --plan` plans on.

## Person track preview

**Decision:** draw the boxes and ByteTracker ids in `workers/persons_worker.py`, inside the pass
that measured them, and publish `persons/tracks.mp4` (`person_track_video`) under a new
`persons.write_video`; the tables stay authoritative. A switch rather than a derived export, because
a second decode pass could not prove it drew on these frames — so the render shares the tracking
pass and its key joins the stage request.

**Consequences:** frames reach ffmpeg as one argv list (`libx264`, `yuv420p`, odd sides padded to
the chroma grid, source audio muxed only when the source had one) at the rate
`source/frame_index.parquet` gives and `ffprobe` otherwise, and the decoded count must equal
`frames_measured` exactly — OpenPose's three-frame slack serves upstream's renderer, while a short
preview here slides every overlay off its frame. Published as one preflighted step, so a mid-pipe
death keeps the prior run's files; a resume hashes the preview once and reruns on a mismatch.
`extra_args` cannot carry `--output-video`, `--ffmpeg` or `--ffprobe`; disabling deletes nothing.

## Transcript contract, retired records and stale tables

**Decision:** a missing transcript is an install state, and a label nobody produced is a defect.
`whisperx` skipping makes `translation`, `spacy_source`, `acoustic`, `stories` and
`speaker_assignment` skip with `upstream skipped: transcript unavailable` (quoting whisperx's own
reason) instead of failing on an absent table, which ended the video `partial` and exited 2.

**Consequences:** every derived table that copies a speaker out of a segment —
`translation_segments`, the four `linguistic/*` tables and `acoustic/segment_features` — is checked
against the transcript on `segment_id`, and a disagreeing one is `stale`: kept out of `artifacts`,
recorded under `artifacts_not_generated`, reported by `validate` with exit 0, and skipped as an
ELAN tier so no `.eaf` quotes two speakers for one segment; a damaged turn table fails
`speaker_assignment` instead of nulling the transcript. Retired `diarization`/`speaker_fusion`
records stay in `status.json`, but only the persisted `current_stage_order` decides
`overall_status`; `stages/elan.py` sits in that stage's code digest, so new skip wording
re-exports every `.eaf` once — not a model rerun.

## Local full-output profile

**Decision:** the ignored `config/config.local.yaml` adds the four preview switches the example
leaves off (`openpose.write_images` at 640 px plus the three `write_video` flags); every stage is
on, so it adds rendering, not measurement. YAML installs no spaCy model, credentials stay in
`.env`, `image_max_side: null` uncaps source-size images, and the native
`speaker/raw/video_out.avi` stays byte-identical with its audio.

## Verification and recovery

- Stable unit/e2e after the rebase: **2248 passed, 12 skipped, 0 failed** (`-p no:randomly`; the
  four `test_elan` corpus tests read-only against an operator dataset). One earlier run failed
  `validate --json` in e2e and did not reproduce in the runs after it.
- Person preview: real YOLO over the Kimmel clip copied to `/tmp` (read-only input): 126/126
  decoded frames at the frame index's 29.97 fps, h264/yuv420p plus AAC at offset 0.0, `ffmpeg`
  decode exit 0, `validate` ok, and a second run reused the published preview. Overlays were read
  by pixels, not a vision model, and each guard was broken once on purpose so its test died.
  Track accuracy and a regenerated corpus are not established; RDD stays ON and only observed
  current authority establishes closure. No downloads, paid API requests or input edits were made.
  `backup/persons-pre-rebase` is the pre-rebased branch.

## Known limitations and documentation policy

- Person tracks and diarizer labels are not identities; gaps are not absence durations, person
  tables cannot prove frame coverage, stories lack validated truth.
- Historical review advisories (pose assembly, persons, figure errors) without finding detail
  remain unconfirmed, not invented fixes. Mixed annotation keys fail loudly, never coerced.
- Literal percent-escape filenames retain a historical ELAN caveat. Keep topic paths stable
  (architecture = design, datasets = inventory, modalities/ELAN = interpretation), and read no
  historical receipt as proof of a live tool run.
