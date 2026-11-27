# Multimodal video pipeline — current work

This is the live ODD tracker, not an implementation diary or a parallel specification. Start
with [README](../../README.md), [Architecture](../../docs/architecture.md) and
[Datasets](../../docs/datasets.md); completed plans and review receipts stay in Git history.

## Delivered capabilities

“Built” claims no accuracy, corpus regeneration, desktop usability or hosted CI. Output guides
and the preview MP4 are contracts in [CLI](../../docs/cli.md) and [Datasets](../../docs/datasets.md).

| Stage | State | Current contract |
|---|---|---|
| `persons` | built | Detections/tracks; IDs are not unique-human counts |
| `stories` | built | Transcript-bound candidates, including legitimate empty answers |
| `pose_normalized` | built | Separate body-centred table; measured pixels preserved |
| `openpose` | built | Measured landmarks, opt-in skeleton JPGs and native MP4 |
| `activespeaker` | built | Dense selected-face tables and opt-in native annotated AVI with audio |
| `elan` | built | Up to 12 flat summary tiers, explicit omissions and nonoverlap |

## Open and deferred work

- [ ] **Operator:** check the current 12-tier export in ELAN on a laptop/desktop — labels and media
  playback; a headless preview cannot close usability.
- [ ] **Deferred:** committed non-English synthetic fixture and e2e coverage; historical Spanish
  live-run evidence is not a reproducible CI fixture.
- [ ] **Verification not established here:** hosted CI, whose unit-test workflow certifies no GPU
  tool, corpus accuracy, desktop ELAN or hosted run.
- [ ] **Deferred:** an ASD renderer over normalized selected-face labels; native TalkNet overlays
  remain a distinct interpretation.

## Diarization: Nemotron only

**Decision:** Nemotron (`diarization_nemotron`) is the only diarizer, enabled by default. The
pyannote engine and the diarization × TalkNet fusion are removed from the code, the config schema,
the artifact registry and the ELAN tiers. Assumption: the fusion mixes did not inform a labelling,
and a second engine bought a comparison nobody acted on while doubling artifacts and id namespaces.

**Consequences:** `speaker_assignment` labels the transcript from
`speech/speaker_turns_nemotron.parquet` by maximum overlap (`overlap_s` stays) and owns that column
outright: a cached turn table is used even when the diarizer stage was skipped, and with no table
it writes `speaker_id: null` plus `no_diarization` rather than leaving pyannote's `SPEAKER_00`
behind (raw WhisperX untouched). The `_nemotron` artifact name stays: renaming it would leave old
directories with a populated table beside an empty new one. A config carrying `diarization:` or
`speaker_fusion:` fails naming the replacement, and `config/config.local.yaml` has both. `HF_TOKEN`
stays optional. Alternative not taken: ship fusion disabled for a release — the code would keep
two id namespaces.

## Every stage enabled by default

**Decision:** every tool/analysis stage is on by default, in the code defaults and in
`config/config.example.yaml`, so a config naming only input/output measures everything. Only
preview rendering stays opt-in (`openpose.write_images`, `openpose.write_video`,
`activespeaker.write_video`): it adds no analysis and each reruns its stage.

**Consequences:** `persons` flips to on and `activespeaker` is enabled in the example. To make
that safe, `enabled()` separates *install state* from defects: a missing uv project, worker
script, OpenPose binary, TalkNet checkout or `weights_dir` file skips naming the fix, while a
present but broken tool fails loudly. Unchanged: an unset `weights_dir` may download on first
use — the one prerequisite no pre-flight check sees.

## Transcript contract, retired records and stale tables

**Decision:** a missing transcript is an install state, and a label nobody produced is a defect.
`whisperx` skipping now makes `translation`, `spacy_source`, `acoustic`, `stories` and
`speaker_assignment` skip with `upstream skipped: transcript unavailable` (quoting whisperx's own
reason) instead of failing on an absent table, which ended the video `partial` and exited 2 over
an install state the run had already reported as a skip.

**Consequences:** a translation whose `speaker_id` disagrees with its source segment is `stale` —
kept out of `artifacts`, recorded under `artifacts_not_generated` with the reason, its file
preserved, reported by `validate` under `results[].stale` with exit 0. Retired
`diarization`/`speaker_fusion` records stay in `status.json`, but only the persisted
`current_stage_order` decides `overall_status`. Skip reasons reach `batch_report.json` under
`videos[].skipped` and the closing summary; OpenPose's absent path skips while a present-but-
unusable one fails. Deferred: `linguistic/english/*` derives from that table and is not marked
stale with it. `stages/elan.py` is in that stage's code digest, so new skip wording re-exports
every `.eaf` once — seconds, unlike a model rerun.

## Local full-output profile

**Decision:** the ignored `config/config.local.yaml` adds the three preview switches the example
leaves off (`openpose.write_images` at 640 px, `openpose.write_video`,
`activespeaker.write_video`); every stage is on by default, so it adds rendering, not measurement.
YAML installs no spaCy models and credentials stay in `.env`; “full” means artifact families, and
`image_max_side: null` would request uncapped source-size images. **ASD decision:** preserve native
`speaker/raw/video_out.avi` byte-identically with audio, toggle opt-in (enabling invalidates ASD
reuse, disabling deletes nothing).

## Verification and recovery

- Stable unit/e2e: **2047 passed, 12 skipped, 0 failed** — the 12 are GPU/download guards. The
  four `test_elan` corpus tests pass read-only against an operator dataset and fail without one,
  identically on `master`; each fix in this branch was observed RED then GREEN. Earlier review
  found and rechecked failed-mux data loss, symlink shell injection and false human-note claims.
- Corpus-copy evidence: with the Nemotron environment missing, `speaker_assignment` relabelled a
  copied operator dataset to `speaker_0` from its cached turns; with the turn table removed it
  cleared every `SPEAKER_00` to null while `speech/raw/whisperx.json` stayed byte-identical, the
  stale translation left the manifest and `validate` exited 0. Earlier corpus validation reached
  **7/7 OK**; nothing here establishes accuracy or desktop ELAN.
- Native RDD remains ON; the review workload is split by work unit, not file type, and only
  observed current authority establishes closure — this document proves none of it. Recovery:
  `.git/resume-explanations-checkpoint/` holds the earlier diff, config backups and corpus backup;
  no downloads, paid API requests, input edits or external script edits were made.

## Known limitations and documentation policy

- Person tracks and diarizer labels are not identities; gaps are not absence durations, person
  tables cannot prove frame coverage, stories lack validated truth.
- Historical review advisories (pose assembly, persons, figure errors) without finding detail
  remain unconfirmed, not invented fixes. Mixed annotation keys fail loudly, never coerced.
- Literal percent-escape filenames retain a historical ELAN caveat. Keep topic paths stable
  (architecture = design, datasets = inventory, modalities/ELAN = interpretation), and read no
  historical receipt as proof of a live tool run.
