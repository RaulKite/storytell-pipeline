# Multimodal video pipeline — current work

This is the live ODD tracker, not an implementation diary or a parallel specification.
For how the project works, start with [README](../../README.md),
[Architecture](../../docs/architecture.md) and [Datasets](../../docs/datasets.md).
Completed plans, probes and review receipts remain in Git history, not live authority.

## Delivered capabilities

“Built” does not claim accuracy, corpus regeneration, desktop usability or hosted CI.

| Stage | State | Current contract |
|---|---|---|
| `persons` | built | Optional detections/tracks; IDs are not unique-human counts |
| `stories` | built | Transcript-bound candidates, including legitimate empty answers |
| `pose_normalized` | built | Separate body-centred table; measured pixels preserved |
| `openpose` | built | Measured landmarks, opt-in skeleton JPGs and native MP4 |
| `activespeaker` | built | Dense selected-face tables and opt-in native annotated AVI with audio |
| `elan` | built | Up to 12 flat summary tiers, explicit omissions and nonoverlap |

## Open and deferred work

- [ ] **Operator:** check the current 12-tier export in ELAN on a laptop/desktop.
  Check labels and media playback; a headless preview cannot close usability.
- [ ] **Deferred:** committed non-English synthetic fixture and e2e coverage.
  Historical Spanish live-run evidence is not a reproducible CI fixture.
- [ ] **Verification not established here:** observe hosted CI. Its unit-test workflow
  does not certify GPU tools, corpus accuracy, desktop ELAN or an actual hosted run.
- [ ] **Deferred:** a new ASD renderer over normalized selected-face labels. Native
  TalkNet overlays intentionally remain a distinct interpretation, not that renderer.

## Diarization: Nemotron only

**Decision:** Nemotron (`diarization_nemotron`) is the only diarizer and it is enabled by
default. The pyannote engine and the diarization × TalkNet fusion stage are removed from the
code, the config schema, the artifact registry and the ELAN tiers. Assumption: the fusion
mixes did not inform a labelling, and a second engine in the DAG bought a comparison nobody
acted on while doubling artifacts, tiers and speaker-id namespaces.

**Consequences:** `speaker_assignment` labels the transcript from
`speech/speaker_turns_nemotron.parquet` by maximum overlap (no exclusive timeline to prefer;
`overlap_s` stays in the table). Transcript `speaker_id` is now Nemotron's `speaker_0`
namespace, so re-running an old output directory relabels the transcript — nothing on disk
is deleted. The artifact keeps its `_nemotron` name: renaming it would leave every existing
directory with a populated old table beside an empty new one. A config still carrying
`diarization:` or `speaker_fusion:` fails with an instruction naming the replacement;
`config/config.local.yaml` has both and needs that edit. `HF_TOKEN` stays optional.
Alternative not taken: ship fusion disabled for one release — rejected because the code
would keep carrying two id namespaces.

## Local full-output profile

**Decision:** the ignored `config/config.local.yaml` enables every implemented output
family: the Nemotron diarizer, linguistic variants, normalized pose, ELAN, OpenPose
body/hands/face JPGs, `openpose.write_video` and `activespeaker.write_video`. Example spaCy routing
is explicit, but YAML does not install missing models. Credentials remain in `.env`.
Assumption: “full” means artifact families, not maximum dimensions. Skeleton JPGs
use a 640 px longest side; measured coordinates stay in source pixels. Alternative:
`openpose.image_max_side: null` requests uncapped source-size images.

**ASD decision:** preserve native `speaker/raw/video_out.avi` byte-identically with
audio. Assumption: the operator wants TalkNet's all-face boxes/scores; native smoothing
and the zero-score colour boundary are not normalized selected-face hysteresis or the
configured speaker threshold. Alternative normalized rendering stays deferred.
The new toggle defaults false; enabling invalidates ASD reuse, disabling deletes
nothing. Missing/empty enabled AVI forces model execution even with matching raw JSON.
Safe-named staged audio protects upstream's unquoted shell mux; unsafe temp paths fail
before that invocation. The worker checks video/audio streams and atomically publishes
the AVI before copying new pickles/scenes, preserving prior raw bytes on export failure.
`keep_temp_files` does not control TalkNet's worker temporary directory.

## Output guides and preservation decisions

**Decision:** English `explanation.md` in the batch root and every existing clip folder,
including nested caches and supplementary folders. Assumption: explain the actual output
tree, not only registered slots. Exact generator-marked guides refresh idempotently;
handwritten, non-UTF8 and symlinked guides are preserved. Guides never count as artifacts
or change model fingerprints. Run/resume backfills reuse; status/plan does not write them.

**Skeleton decision:** independent opt-in native MP4, with source audio when present.
Native CFR timing is a preview; authoritative PTS stays in the frame index. Allow at most
three trailing frames (seven observed raw/preview counts: 126/125, 124/121, 240/240,
205/205, 249/249, 299/299, 100/100). Stage raw/tables/images/video before mux/validation;
canonical paths and refusing video overrides protect upstream's shell. Publication after
validation is not a cross-artifact filesystem transaction. Disabling deletes nothing.

## Verification and recovery

- Stable unit/e2e: **1981 passed, 21 skipped, 4 failed**; the four failures are the
  `test_elan` corpus tests, which need local corpus datasets and fail identically on
  `master` (no corpus on this machine). Independent review earlier found and
  rechecked failed-mux data loss, lexical symlink shell injection, video-only provenance,
  truncation and false ownership of human notes; named regressions and sensitivity
  mutations observed RED then GREEN, not mock-only codec evidence.
- Earlier corpus validation reached **7/7 OK** after stale ASD sidecars were replaced; this
  change does not establish accuracy, desktop ELAN or a regenerated corpus.
- Review workload is split by work unit (ASD preservation, skeleton preview, output guides),
  not file type. The guide unit is large because readable contract prose and behavior tests
  are retained, not compressed. Native RDD remains ON; only observed current authority
  establishes closure. This document and historical `7707fb3` prove no new closure.
- Recovery: `checkpoint/resume-explanations-7707fb3`, plus
  `.git/resume-explanations-checkpoint/` contains the initial diff, local config backups,
  hashes, native checks and full corpus backup before metadata renewal. No downloads,
  paid API requests, original input edits or external TalkNet script edits were made.

## Known limitations and documentation policy

- Person tracks/diarizer labels are not identities. Sighting gaps are not absence
  durations; person tables cannot prove sampled-frame coverage.
- Story candidates lack validated ground truth; temperature zero is not service
  determinism. Retained bytes/cache support auditing, not accuracy claims.
- Historical review advisories (pose assembly, persons and figure errors)
  without finding detail remain unconfirmed, not invented fixes.
- Derived source fingerprints can lose readable-module coverage when another module
  is unreadable. Mixed annotation keys fail loudly rather than being coerced.
- Literal percent-escape filenames retain a historical ELAN media-validation caveat.
- English guides explain current contracts; superseded plans/diaries stay in Git,
  not a checked-in archive. Assumption: live contracts/open work matter most.
  Keep guides' topic paths stable; architecture owns design, datasets the inventory,
  modalities/ELAN interpretation. Historical receipts never prove a live tool run.
