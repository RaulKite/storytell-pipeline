# Multimodal video pipeline — current work

This is the live ODD tracker, not an implementation diary or a parallel specification.
For how the project works, start with [README](../../README.md),
[Architecture](../../docs/architecture.md) and [Datasets](../../docs/datasets.md).
Completed plans, probes and review receipts remain in Git history, not live authority.

## Delivered capabilities

“Built” does not claim accuracy, corpus regeneration, desktop usability or hosted CI.

| Stage | State | Current contract |
|---|---|---|
| `speaker_fusion` | built | Agreement tables beside diarization; transcript labels unchanged |
| `persons` | built | Optional detections/tracks; IDs are not unique-human counts |
| `stories` | built | Transcript-bound candidates, including legitimate empty answers |
| `pose_normalized` | built | Separate body-centred table; measured pixels preserved |
| `openpose` | built | Measured landmarks, opt-in skeleton JPGs and native MP4 |
| `activespeaker` | built | Dense selected-face tables and opt-in native annotated AVI with audio |
| `elan` | built | Up to 15 flat summary tiers, explicit omissions and nonoverlap |

## Open and deferred work

- [ ] **Operator:** check the current 15-tier export in ELAN on a laptop/desktop.
  Check labels and media playback; a headless preview cannot close usability.
- [ ] **Deferred:** committed non-English synthetic fixture and e2e coverage.
  Historical Spanish live-run evidence is not a reproducible CI fixture.
- [ ] **Deferred, behavior decision required:** different transcript-assignment diarizer.
  It currently uses pyannote; changing it would redefine existing labels.
- [ ] **Verification not established here:** observe hosted CI. Its unit-test workflow
  does not certify GPU tools, corpus accuracy, desktop ELAN or an actual hosted run.
- [ ] **Deferred:** a new ASD renderer over normalized selected-face labels. Native
  TalkNet overlays intentionally remain a distinct interpretation, not that renderer.

## Local full-output profile

**Decision:** the ignored `config/config.local.yaml` enables every implemented output
family: both diarizers/fusions, linguistic variants, normalized pose, ELAN, OpenPose
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

**Corpus decision:** add missing visuals only, retain three existing MP4s. Initial
publication halted on two rerun JSON mismatches; isolated rechecks matched all seven.
Assumption: new previews must have matching raw measurements before installation.
Seven clips now have skeleton MP4/native ASD AVI; 1343 JPGs and 261 guides were added.
No inputs, native measurement raw bytes or Parquet tables changed. Fourteen prior ASD
raw/sidecar versions live in `speaker/raw/history/`; only sidecars/bookkeeping/logs changed.
The local profile uses the existing repository-local TalkNet mirror and its checkpoints,
not operator scripts or missing `/tmp` weights. Assumption: reuse observed matching native
results rather than execute external tools again; all seven ASD logs confirm no model run.

## Verification and recovery

- Stable unit/e2e: **2126 passed, 12 skipped**; focused current surfaces: **136 passed**.
  Independent review found and rechecked failed-mux data loss, lexical symlink shell
  injection, video-only provenance, truncation and false ownership of human notes.
  Named regressions and sensitivity mutations observed RED then GREEN; not mock-only
  codec evidence. AST/compile and `git diff --check` pass.
- Real seven-clip OpenPose/TalkNet previews decode with video/audio. A genuinely audio-less
  native source also passed with video only; its first concurrent run failed GPU OOM,
  then passed in isolation. Offline build failed: missing hatchling; ruff unavailable.
- Corpus validation initially failed seven stale ASD request sidecars. Genuine matching
  results/sidecars were installed with version archives, then ASD normalization reused raw
  in seven logs (no model run). Final validate: **7/7 OK**, exit 0. All **205 Parquet**
  tables remain semantically and byte-identical; input/native raw hashes unchanged.
  This does not establish accuracy, desktop ELAN or regenerate changed spaCy settings.
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
- Historical review advisories (fusion, pose assembly, persons and figure errors)
  without finding detail remain unconfirmed, not invented fixes.
- Derived source fingerprints can lose readable-module coverage when another module
  is unreadable. Mixed annotation keys fail loudly rather than being coerced.
- Literal percent-escape filenames retain a historical ELAN media-validation caveat.
- English guides explain current contracts; superseded plans/diaries stay in Git,
  not a checked-in archive. Assumption: live contracts/open work matter most.
  Keep guides' topic paths stable; architecture owns design, datasets the inventory,
  modalities/ELAN interpretation. Historical receipts never prove a live tool run.
