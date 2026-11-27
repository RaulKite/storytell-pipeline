# Multimodal video pipeline — current work

This is the live ODD tracker, not an implementation diary or a parallel specification. Start
with [README](../../README.md), [Architecture](../../docs/architecture.md) and
[Datasets](../../docs/datasets.md); completed plans and review receipts stay in Git history.

## Delivered capabilities

“Built” does not claim accuracy, corpus regeneration, desktop usability or hosted CI.

| Stage | State | Current contract |
|---|---|---|
| `persons` | built | Detections/tracks; IDs are not unique-human counts |
| `stories` | built | Transcript-bound candidates, including legitimate empty answers |
| `pose_normalized` | built | Separate body-centred table; measured pixels preserved |
| `openpose` | built | Measured landmarks, opt-in skeleton JPGs and native MP4 |
| `activespeaker` | built | Dense selected-face tables and opt-in native annotated AVI with audio |
| `elan` | built | Up to 12 flat summary tiers, explicit omissions and nonoverlap |

## Open and deferred work

- [ ] **Operator:** check the current 12-tier export in ELAN on a laptop/desktop — labels and
  media playback; a headless preview cannot close usability.
- [ ] **Deferred:** committed non-English synthetic fixture and e2e coverage; historical Spanish
  live-run evidence is not a reproducible CI fixture.
- [ ] **Verification not established here:** observe hosted CI. Its unit-test workflow certifies
  no GPU tool, corpus accuracy, desktop ELAN or hosted run.
- [ ] **Deferred:** a new ASD renderer over normalized selected-face labels; native TalkNet
  overlays remain a distinct interpretation, not that renderer.

## Diarization: Nemotron only

**Decision:** Nemotron (`diarization_nemotron`) is the only diarizer, enabled by default. The
pyannote engine and the diarization × TalkNet fusion are removed from the code, the config
schema, the artifact registry and the ELAN tiers. Assumption: the fusion mixes did not inform a
labelling, and a second engine bought a comparison nobody acted on while doubling artifacts,
tiers and id namespaces.

**Consequences:** `speaker_assignment` labels the transcript from
`speech/speaker_turns_nemotron.parquet` by maximum overlap (`overlap_s` stays); `speaker_id` is
Nemotron's `speaker_0` namespace, so re-running an old directory relabels the transcript and
deletes nothing. The `_nemotron` artifact name stays — renaming it would leave existing
directories with a populated old table beside an empty new one. A config carrying
`diarization:` or `speaker_fusion:` fails naming the replacement; `config/config.local.yaml`
has both and needs that edit. `HF_TOKEN` stays optional. Alternative not taken: ship fusion
disabled for a release — the code would keep two id namespaces.

## Every stage enabled by default

**Decision:** every tool/analysis stage is on by default, in the code defaults and in
`config/config.example.yaml`, so a config naming only input/output measures everything. Only
preview rendering stays opt-in (`openpose.write_images`, `openpose.write_video`,
`activespeaker.write_video`): it adds no analysis and each reruns its stage. Assumption: the
operator wants the full measurement set unless they say otherwise.

**Consequences:** `persons` flips to on and `activespeaker` is enabled in the example. To make
that safe, `enabled()` separates *install state* from defects — a missing uv project, worker
script, OpenPose binary, TalkNet checkout or `weights_dir` file skips naming the fix, while a
present but broken tool still fails loudly; before this, a default run without OpenPose or with
an unsynced environment failed videos. `enabled_stage_names` reports both flags, so a
switched-off stage loses its status column. Unchanged: an unset `weights_dir` may download on
first use — the one prerequisite no pre-flight check sees. Alternative not taken: keep OpenPose's
missing binary a hard error, which is louder but fails a default run.

## Local full-output profile

**Decision:** the ignored `config/config.local.yaml` adds the three preview switches the example
leaves off (`openpose.write_images` at a 640 px longest side, `openpose.write_video`,
`activespeaker.write_video`); every stage is on by default now, so what it adds is rendering,
not measurement. Example spaCy routing is explicit but YAML installs no models; credentials stay
in `.env`. “Full” means artifact families, not maximum dimensions — `openpose.image_max_side:
null` would request uncapped source-size images, and coordinates stay in source pixels anyway.

**ASD decision:** preserve native `speaker/raw/video_out.avi` byte-identically with audio.
TalkNet's all-face boxes and zero-score colour boundary are not the normalized selected-face
labels or the configured threshold, and the toggle stays opt-in: enabling invalidates ASD reuse,
disabling deletes nothing, and a missing enabled AVI forces execution despite matching raw JSON.
Safe-named staged audio protects upstream's unquoted shell mux; the worker checks streams and
publishes the AVI atomically before copying pickles/scenes, which `keep_temp_files` does not
control.

## Output guides and preservation decisions

**Decision:** English `explanation.md` in the batch root and every existing clip folder,
including nested caches. Assumption: explain the actual output tree, not only registered slots.
Generator-marked guides refresh idempotently; handwritten, non-UTF8 and symlinked guides are
preserved, and none counts as an artifact or changes a fingerprint. Run/resume backfills reuse;
status/plan does not write them.

**Skeleton decision:** independent opt-in native MP4 with source audio. Native CFR timing is a
preview; authoritative PTS stays in the frame index, at most three trailing frames allowed
(seven observed raw/preview counts: 126/125, 124/121, 240/240, 205/205, 249/249, 299/299,
100/100). Publication after validation is not a cross-artifact filesystem transaction.

## Verification and recovery

- Stable unit/e2e: **2002 passed, 21 skipped, 4 failed**; the failures are the `test_elan` corpus
  tests, which need local corpus datasets and fail identically on `master`. Earlier independent
  review found and rechecked failed-mux data loss, lexical symlink shell injection, video-only
  provenance, truncation and false ownership of human notes; named regressions and sensitivity
  mutations observed RED then GREEN, not mock-only codec evidence.
- Earlier corpus validation reached **7/7 OK** after stale ASD sidecars were replaced; nothing
  here establishes accuracy, desktop ELAN or a regenerated corpus.
- Review workload is split by work unit, not file type. Native RDD remains ON; only observed
  current authority establishes closure — this document and `7707fb3` prove none.
- Recovery: `checkpoint/resume-explanations-7707fb3` and `.git/resume-explanations-checkpoint/`
  hold the initial diff, config backups, hashes, native checks and full corpus backup. No
  downloads, paid API requests, input edits or external TalkNet script edits were made.

## Known limitations and documentation policy

- Person tracks and diarizer labels are not identities; gaps are not absence durations, person
  tables cannot prove frame coverage, and stories lack validated truth.
- Historical review advisories (pose assembly, persons, figure errors) without finding detail
  remain unconfirmed, not invented fixes. Derived source fingerprints can lose readable-module
  coverage when a module is unreadable; mixed annotation keys fail loudly, never coerced.
- Literal percent-escape filenames retain a historical ELAN media-validation caveat. English
  guides explain current contracts; superseded plans and diaries stay in Git. Keep topic paths
  stable: architecture owns design, datasets the inventory, modalities/ELAN interpretation.
  Historical receipts never prove a live tool run.
