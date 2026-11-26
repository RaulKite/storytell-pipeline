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
| `stories` | built | Storytelling events + the stories inside them + uncertain spans, all transcript-bound; empty is a valid answer |
| `pose_normalized` | built | Separate body-centred table; measured pixels preserved |
| `openpose` | built | Measured landmarks, opt-in skeleton JPGs and native MP4 |
| `activespeaker` | built | Dense selected-face tables and opt-in native annotated AVI with audio |
| `elan` | built | Up to 15 flat summary tiers, explicit omissions and nonoverlap |

## Open and deferred work

- [ ] **Operator:** check the current 15-tier export in ELAN on a laptop/desktop.
  Check labels and media playback; a headless preview cannot close usability.
- [ ] **Deferred:** committed non-English synthetic fixture and e2e coverage; Spanish
  live-run evidence is not a reproducible CI fixture.
- [ ] **Deferred, behavior decision required:** a different transcript-assignment
  diarizer; it uses pyannote, and changing it would redefine existing labels.
- [ ] **Verification not established here:** observe hosted CI. Its unit-test workflow
  does not certify GPU tools, corpus accuracy, desktop ELAN or an actual hosted run.
- [ ] **Deferred:** a new ASD renderer over normalized selected-face labels. Native
  TalkNet overlays intentionally remain a distinct interpretation, not that renderer.

## Local full-output profile

**Decision:** the ignored `config/config.local.yaml` enables every implemented output
family: both diarizers/fusions, linguistic variants, normalized pose, ELAN, OpenPose
body/hands/face JPGs, `openpose.write_video` and `activespeaker.write_video`. spaCy
routing is explicit, but YAML installs no missing model, and credentials stay in `.env`.
Assumption: “full” means artifact families, not maximum dimensions; skeleton JPGs use a
640 px longest side and coordinates stay in source pixels.

**ASD decision:** preserve native `speaker/raw/video_out.avi` byte-identically with
audio. Assumption: the operator wants TalkNet's all-face boxes/scores; native smoothing
and the zero-score colour boundary are not normalized selected-face hysteresis or the
threshold, and normalized rendering stays deferred. The toggle defaults false: enabling
invalidates ASD reuse, disabling deletes nothing, and a missing/empty enabled AVI forces
model execution even with matching raw JSON. Safe-named staged audio protects upstream's
unquoted shell mux (unsafe temp paths fail first); the worker checks the streams and
publishes the AVI atomically before copying new pickles/scenes, preserving prior raw bytes
on export failure. `keep_temp_files` controls no TalkNet temporary directory.

## Output guides and preservation decisions

**Decision:** English `explanation.md` in the batch root and every existing clip folder,
nested caches and supplementary folders included — the actual output tree, not only
registered slots. Generator-marked guides refresh idempotently; handwritten, non-UTF8 and
symlinked ones are preserved. Guides are not artifacts and change no fingerprint;
run/resume backfills reuse, status/plan writes none.

**Skeleton decision:** independent opt-in native MP4, with source audio when present.
Native CFR timing is a preview; authoritative PTS stays in the frame index. Allow at most
three trailing frames (seven observed raw/preview counts: 126/125, 124/121, 240/240,
205/205, 249/249, 299/299, 100/100). Raw/tables/images/video precede mux/validation,
canonical paths plus refusing video overrides protect upstream's shell, and publication
after validation is not a cross-artifact transaction; disabling deletes nothing.

**Stories v2 decision:** prompt `v2` is the domain expert's definition: a **storytelling
event** (launching, telling, coda) and, separately, the **story** inside it, plus uncertain
spans. The model cites a `segment_id` + quote and outputs no times; the stage maps the
quote's edge words and falls back to the cited segment (`segment`) when an edge word has no
exact alignment, so no time is borrowed. A repeated quote is disambiguated by an optional
`occurrence` (required when it repeats) instead of an unmeetable uniqueness rule, and a
rejected answer is retried as a repair (same prompt, rejected answer, the problems). Only
segments the prompt showed may be cited: a real but unshown id is a breach, not a window
drop, while context-only references still drop and count. `story_evidence` is her list of
explanations (never bare ids) beside `evidence_segment_ids`; schema 2.1. Speakers come from
Nemotron on the operator's decision — overlapping turns `unknown`, coverage the union of a
speaker's turns, a missing source `used: none`, never pyannote — and only labels the prompt
showed are allowed, context included; a speaker her rule lets the transcript name goes in
`notes`, the one documented deviation. Default is one whole-transcript request; a windowed
run drops each event **group** once and counts child stories apart. Raw windows are listed
in the summary and re-read by `validate()`; rejected attempts get unique, redacted files.
Rendering, cache keys and `validate()` follow `docs/modalities.md`; v1's path is removed.
Offsets stay NULL in `words.parquet`, so placement is the token alignment.

**Corpus decision:** add missing visuals only, retain three existing MP4s; seven clips now
have skeleton MP4/native ASD AVI (1343 JPGs, 261 guides) and fourteen prior ASD raw versions
in `speaker/raw/history/`. Assumption: new previews must match rechecked raw measurements
before installation, and reuse observed results rather than re-executing external tools. No
inputs, native raw bytes or Parquet tables changed; the profile uses the local TalkNet mirror.

## Verification and recovery

- Previous candidate, re-measured then: stable unit/e2e **2126 passed, 12 skipped**;
  corpus validate **7/7 OK**, all **205 Parquet** tables unchanged (see Git for details).
- Native RDD remains ON; only observed current authority establishes closure, so this
  document and historical `7707fb3` prove none.
- Recovery: `checkpoint/resume-explanations-7707fb3` plus
  `.git/resume-explanations-checkpoint/` hold the initial diff, config backups, hashes,
  native checks and a full corpus backup before metadata renewal; no downloads, paid API
  requests or edits to inputs or external TalkNet scripts were made.

## Known limitations and documentation policy

- Person tracks/diarizer labels are not identities. Sighting gaps are not absence
  durations; person tables cannot prove sampled-frame coverage.
- Storytelling-event/story candidates lack validated ground truth; temperature zero
  is not service determinism, and retained bytes/cache support auditing, not accuracy.
  Diarized labels are arbitrary ids. The v2 prompt quotes her annotation prompt verbatim
  (`docs/stories-annotation-guidelines.md`; two sections adapted); schema 2.1 is local-only.
- Historical review advisories (fusion, pose assembly, persons, figure errors) without
  finding detail stay unconfirmed, not invented fixes; derived source fingerprints can
  lose readable-module coverage when another module is unreadable. Mixed annotation keys
  fail loudly rather than being coerced.
- Literal percent-escape filenames retain a historical ELAN media-validation caveat.
- English guides explain current contracts; superseded plans stay in Git, not a checked-in
  archive — live contracts and open work matter most. Keep topic paths stable: architecture
  owns design, datasets the inventory, modalities/ELAN; no receipt proves a live run.
