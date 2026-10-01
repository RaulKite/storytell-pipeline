# ELAN basics

## Intent

Authorized: truthful existing summaries, honest person sightings, four SpaCy tiers, acoustic segment statistics, coverage and verification. Detailed pose/landmarks and dense acoustic numeric tracks are deferred. Source measurements and unrelated untracked `scripts/make_elan_view.py` stay unchanged.

Branch: `feat/elan-basics`; base/checkpoint `ca6e161` / `checkpoint/elan-basics-ca6e161`. No push, merge, external installation or producer rerun authorized here. Master assumptions: `odd/tasks/multimodal-video-pipeline.md`, ELAN follow-up.

## Decisions

- Retain existing tier IDs/order (including `gloss_en`); add new tiers. English gloss is segment translation, never token alignment.
- Audio engine speaker IDs and YOLO person IDs are separate from TalkNet tracks; fusion face_track_id references the SAME TalkNet track_id.
- Unknown/null/nonfinite measurements are not zero. Negative ASD is about the selected face, not global silence. Both imputed verdicts retain provenance.
- Persons files contain detections, not a complete sampled grid. Use only proven source-frame adjacency; otherwise isolated sightings with coverage unknown. Never infer stride via median timestamps. Stored longest_gap_seconds is elapsed sighting separation, not exclusively absence.
- English tokens/sentences without independent times use explicitly labeled segment context. No fabricated alignment.
- Coverage separates treatment, availability and outcome; valid XML does not establish complete coverage.
- Official MPI ELAN manual, independent tiers (stereotype None): two annotations cannot overlap. Add B2a before new tiers: partition overlapping intervals into disjoint segments carrying ALL active labels/IDs. Do not stagger times or discard simultaneous events. Source: https://www.mpi.nl/tools/elan/docs/manual/Sec_Basic_Information_Annotations_tiers_and_linguistic_types.html (retrieved during implementation).
- No changes to producers, schemas, artifact paths, operator config, environments or raw/input data.
- Strict TDD not configured in inspected settings; ordinary functional verification plus observed failing cases/mutations. Runner: `uv run --with pytest pytest ... -q -p no:randomly`.
- CLI review mode is on(default), contrary to AGENTS prose. Facade assess/inspect failed package-local-binary-missing; no lineage or native receipt. Do not install outside repo or alter switch. Independent verification is required.
- Delivery: reversible feature-branch chain of five work units, no PR. Initial forecast 1,200–1,800 authored lines; B1 alone measured 1,260 including tracking. B2 plus corrections also exceeds the initial unit forecast; B2a adds format conformance. Forecast revised to 4,000–5,000; 400/unit advisory, not a reason to omit tests. No review or delivery claim from size.

## Tasks

- [x] B1 Correct labels/unknown states. Delegated writer (multi-file) plus independent verifier. Commit `003b86038353911c9dadb4f5a9d9be511439194a`.
- [ ] B2 Honest person intervals and explicit secondary-input fingerprints. Delegated writer; IN PROGRESS.
- [ ] B2a ELAN-compatible non-overlapping flat tiers with lossless simultaneous-label representation. Delegated writer plus independent verifier; newly added after official-format constraint discovery.
- [ ] B3 Four SpaCy tiers and transparent timing context. Delegated writer.
- [ ] B4 Acoustic segment summaries with units/unknown states. Delegated writer.
- [ ] B5 Coverage/validation, full checks, regenerated corpus EAFs and docs. Delegated writer plus independent verifier.

## Evidence

B1: 197 focused tests passed independently; pyflakes clean; 1613 unit tests collected. Baseline and mutations exposed unknown-to-zero and unscored-to-negative states; follow-up cases exposed null speakers, false namespaces, negative imputation and engine omission. Temporary reopened EAFs on KABC/CNN/La-1/person_demo: 50/37/90/95 annotations. 86 protected source/input hashes unchanged. Native review unavailable, not approved.

B2 mapping: raw `frames` contains detections only, parameters has no vid_stride or complete measurement grid. Persons frame_number/PTS matches source frame index exactly for four corpus clips; La-1 source frames 2 and 73 have no detection rows. Proven source adjacency may split runs without inference. Full suite and corpus regeneration pending.

## Checks and rollback

Per-unit focused trio: `uv run --with pytest pytest tests/unit/test_elan.py tests/unit/test_elan_stage.py tests/unit/test_readme_claims.py -q -p no:randomly`; measured collection ratchet and pyflakes. Temp EAF/schema probes allowed, no source changes.

Closure: full unit+e2e suite, pyflakes src/workers/tests/scripts, corpus validate, all seven reopened EAFs/media/counts and independent semantic comparisons. Snapshot derived outputs and hash original tables before authorized ELAN-only refresh. No laptop GUI verification claim. Original measurement hashes must remain identical.

Rollback each work-unit commit without unrelated changes; retain pre-refresh EAF copies. Running authored lines: 1,260 after B1. Next: B2 implementation using verified source adjacency.
