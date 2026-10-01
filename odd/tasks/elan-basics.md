# ELAN basics

## Intent and constraints

Authorized: truthful summaries/person sightings, four SpaCy tiers, acoustic segment statistics and coverage/verification. Detailed pose, landmarks and dense numerical tracks are deferred. Preserve original measurements and unrelated untracked `scripts/make_elan_view.py`.

Branch: `feat/elan-basics`; base/checkpoint: `ca6e161` / `checkpoint/elan-basics-ca6e161`. No push, merge, external installation or producer rerun. Master assumptions: `odd/tasks/multimodal-video-pipeline.md`, ELAN follow-up.

## Decisions

- Retain existing tier IDs/order; add new tiers. `gloss_en` means segment English translation, not word gloss.
- Fusion face_track_id references the SAME TalkNet track_id; engine speakers and YOLO persons are separate. Unknown is not zero; imputation stays visible; selected-face inactivity does not establish global silence.
- A missing or materially negative timestamp is refused and counted, never placed at second zero. `seconds_to_ms` keeps its clamp as a converter answer only. The negative tolerance is 0.0005 s because half-up rounding makes `[-0.0005, 0]` the exact band the millisecond grid cannot distinguish from zero.
- Detection-only persons cannot prove sampled coverage. Group only verified source-frame/PTS adjacency; otherwise isolated marks with unknown coverage. No inferred stride or duration extension. Report source elapsed gaps, not absence. Person endpoints are measured PTS; singleton 1 ms is a display minimum.
- Null semantics are per column, decided by the producer expression: `ent_type` null is the answer "no entity found" (`workers/spacy_worker.py` writes `token.ent_type_ or None`), `morph` null means unrecorded while `""` means answered-none, and all-null lexical flags are unknown rather than measured false.
- Linguistic timing states: token's own times, times as reported without a provable pairing, or enclosing segment labelled context. English rows claim the translation disclaimer always, and segment-context wording only where the bar actually sits on the segment.
- ELAN independent tiers cannot overlap: partition at original millisecond boundaries carrying every active label, retaining logical intervals/membership metadata. Official source: https://www.mpi.nl/tools/elan/docs/manual/Sec_Basic_Information_Annotations_tiers_and_linguistic_types.html (retrieved during implementation).
- Coverage separates treatment, availability and outcome. No producer, schema, artifact, configuration, environment, raw or input changes; valid XML is not completeness.
- Strict TDD is unconfigured; ordinary checks plus observed failing cases/mutations. Runner: `uv run --with pytest pytest ... -q -p no:randomly`. Runners may export `UV_OFFLINE=1 UV_NO_SYNC=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider'` without changing command strings; runner caches/temp tool environments and pytest temporary directories are authorized, network installs and project sync are not.
- CLI native review reads on(default), contrary to AGENTS; the facade assess/inspect is unavailable (`package-local-binary-missing`), so no lineage and no receipt exist for any commit here. Preserve the switch, no external repair; independent verification is required.
- Delivery: feature-branch chain, no PR. Measured `git diff --shortstat ca6e161..HEAD` after B3: 7 files, 5,838 insertions, 258 deletions (6,096 authored lines). The initial 1,200-1,800 forecast was exceeded by tests and docs; the 400-line/unit heuristic stays advisory and tests were never minified.

## Tasks and evidence

- [x] B1 Labels/states. Delegated writer + independent verifier. Commit `003b86038353911c9dadb4f5a9d9be511439194a`: 197 focused passed, pyflakes clean, 1613 collected, 86 source hashes unchanged; baseline/mutations observed. No native review.
- [x] B2 Sightings/dependencies/missing-time refusal. Delegated writer + independent verifier. Commit `60e17a02e8a7e3a22bbda6463c550a5009556daa`: 237 focused passed, 1653 collected, pyflakes clean; La-1 person10 split source 112-114 / 144-239; 92 hashes unchanged; temporary null rows dropped, reported 999 s preserved. Pre-correction worker full suite 1637 passed/8 skipped, e2e 42 passed. No native review.
- [x] B2a Nonoverlap/membership preservation. Delegated writer + independent verifier. Commit `129d48166a1f43e3c212f8ec3a77d56e8309d006`: 279 focused passed offline, 1695 collected, pyflakes clean; 7 datasets/84 tiers, 463 logical rows to 492 emitted bars; source-derived membership unions and 125 unchanged hashes. CNN keeps 4 IDs in one bar; KABC Nemotron 3 to 4; person_demo 138 to 157. No native review.
- [x] B3 Four linguistic tiers with honest placement. Delegated writer + independent verifier (one launch aborted on a harness error and was relaunched; a relaunch found two further defects, both fixed). Commit `4e36ce420298903757927a8425805008e8ee2b03`: 363 focused passed (parent re-ran), 1779 collected, pyflakes clean. Independent verifier confirmed the producer's `ent_type` collapse and the 228-of-231 measurement, correct placement against real rows, and token IDs surviving projection (KABC 24 to 20 and 23 to 2 bars; La-1 21 to 17 and 25 to 4). All 154 corpus tables hold zero negative timestamps and the seven-corpus rebuild stays 615 annotations, so the negative guard costs no real row. Inputs byte-identical. No native review.
- [ ] B4 Acoustic segment summaries, units and unknown states. Delegated writer + independent verifier. IN PROGRESS.
- [ ] B5 Coverage/validation, full checks, corpus refresh and docs. Delegated writer + independent verifier.

## Checks and recovery

Per-unit focused trio: test_elan.py, test_elan_stage.py, test_readme_claims.py; measured collection ratchet, scoped pyflakes, temporary schema/EAF probes. Meaningful logic needs observed failures. Source tables remain unchanged.

Closure: full unit+e2e suite; pyflakes over src/workers/tests/scripts; corpus validate; seven reopened EAF links, counts, semantic fidelity and nonoverlap. Snapshot derived outputs and hash originals before the authorized ELAN-only refresh. No laptop GUI claim. Roll back work units individually; restore snapshots if needed.

Next: B4. Corpus EAFs remain historical/stale (12 tiers on disk, 16 measured); no refresh yet.
