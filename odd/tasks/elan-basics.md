# ELAN basics

## Intent and constraints

Authorized: truthful summaries/person sightings, four SpaCy tiers, acoustic segment statistics and coverage/verification. Detailed pose, landmarks and dense numerical tracks are deferred. Preserve original measurements and unrelated untracked `scripts/make_elan_view.py`.

Branch: `feat/elan-basics`; base/checkpoint: `ca6e161` / `checkpoint/elan-basics-ca6e161`. No push, merge, external installation or producer rerun. Master assumptions: `odd/tasks/multimodal-video-pipeline.md`, ELAN follow-up.

## Decisions

- Retain existing tier IDs/order; add new tiers. `gloss_en` means segment English translation, not word gloss.
- Fusion face_track_id references the SAME TalkNet track_id; engine speakers and YOLO persons are separate. Unknown is not zero; imputation remains visible; selected-face inactivity does not establish global silence. Drop/count untimed rows rather than invent placement at zero.
- Detection-only persons cannot prove sampled coverage. Group only verified source-frame/PTS adjacency; otherwise isolated marks with unknown coverage. No inferred stride or duration extension. Report source elapsed gaps, not absence. Person endpoints are measured PTS; singleton 1 ms is a display minimum.
- Untimed linguistic rows use explicitly labeled segment context, not fabricated alignment.
- ELAN independent tiers cannot overlap: partition at original millisecond boundaries with every active label, retaining logical intervals/membership metadata. Official source: https://www.mpi.nl/tools/elan/docs/manual/Sec_Basic_Information_Annotations_tiers_and_linguistic_types.html (retrieved during implementation).
- Coverage separates treatment, availability and outcome. No producer, schema, artifact, configuration, environment, raw or input changes; valid XML is not completeness.
- Strict TDD is unconfigured; use ordinary checks plus observed failing cases/mutations. Runner: `uv run --with pytest pytest ... -q -p no:randomly`. Independent runners may use UV_OFFLINE=1, PYTHONDONTWRITEBYTECODE=1 and PYTEST_ADDOPTS='-p no:cacheprovider'; normal nested caches/temporary test environments are authorized, not network installations.
- CLI native review is on(default), contrary to AGENTS; facade assess/inspect is unavailable with package-local-binary-missing, no lineage/receipt. Preserve switch, no external repair. Independent verification is required.
- Delivery: six feature-branch work units, no PR. Running authored lines: 4,325 after B2a including tracking. Initial forecast was exceeded by tests/docs; revised forecast: 6,000–7,000. The 400-line/unit heuristic is advisory; never omit or minify tests.

## Tasks and evidence

- [x] B1 Labels/states. Delegated writer + independent verifier. Commit `003b86038353911c9dadb4f5a9d9be511439194a`: 197 focused passed, pyflakes clean, 1613 collected, 86 source hashes unchanged. Baseline/mutations observed. No native review.
- [x] B2 Sightings/dependencies/missing-time refusal. Delegated writer + independent verifier. Commit `60e17a02e8a7e3a22bbda6463c550a5009556daa`: 237 focused passed, 1653 collected, pyflakes clean; person10 split source112–114 /144–239; 92 hashes unchanged; temporary null rows dropped and reported999 preserved. Pre-correction worker full suite: 1637 passed/8 skipped; e2e: 42 passed. Closure checks remain pending. No native review.
- [x] B2a Nonoverlap/membership preservation. Delegated writer + independent verifier. Commit `129d48166a1f43e3c212f8ec3a77d56e8309d006`: 279 focused passed offline, 1695 collected, pyflakes clean; 7 datasets/84 tiers, 463 logical to 492 emitted; source-derived membership unions and 125 unchanged hashes. CNN retains 4 IDs in one bar; KABC Nemotron 3 to 4; person_demo 138 to 157. No native review.
- [ ] B3 Four SpaCy tiers with transparent timing context. Delegated writer. IN PROGRESS.
- [ ] B4 Acoustic segment summaries, units and unknown states. Delegated writer.
- [ ] B5 Coverage/validation, full checks, corpus refresh and docs. Delegated writer + independent verifier.

## Checks and recovery

Per-unit focused trio: test_elan.py, test_elan_stage.py, test_readme_claims.py; measured collection ratchet, scoped pyflakes, temporary schema/EAF probes. Meaningful logic needs observed failures. Source tables remain unchanged.

Closure: full unit+e2e suite; pyflakes src/workers/tests/scripts; corpus validate; seven reopened EAF links, counts, semantic fidelity and nonoverlap. Snapshot derived outputs/hash originals before the authorized ELAN-only refresh. No laptop GUI claim. Roll back work units individually; restore snapshots if needed.

Next: B3. Corpus EAFs remain historical/stale; no refresh yet.
