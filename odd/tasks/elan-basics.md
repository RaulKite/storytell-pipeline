# ELAN basics

## Intent and constraints

Authorized: truthful summaries/person sightings, four SpaCy tiers, acoustic segment statistics and coverage/verification. Detailed pose/landmarks/dense numeric tracks are deferred. Preserve original measurements and unrelated untracked `scripts/make_elan_view.py`.

Branch `feat/elan-basics`; base/checkpoint `ca6e161` / `checkpoint/elan-basics-ca6e161`. No push/merge/external install/producer rerun. Master assumptions: `odd/tasks/multimodal-video-pipeline.md`, ELAN follow-up.

## Decisions

- Retain existing tier IDs/order; `gloss_en` is segment translation, not word gloss. Add new tiers.
- Audio engine speakers and YOLO persons are separate namespaces; fusion face_track_id references the SAME TalkNet track_id.
- Unknown is not zero. Negative ASD concerns the selected face, not global silence; both imputed states retain provenance. Untimed rows are dropped and counted, not placed at zero.
- Detection-only person tables do not establish a sampled grid. Group only proven source-frame/PTS adjacency; otherwise isolated marks, coverage unknown. Do not infer stride or extend by stride. Display reported elapsed sighting separation, never call it time absent. Person intervals retain measured endpoints; 1ms singleton width is a display minimum.
- Untimed English tokens/sentences use labeled enclosing-segment context; no fabricated alignment.
- Official ELAN independent tiers cannot overlap. B2a partitions intervals carrying ALL active labels/IDs and retains logical intervals in metadata; no staggering/discarding. Source: https://www.mpi.nl/tools/elan/docs/manual/Sec_Basic_Information_Annotations_tiers_and_linguistic_types.html (retrieved during implementation). XML/pympi alone did not enforce this.
- Coverage separates treatment/availability/outcome, not a completeness boolean. No producer/schema/artifact/config/environment/raw/input changes.
- Strict TDD not configured; ordinary functional checks plus observed failing cases/mutations. Runner `uv run --with pytest pytest ... -q -p no:randomly`. Independent verification may disable cache using `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`; normal fixture temp writes allowed.
- CLI native review on(default), contrary to AGENTS prose. Facade assess/inspect unavailable (`package-local-binary-missing`), no lineage/receipt. Preserve switch, do not install externally; require independent verification.
- Delivery: six coherent feature-branch units, no PR. Initial forecast 1,200–1,800 exceeded; revised 4,000–5,000. 400/unit advisory, never omit tests/minify. Measured running authored lines 2,910 after B2, including tracking.

## Tasks and evidence

- [x] B1 Correct labels/unknown states. Delegated multi-file writer + independent verifier. `003b86038353911c9dadb4f5a9d9be511439194a`: 197 focused passed, pyflakes clean, 1613 collected, temporary reopened EAF counts 50/37/90/95; 86 protected hashes unchanged. Observed baseline/mutations. Native review unavailable.
- [x] B2 Honest sighting intervals/secondary fingerprints/missing-time refusal. Delegated writer + independent verifier. `60e17a02e8a7e3a22bbda6463c550a5009556daa`: 237 focused passed independently (cache disabled), 1653 collected, pyflakes clean. Real person10 split source112–114 /144–239; 92 protected hashes unchanged. Temp null times dropped, valid siblings retained; reported999 preserved; no-index isolated marks. Worker pre-correction full suite1637 passed/8skipped and e2e42passed; closure suite still pending. No native review.
- [ ] B2a Non-overlapping independent tiers retaining simultaneous labels/logical intervals. Delegated writer + independent verifier. IN PROGRESS.
- [ ] B3 Four SpaCy tiers with transparent timing context. Delegated writer.
- [ ] B4 Acoustic segment summaries/units/unknown states. Delegated writer.
- [ ] B5 Coverage/validation/full checks/corpus EAF refresh/docs. Delegated writer + independent verifier.

## Checks and recovery

Per unit: focused trio `tests/unit/test_elan.py tests/unit/test_elan_stage.py tests/unit/test_readme_claims.py`, measured collection ratchet, scoped pyflakes, temp EAF/schema probes. Important logic must have observed failing cases. No source-table writes.

Closure: full unit+e2e suite, pyflakes src/workers/tests/scripts, corpus validate, all seven reopened EAF links/counts/semantic fidelity/nonoverlap. Snapshot derived outputs and hash original tables before authorized ELAN-only refresh. No laptop GUI claim.

Rollback each work-unit commit independently; retain pre-refresh EAF snapshots. Next: B2a format conformance, then linguistic tiers. Current corpus EAFs still historical/stale; no refresh yet.
