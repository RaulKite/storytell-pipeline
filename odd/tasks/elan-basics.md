# ELAN basics: truthful summaries and obvious missing analyses

## Intent and authorization

The operator accepted the ELAN audit proposal and requested implementation of the obvious, well-grounded part. Implement accurate identities/states, non-continuous person sightings, four linguistic tables and acoustic segment statistics. Preserve all original measurements. Detailed pose/hand/face representation and dense acoustic numerical signals are deferred; do not imply they are exported.

Base/checkpoint: `ca6e161`; checkpoint branch `checkpoint/elan-basics-ca6e161`; feature branch `feat/elan-basics`. Existing untracked `scripts/make_elan_view.py` is unrelated and must stay untouched. No push, merge, external installation or producer rerun is part of this feature.

## Decisions and constraints

- Preserve existing tier IDs/order, including `gloss_en`; describe it as segment-level English translation through labels/document metadata/docs. Add new tiers rather than silently renaming old tiers.
- Preserve IDs with explicit namespaces; no person/face/audio identity equivalence is inferred.
- Missing/nonfinite scores display unknown, never measured zero. ASD unavailable measurements are not negative speech evidence.
- Person intervals represent sampled sighting runs, not identity or guaranteed continuous presence. Use measured grid evidence where available, report uncertainty otherwise; no median-gap heuristic or extending by stride. Producer detection-only tables cannot prove that all sampled instants are represented. Stored `longest_gap_seconds` is elapsed time between sightings, not exclusively time absent.
- English token and sentence annotation placement uses enclosing segment context when independently aligned times do not exist, explicitly labeled as context placement. Never fabricate token timing.
- Coverage reporting must separate treatment, availability and outcome; XML validity is not complete coverage.
- Heavy modules remain in isolated worker environments. No changes to schemas, artifact paths, producer workers, operator config, environments, raw outputs or inputs.
- The live `gentle-ai review mode status` says on(default), contradicting AGENTS' disabled statement. Preserve the switch and follow observed native candidate review routes; do not label unreviewed work reviewed.
- No strict TDD configuration found by mapping (project/global relevant settings). Resolve this feature to ordinary functional verification with targeted observed failing tests/mutations; this does not enable a repository-wide TDD switch. Exact runner: `uv run --with pytest pytest ... -q -p no:randomly`.
- Delivery strategy: reversible feature-branch-chain of five coherent work-unit commits, no PR/push. Forecast ~1,200–1,800 authored diff lines including tests/docs; per-unit 400 lines is advisory, not a reason to omit tests. Native candidate budgets may require narrower commits; never minify for budget.

## Tasks and route

- [ ] B1 Accurate existing labels, unknown states and translation semantics. Delegated writer: multi-file nontrivial edits; focused tests and docs together.
- [ ] B2 Honest person sighting intervals with explicit secondary input/fingerprint tracking. Delegated writer: multi-file edits and sampling semantics.
- [ ] B3 Four fixed SpaCy tiers with explicit timing context and analysis fields. Delegated writer: source/schema mapping and multi-file edits.
- [ ] B4 Acoustic segment statistics tier, units and unknown values. Delegated writer: multi-file edits.
- [ ] B5 Coverage/validation and verified corpus delivery. Delegated writer for metadata/docs/tests; independent verifier for full suite, real-table checks and generated EAFs.

## Acceptance and checks

- IDs and finite measurements survive their declared summary mapping; missing values are distinguishable from zero.
- Unscored ASD cannot become not-speaking; person gaps cannot be silently presented as measured continuity.
- New linguistic and acoustic analyses appear on flat tiers with transparent placement/units.
- Optional absent/corrupt sources remain isolated and reported; secondary inputs invalidate reuse.
- File metadata/docs say what is summarized/excluded, not every module or every number.
- Tests exercise schema-backed fixtures, reopened EAFs and real corpus tables where available. At least one failing targeted test/mutation is observed for important changed logic.
- Verify source-table hashes before/after export; only derived EAF/status/provenance outputs may change in authorized corpus refresh.
- Focused checks per unit: `uv run --with pytest pytest tests/unit/test_elan.py tests/unit/test_elan_stage.py tests/unit/test_readme_claims.py -q -p no:randomly` (narrow subsets during iteration).
- Closure: `uv run --with pytest pytest tests/unit tests/e2e -q -p no:randomly`, pyflakes over src/workers/tests/scripts, `validate -c config/config.local.yaml --json`, re-open all seven EAFs and check labels/links/counts. No claim of laptop GUI inspection.

## Progress and evidence

Initial mapping completed read-only. Measured base: 12 tier specifications consume 12 of 22 logical normalized tables; TABLE_SCHEMAS contains 20 keys plus two separately declared ASD schemas. Persons producer has detection-only frames; source English tokens lack independent times. No source write or test run yet. B1 is next.

Running authored lines: 0. Work-unit commits/review evidence: pending.

## Rollback

Each work-unit commit restores only its behavior/tests/docs when reverted; original measurements are untouched. Restore generated EAFs from the pre-refresh snapshot if needed, retaining both copies. Feature/checkpoint branches preserve code recovery without history rewrite.
