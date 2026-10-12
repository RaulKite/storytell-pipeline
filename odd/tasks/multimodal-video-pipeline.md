# Multimodal video pipeline — current work

This is the live ODD tracker, not an implementation diary or a parallel specification.
For how the project works, start with [README](../../README.md),
[Architecture](../../docs/architecture.md) and [Datasets](../../docs/datasets.md).
Completed plans, numbered issues/tasks, dated probes and old receipts remain in Git
history; they are not current product or review status.

## Delivered capabilities

These capabilities exist in code. “Built” does not claim accuracy, a new corpus
run, desktop usability or successful hosted CI.

| Stage | State | Current contract |
|---|---|---|
| `speaker_fusion` | built | Agreement tables beside diarization; transcript labels unchanged |
| `persons` | built | Optional detections/tracks; IDs are not unique-human counts |
| `stories` | built | Transcript-bound candidates, including legitimate empty answers |
| `pose_normalized` | built | Separate body-centred table; measured pixels preserved |
| `openpose` | built | Measured landmarks and opt-in skeleton renderings |
| `elan` | built | Up to 15 flat summary tiers, explicit omissions and nonoverlap |

## Open and deferred work

- [ ] **Operator:** check the current 15-tier export in ELAN on a laptop/desktop.
  Earlier feedback led to removing verbose linguistic/acoustic-segment tiers and
  windowing frame signals. Check the resulting labels and media playback, not the
  withdrawn tiers. A headless HTML preview cannot close this usability question.
- [ ] **Deferred:** add a committed non-English synthetic fixture and e2e coverage.
  Historical Spanish live-run evidence is not the same as a reproducible CI fixture.
- [ ] **Deferred, behavior decision required:** allow a different diarizer for
  transcript speaker assignment. Today it uses pyannote; switching would change
  existing labels, so separate engine outputs are intentionally preserved.
- [ ] **Verification not established here:** observe a hosted CI run. The workflow
  runs unit tests, not the GPU tools or e2e suite; configuration alone proves no run.

No new runtime features, paid requests, environment upgrades or dataset refreshes
are authorized by this documentation task. A stories tier is not an implicit backlog
item; the current export deliberately omits narrative claims.

## Known limitations

- Person detection tables cannot prove sampled-frame coverage; sighting gaps are not
  absence durations. Tracks and diarizer speaker labels are not identities.
- Story candidates have no validated ground truth; temperature zero is not service
  determinism. Cache/retained bytes support auditing, not accuracy claims.
- Some historical review advisories lack enough finding detail to certify closure
  (fusion/readability, pose row assembly, persons robustness and figure errors).
  Keep them as unconfirmed limitations, not invented fixes or promised features.
- Derived source fingerprints can lose readable-module coverage if another module
  cannot be read. Mixed annotation keys fail loudly rather than being coerced.
- Literal percent-escape filenames have a historical ELAN media-validation caveat;
  do not claim arbitrary filename portability without checking it.
- Historical review lineage dispositions are not product tasks. Inspect native
  authority before review; never reconstruct it from this file or deleted diaries.

## Documentation cleanup

**Decision:** English documentation for users and developers; delete superseded
plans rather than create an archive directory. Git is the recovery path. Assumption:
current contracts and open work matter more than navigating completed task history.
The alternative (a checked-in archive) was rejected because it would retain clutter
and competing statements of current behavior.

Eight guides retain their topic paths for stable navigation. Architecture owns design
decisions; datasets owns the output inventory; modalities and ELAN own interpretation.
Machine-specific counts, historical section references and run diaries were removed.
Four historical ODD documents and the unused stale stage-graph PNG were removed;
the three synthetic schema illustrations remain.

Documentation guards now compare the actual DAG, Parquet registry, CLI/navigation
and compact tracker. Old assertions about obsolete queue IDs, remembered test counts
and this machine's dataset receipts were replaced, not used as product contracts.
Runtime algorithms and operator data remain outside the edit scope. A proposed stage
docstring cleanup was reverted: source-byte fingerprints include prose and could
invalidate story-stage reuse. Its historical source comment stays unchanged; all
`src/`, workers and environment files match the checkpoint.

## Verification and recovery

- Initial guard run against old docs: **15 failed, 80 passed, 3 skipped**. Two
  test-parser false positives (filenames as config keys; aliases with digits) were
  corrected before the guide rewrite. This is documentation-structure evidence,
  not a claim of a runtime bug fixed.
- Final focused docs/contracts/assets: **130 passed, 12 skipped**. Four skips are guides
  without shell examples; eight require optional matplotlib, which is unavailable.
- Pre-slice full unit suite: **2006 passed, 12 skipped** (84.61 s).
- `git diff --check` passed; the example config loads. Executable AST comparison
  confirmed the two edited existing test docstrings do not change their logic.
- Independent verifier: **no outstanding findings**, **275 passed, 12 skipped**
  in an isolated scratch tree (including six independent behavioral probes).
  It caught the source-byte fingerprint effect above and an inherited ELAN prose
  error: no transcript inputs normally skip, rather than fail, the stage. Source
  restored and guide corrected. No live inference/corpus/desktop/hosted-CI claim.
- Pre-slice e2e: **42 passed** (313.44 s). Final combined unit/e2e rerun at
  `d58d1bc`: **2048 passed, 12 skipped**, exit 0 (363.11 s).
- The whole-scope START exceeded native context before authority creation. The
  operator requested small chunks, so local work-unit commits replaced that scope.
  Later authorization adds view cleanup/normal branch push, not merge or changing RDD.
- CLI slice `de1e079`: native low-tier approval was acknowledged/consumed under
  `review-26592e44cb7261bf`; this is not a claim that model lenses ran for that tier.
- Guide/history boundaries `9091caf`–`b8a347a` were assessed individually:
  docs passive; config/assets medium, under budget. Required checks passed.
- Policy/tracker `08dcf87` and guards `535bb23`/`222c086`: medium/under budget;
  self-checks passed. No model review was due for these exact scopes.
- Workflow comment `d58d1bc`: high; three native lenses approved under
  `review-500d9040dc452f4e`, acknowledged/consumed. Fresh ASSESS confirmed
  `closed` (derived), `consumed=true`, `reviewDue=false` (already reviewed).
- With operator authorization, all five session-owned review views were removed;
  two staged snapshots were checkpointed first. Native authority stayed valid/unchanged.
  Assumption: completed task views are disposable, not operator data; no Git upgrade.
  Publication is limited to `docs/clean-documentation`: normal push, no merge/force.
- Checkpoint: `checkpoint/docs-cleanup-e7dc90e`, base `e7dc90e`.
  Work branch: `docs/clean-documentation`.
- Rollback: revert the documentation/guard work unit or restore its files from the
  checkpoint. No input media, raw output or processed dataset needs regeneration.
- Review policy remains operator-owned; read `gentle-ai review mode status` and
  native authority. A reviewed document does not prove a live tool run.
