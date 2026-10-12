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
- Final full unit suite: **2006 passed, 12 skipped** (84.61 s).
- `git diff --check` passed; the example config loads. Executable AST comparison
  confirmed the two edited existing test docstrings do not change their logic.
- Independent verifier: **no outstanding findings**, **275 passed, 12 skipped**
  in an isolated scratch tree (including six independent behavioral probes).
  It caught the source-byte fingerprint effect above and an inherited ELAN prose
  error: no transcript inputs normally skip, rather than fail, the stage. Source
  restored and guide corrected. No live inference/corpus/desktop/hosted-CI claim.
- E2E: **42 passed** (313.44 s).
- The whole-scope START exceeded native context before authority creation. The
  operator requested small chunks, so local work-unit commits now replace that scope;
  no push, merge or change to RDD mode is part of this work.
- CLI slice `de1e079`: native low-tier approval was acknowledged/consumed under
  `review-26592e44cb7261bf`; this is not a claim that model lenses ran for that tier.
- Native ASSESS over each following committed boundary reports **no review due**:
  pure docs/history are passive; config comments and assets are medium/under budget.
  Their structural/self checks and independent verification satisfy the returned plan.
- Assessed boundaries (commit / topic): `9091caf` overview/development; `b1e0ddd`
  architecture; `aaad4cd` modalities; `5f8e7c6` datasets; `b4d5e6b` ELAN; `741a992`
  installation; `ee3d3a2` config; `b52d10c` assets; `9d1ffca` derived-stage history;
  `b8a347a` ELAN/story history. These are local chain slices, not release points.
- Remaining guard/workflow slices are assessed separately, never hidden in the
  passive-doc scope. Final branch checks and outcomes belong in their commit messages.
- A candidate-view Git incident was diagnosed read-only: installed Git 2.34.1 rejects
  `worktree list --porcelain -z`. No external runtime, worktree or authority repair
  was attempted. Native outcomes must be read back, not inferred from this diagnosis.
- Checkpoint: `checkpoint/docs-cleanup-e7dc90e`, base `e7dc90e`.
  Work branch: `docs/clean-documentation`.
- Rollback: revert the documentation/guard work unit or restore its files from the
  checkpoint. No input media, raw output or processed dataset needs regeneration.
- Review policy remains operator-owned; read `gentle-ai review mode status` and
  native authority. A reviewed document does not prove a live tool run.
