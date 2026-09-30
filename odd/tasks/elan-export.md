# ELAN export (T22): one `.eaf` per video, one tier per module

Operator request (2026-10-01): a new last pipeline stage that reads the outputs every
worker already produced and writes one ELAN `.eaf` per video, with the modules' outputs
as annotation rows, the video linked so ELAN shows it.

Operator decisions (asked, answered):

| Question | Answer |
|---|---|
| Where does it live | **Last stage of the pipeline**, reading the real outputs of every worker |
| Annotation text | label + key details (`SPEAKER_00 (nemotron)`, `face track 0 · 74/77 act`) |
| Per-frame signals (ASD, pose) | **collapse contiguous runs into blocks** |
| Row layout | **flat, one tier per module** (no hierarchy, no per-speaker tiers) |

Parent decisions (announced, reversible — alternatives noted):

- Library: `pympi-ling` (`pympi.Elan.Eaf`), the standard Python EAF writer. Media through
  `add_linked_file` with BOTH an absolute `file://` URL and a portable
  `RELATIVE_MEDIA_URL` (`../../../input_videos/<name>`), so the `.eaf` opens the video in
  place and survives a folder move.
- Artifact name is fixed: `elan/annotations.eaf` inside each dataset dir (the registry
  maps key→fixed path; one dataset dir per video, so one `.eaf` per video). Alternative
  `<video_id>.eaf` rejected: `VideoPaths.artifact` cannot express per-video names.
- Tier set (12, flat): `words`, `segments_src`, `gloss_en`, `turns_pyannote`,
  `turns_nemotron`, `fusion_pyannote`, `fusion_nemotron`, `asd_speaking` (blocks),
  `face_tracks`, `person_tracks`, `pose_presence` (blocks), `voiced_blocks`.
- Times: seconds → integer milliseconds; a zero-width interval gets +1 ms (ELAN requires
  start < end; documented in code).
- Stage order: after `finalization`; `finalization`'s dependency tuple must exclude `elan`
  (else the manifest's stage that declares `elan` can never declare it produced).
- Reuse: derived pure-Python stage → mixes `python_source_digest` per §31.
- `elan.enabled` defaults **true**: it is cheap pure Python over files already on disk,
  and unlike `persons` it costs no GPU and no download. Reversible by config.
- Adding a manifest artifact moves `MANIFEST_ARTIFACTS` 43→44 and the corpus 42→43 once
  the stage has run everywhere; the README corpus prose and its guard get updated in the
  same work unit (this is exactly the drift §40 just instrumented).

## Tasks

- [x] E1 `elan` module: tables → ELD structure (tiers, blocks collapse, ms conversion), unit-tested against real corpus tables
- [x] E2 `ElanConfig` (enabled, default true) + wiring (`stage_configs`, example yaml) + artifact `elan_annotations` + `ensure_dirs`
- [x] E3 `ElanStage` last in `STAGE_ORDER`; `finalization` deps exclude it; media descriptor with relpath + mimetype by suffix
- [x] E4 run on the corpus, open/validate one `.eaf` (XML sanity + pympi round-trip), check ELAN media URL correctness
- [x] E5 README section + corpus-count prose/guard updates + ratchet; suite green with named mutations
- [ ] E6 ODD receipt + native review per commit + push

## Implementation receipt (writer pass, 2026-09-30)

E1–E3 are implemented and verified; E4 is verified read-only (the corpus was never written to);
E5 and E6 are **not** started because every file they touch is outside this task's edit surface.

### Deviation from the design spec (measured, not preferred)

Tier 11 (`pose_presence`) was specified to group `pose/body.parquet` by **`source_timestamp`**.
That column does not exist in that table. `BODY_SCHEMA` names its PTS-seconds column
`timestamp`, and `OpenPoseStage.frame_timings` fills it from
`source/frame_index.parquet`'s `pts_seconds` — so it *is* source presentation time, under the
other name. `source_timestamp` exists only in `ACTIVE_SPEAKER_FRAMES_SCHEMA`, where it is needed
alongside a second `timestamp` column holding TalkNet's synthetic 25 FPS axis.

Grouped by `timestamp` instead. The design's actual point — group by seconds and never by
`frame_number`, because the pose grid is the source's PTS list while ASD's `frame_number` is its
resampled 25 FPS axis — is preserved, and stated in the docstring. Building the tier as written
would have raised `ArrowInvalid: No match for FieldRef.Name(source_timestamp)` and skipped
`pose_presence` on **every** video, silently, because per-tier read failures are logged and
skipped by design. Caught by a corpus-backed assertion, not by reading.

### Two findings the design did not anticipate (both cost a fix in code)

1. **`pympi.Elan.to_eaf` renames an occupied destination to `<name>.bak` before writing.**
   Handing it a path `NamedTemporaryFile` had already created left a zero-byte
   `.annotations.eaf.*.bak` sibling in every dataset's `elan/` directory, which the artifact
   registry would then have counted inside the `elan_annotations` slot. The writer now reserves
   the temp name, releases it, and lets the library find the destination free. Found by
   `test_no_temporary_file_is_left_behind`, which was written before the behaviour was known.
2. **pympi emits a `default` `TIER` element in every document.** `validate`'s tier count and its
   "declares no tier" guard both have to exclude it, or the count is one high and the guard is
   unreachable (a document with no real tier still contains `default`).

Also: `seconds_to_ms` rounds with `floor(x + 0.5)` rather than `round`, because Python's `round`
is banker's rounding and an exact half-millisecond tie (0.0005 s) would fall to 0.

### Pipeline logic that assumed `finalization` is last — and what it needs

Two places in files outside this task's edit surface, both found by the suite rather than by
reading, both still outstanding:

1. `orchestrator.py` — `STAGE_CLASSES` is built from an explicit class tuple, so `elan` is
   missing from it and `build_stages()` raises `KeyError: 'elan'`. **Every CLI command and all 21
   e2e CLI tests die on it.** Needs: import `ElanStage` and add it to that tuple.
2. `cli.py` — `run()` and `process_video()` default `--to-stage` to the literal `"finalization"`,
   so `multimodal-pipeline run -c config/config.local.yaml` selects a stage range that **excludes
   `elan`**: the feature would be dead on the pipeline's main entry point while every unit test
   stayed green. Needs the default changed to `"elan"` (or `None`). The same file's `status`
   command passes `to_stage=None` and is unaffected.

`orchestrator.enabled_stage_names` also needs `"elan": config.elan.enabled`; without it a
default-true stage is reported enabled by `status` while `Stage.enabled` will report a skip
reason (the mapping's default is `True`, and `pose_normalized`/`speaker_fusion` are listed for
exactly this reason).

The DAG trap the design named was handled in surface: `finalization`'s dependency tuple now
excludes both `finalization` and `elan`, and `dependency_chain`/`dependants_of` were checked for
termination (`dependants_of("finalization") == ["elan"]`, `dependency_chain("elan")` is all 16
predecessors in canonical order).

### Mutation discipline (observed, then restored)

- **(a) `+1 ms` zero-width end removed** → died: `test_a_zero_width_end_widens_by_one_millisecond`,
  `test_a_null_timestamp_lands_at_zero`, and — with the pair-level repair in `interval_ms` also
  removed — 24 more, including `test_a_known_word_lands_on_the_expected_millisecond_pair`,
  `test_face_to_no_face_to_face_collapses_into_three_blocks`,
  `test_presence_blocks_are_split_by_a_low_confidence_frame` and
  `test_no_interval_ever_comes_out_negative_or_inverted`. The two-layer guard is why the isolated
  mutation kills only the two unit-level tests: the pair check catches it downstream.
- **(b) `RELATIVE_MEDIA_URL` dropped (absolute URL only)** → died:
  `test_the_media_descriptor_carries_both_urls_and_they_both_resolve`,
  `test_the_corpus_export_has_every_tier_and_real_words`,
  `test_validate_reports_media_that_is_not_reachable` and 10 more that cannot write the file.

Both restored; the suite is back to its pre-mutation state.

### Out-of-surface failures this pass leaves for the parent (11)

| Test | Why |
|---|---|
| `test_elan.py::TestOrchestratorRegistration` (2) | written as the specification for the `orchestrator.py` edits above |
| `test_persons_config.py::...stage_is_in_the_order_and_the_class_map` | `set(STAGE_CLASSES) == set(STAGE_ORDER)` |
| `test_pose_normalized_schema.py::...registered_in_the_orchestrator_classes` | same invariant |
| `test_speaker_fusion.py::...stage_is_constructible_by_the_orchestrator` | `KeyError: 'elan'` |
| `test_speaker_fusion.py::...registered_in_order_after_its_three_inputs` | asserts `STAGE_ORDER[-1] == "finalization"` — true on base, false by design now |
| `test_readme_claims.py::...artifact_count_it_quotes_is_the_manifest_key_count` | pins `len(MANIFEST_ARTIFACTS) == 43`; registry is 44 |
| `test_readme_claims.py::...corpus_counts_the_prose_states_match_the_manifests_on_disk` | README prose says the registry declares 43 |
| `test_readme_claims.py::...example_never_walks_an_absent_artifact_as_a_file_that_exists` | asserts no dataset lists `elan_annotations`; the corpus has not run the stage yet |
| `test_readme_claims.py::TestDocumentedTestCounts::...real_suite_sizes` | README ratchet: 98 new unit tests collected |

`tests/e2e`: 21 failed / 20 passed against a **green baseline** (the same 42 pass on `HEAD` in a
read-only worktree copy), all 21 from the `KeyError: 'elan'` above.

## Parent closure (E4–E5, 2026-09-30)

The parent verified the writer's work independently rather than trusting the report, then
closed the gaps it named. Measurements, not restatements:

**The out-of-surface gaps were real.** Every one of the 11 failing tests reproduced exactly as
listed before any parent edit. Parent edits, all announced here because they are outside the
delegation surfaces:

- `orchestrator.py`: `ElanStage` imported and added to the `STAGE_CLASSES` tuple;
  `enabled_stage_names` gained `"elan": config.elan.enabled`. The writer's reading of both
  gaps was correct — with only those two edits, the 5 class-map/`KeyError` failures died.
- `cli.py`: `--to-stage` default changed from the literal `"finalization"` to `None` on both
  `run` and `process_video` (`None` = no upper bound, so the default *follows* `STAGE_ORDER`
  instead of contradicting it; `resume` already passed `None`). Without this the feature is
  dead on the main entry point — the "feature muerta" risk the writer flagged is exactly
  right and this is the fourth time this repository has found a default that hard-codes a
  stage name.
- `tests/unit/test_speaker_fusion.py`: the `STAGE_ORDER[-1] == "finalization"` pin became
  `STAGE_ORDER[-2:] == ("finalization", "elan")` plus an inequality — a positional pin on
  "the last stage" was always a claim about the length of a list that grows.
- `tests/unit/test_readme_claims.py`: registry pin 43→44 (the constant moving with the
  registry *is* the guard), prose anchors re-pointed at the new sentences.

**E4, measured on the corpus (operator-authorized writes to `data/processed/`).** All seven
`elan/annotations.eaf` written; every one parses as XML, round-trips through pympi, exports
**12/12 tiers**, and both media URLs resolve from disk (absolute `file://` and
`../../input_videos/<name>`). `validate --json`: `ok: true`, 7/7, zero problems, all
manifests declare 43 artifacts. A third consecutive `--only-stage elan` run re-used (0–1 s)
and `status --plan` reports `valid previous result` for all seven — the 0 s of run two was
the `elan.py` source edit between runs invalidating by design (§31), and the fingerprint
claim was then *tested*: appending one comment line to `elan.py` moved all seven to
`configuration changed`, removing it moved them back. The figure `docs/assets/stage_graph.png`
was regenerated (`--synthetic --seed 7`, byte-identical reproduction verified for the other
three figures). Honest limit: the current model cannot see images, so the new PNG was
verified by the renderer's data path (`stage list len: 17` with `elan` proven in the same
process) and by its bytes changing — not by looking at it.

**One real defect the writer shipped, found by the parent reading corpus output.**
`mimetype_for` mapped every unknown suffix to `video/mp4`. The corpus's `person_demo.avi` is
the test case: its real container (ffprobe, and the pipeline's own `source/metadata.json`)
is QuickTime — `video/mp4` in that descriptor would be a false statement inside every
artifact. The fix declares nothing when the suffix does not say (`UNKNOWN_MIME_TYPE = ""`):
`MIME_TYPE` is an optional `MEDIA_DESCRIPTOR` attribute, an empty one serialises and
round-trips (measured with pympi 1.7), and ELAN plays off the extension either way.
Correct-but-absent beats wrong-and-present. The rewritten test died under restoring the old
catch-all. (Process note: the first mutation attempt silently no-op'd because bare `python`
does not exist on this machine — a `python` invocation is not a command here; `uv run python`
re-ran it properly.)

**E5.** New README section “ELAN export: `elan`” (tier table, measured annotation counts for
all seven datasets, the two-URL rule, the empty-MIME rule, the transcript-fails rule); the
file-by-file table gained its `elan/annotations.eaf` row with measured per-tier counts; the
verbatim block was **regenerated by executing the README snippet** (guard
`test_the_verbatim_snippet_output_is_actually_verbatim` passes against it); prose moved to
17 stages / 43 artifacts / 44 registry keys; the `status` example and its `fifteen`/`15=…`
legend were found stale against the real 17-column render (they already lagged at fourteen
before this feature — corrected to the measured output); counts ratcheted to 1551/42. The
committed stage-graph PNG regenerated. Mutation deaths named and observed (each restored
after, in-memory — a `git checkout` would have discarded the uncommitted feature):
prose 43→44 died, stages 17→16 died, verbatim 43→44 died, ratchet 1551→1550 died.

**Suite after all of it:** unit `1543 passed, 8 skipped` (collected 1551), e2e `42 passed`
(run with the stage registered but before the README-only edits; the docs commits touch no
code path e2e exercises), pyflakes clean, `validate` ok.

