# Development

Find the code by responsibility, test a focused change, and distinguish local tool
verification from hosted CI. Runtime contracts live in code; current work lives in ODD.

Back to the overview: [Storytel Pipeline](../README.md).

Commands use repository-relative paths: run them from the repository root.

## Contents

- [Repository map](#repository-map)
- [Testing](#testing)
- [Changing a stage](#changing-a-stage)
- [Documentation and work tracking](#documentation-and-work-tracking)

## Repository map

| Directory | Responsibility |
|---|---|
| `src/multimodal_pipeline/` | CLI, configuration, discovery, orchestration, state and artifact contracts |
| `src/multimodal_pipeline/stages/` | Uniform stage implementations and dependency graph |
| `workers/` | Heavy-tool entry points executed outside the root environment |
| `environments/` | Independently pinned `uv` projects/lockfiles |
| `config/` | Committed example and gitignored local settings |
| `tests/unit/` | Algorithm, schema, tool-boundary and documentation checks |
| `tests/e2e/` | CLI subprocess tests with real files/tool availability |
| `tests/fixtures/` | Reference OpenPose JSON and R pose-transform outputs |
| `scripts/` | Fixtures, model installation, synthetic figures, ELAN preview/bundling |
| `docs/` | Eight current topic guides; synthetic illustrations in `assets/` |
| `odd/tasks/` | One compact current-work tracker |
| `data/input_videos/` | Input media; some synthetic fixtures are committed |

Processed datasets and the OpenPose example person clip are gitignored.
See [Architecture](architecture.md) for module links and design rationale.

## Testing

```bash
uv run --with pytest pytest tests/unit -q -p no:randomly
uv run --with pytest pytest tests/e2e -q -p no:randomly
uv run --with pytest pytest tests/unit tests/e2e --collect-only -q -p no:randomly
```

Use collection for current test counts, not remembered totals in prose.
`-p no:randomly` keeps order stable because some tests share module-level state.
Optional tool/data checks can skip; report skips rather than claiming those paths ran.

Tests use real ffmpeg/ffprobe, local HTTP servers, throwaway `uv` projects and
reference outputs where mocks would hide the defect. GPU/model tests do not become
accuracy evidence just because a table has the expected shape. For a behavior fix,
observe the failing case first; for passive docs, check links and current contracts
without inventing a runtime RED/GREEN story.

Hosted [CI](../.github/workflows/unit.yml) runs the **unit suite** on Linux with
ffmpeg, probes the example configuration and checks documentation claims. It does
not run the e2e suite or establish GPU/OpenPose/live-endpoint/model accuracy.
Workflow configuration is not evidence that a particular hosted run succeeded.

## Changing a stage

1. Read `Stage` and the DAG in [`stages/base.py`](../src/multimodal_pipeline/stages/base.py).
2. Declare artifacts in [`artifacts.py`](../src/multimodal_pipeline/artifacts.py)
   and normalized types in [`schemas.py`](../src/multimodal_pipeline/schemas.py).
3. Add the stage to execution order/construction and logging. Dependencies describe
   consumed upstream stages, not merely the order you prefer.
4. Keep heavy imports in a worker/environment; preserve original raw output.
5. Fingerprint configuration, tool/request identity and derived code appropriately.
   Test invalidation, raw reuse, disabled/missing inputs and empty results.
6. Validate semantic output, not just file existence. Update the dataset inventory,
   dependency table and ELAN coverage when the artifact contract changes.

The generic protocol is `enabled`, `prepare`, `run`, `validate`, artifact outputs
and `config_fingerprint`. Domain arithmetic can live in an in-process module;
worker launch details do not belong in pure transforms.

## Documentation and work tracking

- [README](../README.md) routes users; each guide owns one topic. Architecture owns
  decisions, datasets owns paths/joins, modalities owns interpretation, and ELAN
  owns export semantics. Avoid copying run diaries or version inventories into them.
- [Current ODD tracker](../odd/tasks/multimodal-video-pipeline.md) records live work,
  deliberate deferrals, assumptions and verification. Completed historical plans,
  issue/task numbering and receipts are available in Git history instead.
- [Agent notes](../AGENTS.md) define repository permissions, evidence requirements
  and review policy. ODD tracks work; it is not a source of review authority.
- Documentation tests check relative links/anchors, the actual DAG/table registry,
  CLI flags and the compact tracker. Update tests when navigation changes, without
  weakening runtime behavior checks to make prose pass.

Keep changes scoped and recoverable. Never rewrite the operator's clips, input
videos or external TalkNet checkout as part of documentation maintenance.
