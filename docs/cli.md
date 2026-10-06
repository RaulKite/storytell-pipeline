# CLI reference

Every command the `multimodal-pipeline` console script exposes, the stage-selection flags, and what the exit codes mean.

Back to the overview: [Storytel Pipeline](../README.md).

## Contents

- [Commands](#commands)
- [Stage controls](#stage-controls)
- [Status output](#status-output)
- [Exit codes](#exit-codes)

---

## Commands

| Command | What it does |
|---|---|
| `run` | Process every discovered video, sequentially, one stage at a time |
| `resume` | Continue an interrupted run, reusing every stage whose result is still valid |
| `retry-failed` | Force-recompute exactly the stages that failed, then their dependants |
| `process-video <file>` | Process one file by path (the smoke-test entry point) |
| `status` | Per-video, per-stage state. `--plan` explains every reuse decision, `--json` for scripts |
| `validate` | Re-run semantic validation over artifacts already on disk, without processing |
| `inspect-environment` | Discovered tools, models and GPU/CUDA versions — JSON on stdout, always, no flag needed |

### Stage controls

Stage controls on `run`; `process-video` takes the same four stage controls and takes
the file as its positional argument instead of a flag:

```
--only-stage   acoustic              this stage plus whatever it needs
--from-stage   whisperx              start here
--to-stage     acoustic              stop here       (default: the last stage, elan)
--force-stage  whisperx,acoustic     recompute even if valid
--video        clip.mp4               one file (run only); a bare name resolves
                                     against input.directory
```

`status` and `validate` take `--json` for machine output, and `inspect-environment`
*is* JSON unconditionally; on all three, the human-readable tables go to **stderr**,
so `| jq` and CI capture work without scraping. (`inspect-environment --json` is not a
thing — it exits 2 with typer's "No such option", measured.)

### Status output

`status` numbers its stage columns by position instead of naming them, because eighteen
stage names (or even eighteen abbreviations) do not fit the 80 columns `rich` assumes
when its output is not a tty. The legend under the table gives both mappings — the letter
and the stage behind each index — and each video's id is printed whole on its own line
when it is too long to share one. There are currently 18 stage columns:
`16=stories`, `17=finalization`, `18=elan`, followed by `ov=overall`.
The state legend is `c=completed`, `F=failed`, `P=partial`, `.=pending`,
`r=running`, `s=skipped`. Use `--plan` for the reason behind each reuse decision.

That shape is not cosmetic. The previous `rich` table measured 143 columns against those
80, and `rich` resolves that by stealing width in silence: `video_id` collapsed to a
single `…` and every header to `m…`, exit code 0, so no row said which video it was.
Shortening the headers could not fix it — a real id in this corpus is 72 characters, so
the table never fits. `tests/unit/test_cli_status_layout.py` asserts the structure (which
index holds which stage's mark, and that an id survives as one string) rather than the
presence of a word, because "the id is in the output" is exactly the assertion that
cannot see this failure.

### Exit codes

Exit codes are part of the API:

| Code | Meaning |
|---|---|
| `0` | every video completed |
| `1` | `validate` found a broken artifact, or `status` was given an unknown video id |
| `2` | `run`/`resume` had at least one failed or partial video; or a config/usage error |

`run` uses `2` rather than `1` so a CI job can tell "a video needs attention" apart
from "the pipeline was invoked wrongly".
