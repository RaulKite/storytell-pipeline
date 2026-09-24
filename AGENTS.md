# Agent notes

Operational facts for agents working in this repository. Read this before delegating;
it exists because each item below cost a failed run to learn.

## Decision policy

The operator has asked for autonomous piloting in this repository: choose the best
option instead of asking. Two rules make that safe.

1. **Prefer the reversible choice.** When two options differ in how recoverable they
   are, take the recoverable one and note the alternative in the commit message.
2. **Announce, don't ask.** Make the call, then state it — in the commit message, in
   `odd/tasks/multimodal-video-pipeline.md`, and in the closing summary — with the
   assumption it rests on. Silence about a judgement call is the failure mode, not
   asking.

Three things stay with the operator, because they are irreversible outside this clone
and no amount of code reading resolves them:

- `git push --force`, history rewrite, deleting a remote branch, or merging a PR.
  A normal push to `master` is authorized.
- Anything that leaves this repository: another repo, another machine, package
  publishing, anything costing money or a credential you were not given.
- A destructive change to data the operator produced by hand (their TalkNet scripts,
  their clips, their `data/input_videos/`).

Receipt-driven development is **disabled for this clone** (`gentle-ai review mode
status` reads it, `enable` reverses it). Native review consent prompts therefore do not
appear. That switch is the operator's, not yours: do not re-enable it, and do not
describe a change as reviewed when nothing reviewed it.

## Commands that actually work here

```bash
uv run --with pytest pytest tests/unit tests/e2e -q -p no:randomly   # full suite, ~2 min
uv run --with pytest pytest tests/unit -q -p no:randomly             # unit only, ~25 s
uv run multimodal-pipeline run -c config/config.local.yaml           # full batch
uv run multimodal-pipeline status -c config/config.local.yaml --plan # why each stage reruns
uv run multimodal-pipeline validate -c config/config.local.yaml --json
```

`-p no:randomly` matters: some tests share module-level state and a randomised order
fails them for reasons that have nothing to do with the change. `validate` takes no
`-o` — the output directory comes from the config.

## Delegation

A bounded writer needs a non-empty `## Allowed edit surfaces` in its task with
repository-relative paths, or the run is rejected. Derive them yourself from the plan;
the operator does not author paths.

Do not target a worktree that does not exist. Launching a subagent into a new worktree
fails in this clone with `Select an existing worktree in the same Git clone as this
session`, and left-over review worktrees make that worse. Implement inline when no
valid worktree is available — a routing fallback is acceptable, a stalled launch is not.

## Hard constraints

- The orchestrator never imports `torch`, `pyannote`, `spacy`, or `parselmouth`. Heavy
  tools live in `environments/*` uv projects and are reached through
  `uv run --project <env> python workers/<name>_worker.py`. This is what keeps one
  pinned torch build from breaking another stage.
- Credentials come from `.env` at the repo root (gitignored). `config/config.local.yaml`
  is gitignored too. Never move a secret into YAML, a fixture, a test, or a commit
  message.
- Pins in `environments/*/pyproject.toml` are load-bearing and each one says why. The
  activespeaker torch pin to 2.5.1 is the example: TalkNet calls `torch.load()` without
  `weights_only=`, so torch ≥ 2.6 rejects the 2021 checkpoints. Read the comment before
  "modernising" a version.
- Raw tool output is preserved byte-identical alongside the normalized Parquet. A stage
  that normalizes away the original cannot be re-validated later.
- Tests that hide a bug behind a mock are worse than no test. Where the defect only
  appears against the real tool, run the real tool and record what it said.

## Verification standard

A claim in a commit message, README or ODD document has to be something you observed.
Two commit messages in this repo's history described README content that had not been
written, and one claimed the worker refuses short clips when it silently drops those
frames instead. Fix the artifact, or fix the claim — never leave the claim.

A test you have not seen fail is not evidence. For logic worth trusting, break it once
and name the test that dies.
