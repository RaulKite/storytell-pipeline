# Feature: honest docs, dataset figures, and the derived stages

Branch: `master` (this clone commits work units directly to `master`; see
`odd/tasks/multimodal-video-pipeline.md` §19 for the autonomous-piloting policy).
Operator instruction, 2026-09-24: *"termina todos los bloques"* — finish the three blocks
inventoried in `multimodal-video-pipeline.md` §20 and in the §21 resume point.

## 1. Authorization

Authorized: all three blocks, as one continuous push of work units.
Not authorized (unchanged): force-push, history rewrite, remote branch deletion, merge,
anything outside this clone, hand-made operator data.

## 2. The three blocks, as inventoried and measured

**Block 1 — five measured falsehoods in README.md.** Each was verified against the tree,
not assumed:

| # | Claim | Reality | Evidence |
|---|---|---|---|
| 1 | `README.md:465` "Not yet declared" | `pyproject.toml:7` declares MIT | read both |
| 2 | `README.md:39` "The four heavy tools" | five environments; `README.md:356` says so | read both |
| 3 | `README.md:28` bare `multimodal-pipeline run` | console script of the root project; `uv run` is what works | `pyproject.toml:18` |
| 4 | `README.md:211` a stage missing prerequisites is skipped | OpenPose **fails** when enabled and the binary is missing | `stages/openpose.py:92,108` |
| 5 | `README.md:37` `inspect-environment` lists *every* skip reason | covers HF_TOKEN, translation, input dir, OpenPose binary, uv-project dirs only | `cli.py:449-464` |

Claim 5 is both a documentation defect and a capability gap: an environment that exists
but has never been `uv sync`-ed is invisible to `inspect-environment`, and so are a missing
`talknet_root` and a spaCy stage running `blank`.

Test counts in README were checked with `--collect-only` and are correct: 681 unit + 30 e2e.

**Block 2 — the six scoped capabilities** (`multimodal-video-pipeline.md` §20.1–20.6).

**Block 3 — debt with no §20 entry**: no CI at all; `tests/e2e/test_cli_smoke.py` *fails*
rather than skips on a machine without OpenPose/ffmpeg 7 (so a clean machine cannot verify
its own install); `scripts/make_fixtures.sh` uses a second OpenPose path convention and
checks for `ffmpeg` but needs the `flite` filter; `activespeaker` emits no warnings at all
(review finding R3-001); frames past the imputable tail are indistinguishable from
`no_face`; and nobody decided whether a `language_detection.status: low` should drive spaCy
model choice at all.

## 3. Design decisions taken here, with the assumption each rests on

1. **OpenPose stays loud.** A stage the operator explicitly enabled must fail rather than
   skip when its binary is gone — a silent skip would hide a broken install behind a green
   batch. The README sentence gets narrowed to name that exception instead of the stage
   getting softer. *Assumption: an operator who sets `openpose.enabled: true` wants pose
   data or an error, not a shrug.* Reversible: the alternative (skip with reason) is one
   small diff if the operator prefers it.
2. **License becomes MIT for real.** `pyproject.toml` has declared MIT all along, so the
   inconsistent artifact is the README. Adding the `LICENSE` file makes the declaration
   match what is already machine-readable. Called out explicitly because a license on a
   public repo is the operator's legal decision, and one commit reverts it.
3. **CI is CPU-only and unit-only.** A workflow that needs a GPU, OpenPose, `/opt/openpose`,
   an `HF_TOKEN` or the translation endpoint cannot run on a hosted runner, so the workflow
   runs `tests/unit` on one OS and nothing else. It is honest about being partial in its own
   name. *Pushing it means GitHub will start consuming Actions minutes for this repo* — the
   repo is already pushed to and Actions minutes are free for public repos; nothing is
   published and no new credential is used.
4. **The dense frame table gets a `frame_reason` column instead of a cleverer null.**
   `face_status` answers "was a face located"; the past-the-tail case answers "TalkNet
   produced no measurement for this tracked face". Those are two questions and one column
   cannot hold both. Bumping the frames `schema_version` is the honest cost.
5. **Language policy is config-gated, default = today's behavior.** A `low` detection
   currently drives model choice, which is what produced the confidently-wrong Catalan POS.
   The reversible move is to expose the gate and record the reason, then flip the default
   once the operator has seen it work.

## 4. Task list

Each task closes with at least one work-unit commit carrying its tests and docs.

- [ ] **T1** README: license, four→five, `uv run` in the quick start.
- [ ] **T2** README: narrow the skip-vs-fail claim, naming the OpenPose exception.
- [ ] **T3** `inspect-environment`: warn on unsynced environments, missing `talknet_root`,
      and a spaCy stage that will run `blank`; tests; README claim then holds.
- [ ] **T4** e2e suite: skip cleanly on a machine without OpenPose/ffmpeg 7 instead of
      failing, so a clean install can run the suite.
- [ ] **T5** CI: CPU-only unit workflow, explicit about what it does not cover.
- [ ] **T6** `make_fixtures.sh`: require the `flite` filter it needs, unify the OpenPose
      path convention with `openpose.root`, add a test.
- [ ] **T7** `activespeaker`: the stage's first warning logs (closes R3-001).
- [ ] **T8** dense frames: `frame_reason` distinguishing no-face / past-tail / unscored.
- [ ] **T9** spaCy model choice gated on language-detection status, default unchanged.
- [ ] **T10** §20.5 `pose_skeletons`: opt-in `--write_images`, artifact + fingerprint, live
      render of one clip.
- [ ] **T11** `scripts/make_dataset_figures.py` + committed `docs/assets/` (stage graph,
      active-speaker strip, speaker-turn strip, pose skeleton from T10).
- [ ] **T12** §20.6 README: install from nothing, what each stage decides, one worked
      example dataset, how to consume it.
- [ ] **T13** §20.1 `diarization_v2`: fuse pyannote turns with per-frame active speaker,
      v1 kept beside it.
- [ ] **T14** §20.4 `pose_normalized`: body-centred basis (dfMaker algebra) in Python,
      validated, explicit no-valid-basis state.
- [ ] **T15** §20.2 `persons`: own uv environment, Ultralytics detect+track, ids kept
      separate from TalkNet's.
- [ ] **T16** §20.3 `stories`: prototype the prompt against the live endpoint, read the
      output, then decide the schema and build the stage.

## 5. Execution notes

- Delegation is live in this clone for read-only task-mode work (a `gentle-ai-explore` run
  mapped the install path this session). Writer launches historically failed here with
  `Select an existing worktree in the same Git clone as this session`; `git worktree list`
  now shows exactly one worktree. Try the writer, fall back inline, and say which happened.
- Verification: `uv run --with pytest pytest tests/unit tests/e2e -q -p no:randomly`.
- Live: `uv run multimodal-pipeline run -c config/config.local.yaml`, then
  `validate -c config/config.local.yaml --json`.
- RDD is `on (decided by default)` for this clone despite what `AGENTS.md` says (see §21 of
  the main feature doc). A review receipt is therefore possible; the absence of one is
  never approval.
