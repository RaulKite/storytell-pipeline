"""The README's checkable claims, checked against the tree.

Documentation rots silently: a sentence that was true when a stage was added stays
true forever in the file while the thing it describes moves. Two commit messages in
this repository's history already claimed README content that had not been written
(see ``odd/tasks/multimodal-video-pipeline.md`` §17), so the fix is not "be careful"
— it is a test that fails when the prose and the tree disagree.

Only claims with a mechanical answer belong here. "The frames table is dense" is a
sentence about behaviour that the stage tests own; "the quick start invokes the CLI
in a way a fresh clone can actually run" is a fact about this file and
``pyproject.toml``, and nothing else checks it.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def fenced_blocks(language: str | None = None) -> list[str]:
    pattern = re.compile(r"^```(\w*)\n(.*?)^```", re.DOTALL | re.MULTILINE)
    return [
        body
        for lang, body in pattern.findall(README)
        if language is None or lang == language
    ]


class TestLicense:
    def test_license_file_exists_and_is_mit(self) -> None:
        license_path = ROOT / "LICENSE"
        assert license_path.is_file(), "README's License section points at a missing file"
        text = license_path.read_text(encoding="utf-8")
        assert text.startswith("MIT License")
        # A license with no copyright line grants nothing to nobody.
        assert re.search(r"^Copyright \(c\) \d{4} .+", text, re.MULTILINE)

    def test_pyproject_declares_the_same_license(self) -> None:
        declared = PYPROJECT["project"].get("license", {})
        assert declared.get("text", "").strip() == "MIT"

    def test_readme_does_not_claim_an_undeclared_license(self) -> None:
        # The sentence this test was written for: pyproject declared MIT for the whole
        # project's life while the README still said nothing had been chosen.
        assert "Not yet declared" not in README


class TestCliInvocation:
    """A bare console-script name is not on PATH in a fresh clone."""

    console_script = PYPROJECT["project"]["scripts"]["multimodal-pipeline"]

    def test_console_script_name_matches_the_readme(self) -> None:
        assert self.console_script == "multimodal_pipeline.cli:app_main"

    @pytest.mark.parametrize("index", range(len(fenced_blocks("bash"))))
    def test_bash_blocks_invoke_the_cli_through_uv_run(self, index: int) -> None:
        block = fenced_blocks("bash")[index]
        offenders = [
            line.strip()
            for line in block.splitlines()
            # A line that calls the CLI without `uv run` in front of it, ignoring
            # shell-variable assignments like `HF_TOKEN=... uv run ...`.
            if re.search(r"(?<![\w./-])multimodal-pipeline\s+\w", line)
            and "uv run" not in line
            and not line.lstrip().startswith("#")
        ]
        assert not offenders, (
            "the quick start must be copy-pasteable in a fresh clone, where the "
            f"console script is not yet on PATH: {offenders}"
        )


class TestEnvironmentCount:
    """The count in the prose is measured, not remembered."""

    def environments(self) -> list[str]:
        return sorted(
            path.name for path in (ROOT / "environments").iterdir() if path.is_dir()
        )

    def test_readme_syncs_every_environment_directory(self) -> None:
        synced = set(re.findall(r"environments/(\w+)\s+&& uv sync", README))
        missing = set(self.environments()) - synced
        assert not missing, f"environments present but never synced by the quick start: {missing}"

    def test_readme_counts_the_environments_it_ships(self) -> None:
        words = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
        count = len(self.environments())
        expected = words[count]
        assert f"[Why {expected} environments]" in README
        assert f"{expected} environments" in README.lower()
        for other in words.values():
            if other != expected:
                assert f"Why {other} environments" not in README

    def test_no_stale_heavy_tool_count_remains(self) -> None:
        # "The four heavy tools" survived the fifth environment being added, because the
        # sentence lives far away from the heading that counts them.
        count = len(self.environments())
        words = {3: "three", 4: "four", 5: "five", 6: "six"}
        for number, word in words.items():
            if number != count:
                assert f"{word} heavy" not in README.lower()


class TestDocumentedTestCounts:
    """The test counts in the README are the collected counts, not remembered ones.

    They had already drifted twice by the time this ratchet was written (681/30 printed
    against 764/38 collected). A number that only a human updating prose can keep true is a
    number that will be wrong, so it is compared against pytest's own collection.
    """

    def counts(self) -> tuple[int, int]:
        import subprocess
        import sys

        collected = []
        for suite in ("tests/unit", "tests/e2e"):
            out = subprocess.run(
                [sys.executable, "-m", "pytest", suite, "-q", "--collect-only", "-p", "no:randomly"],
                cwd=ROOT, capture_output=True, text=True, check=True).stdout
            match = re.search(r"(\d+) tests? collected", out)
            assert match, f"could not read a collected count for {suite}: {out[-400:]}"
            collected.append(int(match.group(1)))
        return collected[0], collected[1]

    def test_readme_states_the_real_suite_sizes(self) -> None:
        unit, e2e = self.counts()
        assert f"pytest tests/unit -q     # {unit} tests" in README, (
            f"README does not say {unit} unit tests")
        assert f"pytest tests/e2e -q      # {e2e} tests" in README, (
            f"README does not say {e2e} e2e tests")
        assert f"tests/unit/                {unit} tests" in README
        assert f"tests/e2e/                 {e2e} CLI-driven tests" in README


class TestDocumentedCommandsExist:
    """Every command in the Commands table is a real typer command."""

    def test_table_commands_are_reachable_from_the_cli(self) -> None:
        import typer

        from multimodal_pipeline.cli import app

        assert isinstance(app, typer.Typer)
        names = {
            command.name or command.callback.__name__
            for command in app.registered_commands
        }
        assert names, "the CLI exposes no registered commands to check against"
        table = re.search(r"## Commands\n(.*?)\n\n", README, re.DOTALL)
        assert table, "the Commands section moved; update this test's anchor"
        documented = set(re.findall(r"^\| `([a-z-]+)`", table.group(1), re.MULTILINE))
        assert documented, "no commands parsed from the table"
        assert documented <= names, f"README documents commands that do not exist: {sorted(documented - names)}"


class TestDocumentedFlagsExist:
    """Flags the README shows must exist on the command it shows them on.

    This test was written after measuring that ``inspect-environment --json`` exits 2
    with typer's "No such option": the README had listed inspect-environment beside
    status/validate as taking ``--json``, and it never did — it is JSON unconditionally.
    """

    def options(self, command: str) -> set[str]:
        import typer.main

        from multimodal_pipeline.cli import app

        # typer 0.27 vendors click as typer._click, so never `import click` here: this
        # environment has typer without a top-level click.
        group = typer.main.get_command(app)
        target = group.commands[command]
        return {opt for param in target.params for opt in param.opts}

    def test_stage_control_flags_exist_on_run(self) -> None:
        # `--([a-z-]+)` would also match the `---` horizontal rules of the markdown.
        documented = set(re.findall(r"^--([a-z][a-z-]*)\s", README, re.MULTILINE))
        assert {"only-stage", "from-stage", "to-stage", "force-stage", "video"} <= documented
        real = self.options("run")
        missing = {f"--{name}" for name in documented} - real
        assert not missing, f"README shows run flags that do not exist: {sorted(missing)}"

    def test_process_video_takes_the_four_stage_controls_not_video(self) -> None:
        # The README block is shared by run and process-video and says so; --video is
        # run-only because process-video takes the file positionally. If the CLI ever
        # grows --video on process-video, update the README block and this test together.
        real = self.options("process-video")
        for flag in ("--only-stage", "--from-stage", "--to-stage", "--force-stage"):
            assert flag in real, f"process-video lost {flag}"
        assert "--video" not in real

    def test_json_flag_claims_match_the_commands_that_have_it(self) -> None:
        for command in ("status", "validate"):
            assert "--json" in self.options(command), f"{command} lost --json"
        # The claim this was written for: inspect-environment never had --json.
        assert "--json" not in self.options("inspect-environment"), (
            "inspect-environment now accepts --json; the README's flag prose and this "
            "test should both be updated together"
        )
        assert "no flag needed" in README, (
            "the Commands table must say inspect-environment is JSON unconditionally"
        )

    def test_plan_flag_exists_on_status(self) -> None:
        assert "--plan" in self.options("status")


class TestDatasetLayoutMatchesTheRegistry:
    """The tree diagram lists artifacts the registry actually declares."""

    def test_every_parquet_in_the_diagram_is_registered(self) -> None:
        from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT

        listed = set(re.findall(r"([a-z_]+)\.parquet", README))
        known = {
            Path(relative).name.removesuffix(".parquet")
            for relative in ARTIFACT_LAYOUT.values()
            if relative.endswith(".parquet")
        }
        unknown = listed - known
        assert not unknown, f"README names Parquet artifacts that are not registered: {sorted(unknown)}"


class TestInstallChainNamesRealIdentifiers:
    """The install section must name settings that exist, and no others.

    §20.6 asks for a chain a clean machine can follow in order. The failure mode of that
    prose is not a missing step, it is a step written against a setting nobody implemented
    — an env var that nothing reads, or a config key the schema rejects as unknown.
    """

    SECTION = "## Install from nothing"

    def section(self) -> str:
        start = README.index(self.SECTION)
        rest = README[start + len(self.SECTION):]
        end = rest.index("\n---\n")
        return rest[:end]

    def test_the_section_exists_and_covers_every_prerequisite_class(self) -> None:
        section = self.section()
        for needle in ("uv", "ffmpeg", "flite", "openpose.root", "HF_TOKEN",
                       "talknet_root", "install_spacy_models.sh", ".env.example",
                       "config/config.example.yaml"):
            assert needle in section, f"the install chain no longer names {needle}"

    def test_every_environment_variable_it_names_is_a_real_one(self) -> None:
        """Against .env.example, the schema's holder field, and the .env opt-out switch.

        Every other env var in this repository is an interpolation the *user* invents in
        their own YAML, so the chain may not name one that the shipped template lacks —
        an operator who exports it gets silence, which is the defect class the whole
        install section exists to prevent.
        """
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        template_keys = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", example, re.MULTILINE))
        from multimodal_pipeline.config import DiarizationConfig

        shell_vars = {"PATH"}  # a shell variable, not a credential
        known = (template_keys | {DiarizationConfig().hf_token_env}
                 | {"MULTIMODAL_PIPELINE_NO_DOTENV"} | shell_vars)
        # Only an underscore-joined name in backticks reads as "export this", so that is
        # the shape policed here; `HF_TOKEN` is matched even without a second underscore.
        named = set(re.findall(r"`([A-Z][A-Z0-9]*_[A-Z0-9_]*)`", self.section()))
        invented = named - known
        assert not invented, f"the install chain names env vars nothing reads: {sorted(invented)}"
        assert "HF_TOKEN" in named, "the credential the whole chain exists to supply went missing"

    def test_every_config_setting_it_names_parses_in_the_schema(self) -> None:
        """A dotted path like `openpose.root` must be a real field on a real sub-model."""
        from multimodal_pipeline.config import PipelineConfig

        config = PipelineConfig.model_validate(
            {"input": {"directory": "/tmp/in"}, "output": {"directory": "/tmp/out"}}
        )
        named = set(re.findall(r"`([a-z_]+\.[a-z_]+)`", self.section()))
        # Only a wholly backtick-quoted `section.field` counts, so filenames that happen to
        # look like paths (`…/openpose/openpose.bin`) are not mistaken for config keys. A
        # name whose section is not a real stage section is skipped for the same reason.
        missing = []
        for dotted in sorted(named):
            section, field = dotted.split(".")
            if not hasattr(config, section):
                continue
            sub = type(getattr(config, section))
            if hasattr(sub, "model_fields") and field not in sub.model_fields:
                missing.append(dotted)
        assert not missing, f"the install chain names config keys the schema rejects: {missing}"
        # Teeth: the check above is vacuous if the chain stopped naming settings at all.
        assert {"openpose.root", "activespeaker.talknet_root"} <= named, named


class TestInspectEnvironmentWarningsAreQuotedVerbatim:
    """§20.6: a README that drifts from the tool is worse than a terse one.

    The pre-flight warnings are the install chain's payoff, so the README quotes them as
    text. Two halves keep that honest:

    * every *whole* message built in `cli.py`'s four `*_warnings` functions is still
      quotable there, pulled from the AST rather than typed by hand, so renaming a message
      retires the assertion instead of leaving it to rot;
    * the warnings the shipped example config actually produces are quoted, produced by
      calling the command during the test.

    `tests/unit/test_environment_warnings.py` checks *which* situations warn; nothing else
    checked that the README still quoted the warning it produced.
    """

    MIN_FRAGMENT = 30

    def static_messages(self) -> list[str]:
        """Whole literal messages from the warning functions, docstrings excluded."""
        import ast

        tree = ast.parse((ROOT / "src" / "multimodal_pipeline" / "cli.py").read_text(encoding="utf-8"))
        fragments: list[str] = []
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef) and node.name.endswith("_warnings")):
                continue
            docstring = (node.body[0].value.value
                         if node.body and isinstance(node.body[0], ast.Expr)
                         and isinstance(node.body[0].value, ast.Constant)
                         and isinstance(node.body[0].value.value, str) else None)
            for inner in ast.walk(node):
                if isinstance(inner, ast.JoinedStr):
                    # Adjacent literals of one f-string are one message: the first two
                    # segments of `f"{cfg.hf_token_env} is not set: …"` are meaningless
                    # apart, so they are joined here rather than compared separately.
                    buffer = ""
                    joined = []
                    for part in inner.values:
                        if isinstance(part, ast.Constant) and isinstance(part.value, str):
                            buffer += part.value
                        elif buffer:
                            joined.append(buffer)
                            buffer = ""
                    if buffer:
                        joined.append(buffer)
                    fragments.extend(joined)
                elif isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    if inner.value != docstring:
                        fragments.append(inner.value)

        kept = set()
        for fragment in fragments:
            collapsed = " ".join(fragment.split())
            # A fragment that begins or ends on punctuation sits next to an interpolation
            # ("… not found under ", ": English linguistics fall back to '"): it is a
            # variable's neighbourhood, not a sentence a document can quote.
            if len(collapsed) >= self.MIN_FRAGMENT and collapsed[0].isalnum() and collapsed[-1].isalnum():
                kept.add(collapsed)
        return sorted(kept)

    def test_the_probe_yields_the_messages_it_claims_to_check(self) -> None:
        messages = self.static_messages()
        # Guard against the class passing because a refactor made every message computed:
        # then there would be nothing to check and the ratchet would be silently dead.
        assert len(messages) >= 3, f"only {messages} quotable warnings found in cli.py"
        assert any("diarization will be skipped" in m for m in messages)
        assert any("translation will be skipped" in m for m in messages)

    def test_every_warning_message_is_still_in_the_readme(self) -> None:
        normalised = " ".join(README.split())
        missing = [m for m in self.static_messages() if m not in normalised]
        assert not missing, (
            "inspect-environment says things the README no longer quotes (or the wording "
            f"drifted): {missing}"
        )

    def test_the_example_config_warnings_are_quoted_verbatim(self, monkeypatch) -> None:
        """Produce the warnings the install chain tells you to expect, and match its words.

        Called through the same function the command prints, rather than as a subprocess:
        the strings are the contract, and the command's own output shape is covered in
        `tests/e2e/test_cli_smoke.py`.
        """
        from multimodal_pipeline.cli import _environment_warnings
        from multimodal_pipeline.config import load_config

        # The repository's own .env must not decide what this probe sees.
        monkeypatch.setenv("MULTIMODAL_PIPELINE_NO_DOTENV", "1")
        for name in ("HF_TOKEN", "LITELLM_BASE_URL", "LITELLM_API_KEY", "LITELLM_MODEL"):
            monkeypatch.delenv(name, raising=False)

        warnings = _environment_warnings(load_config(ROOT / "config" / "config.example.yaml"))
        assert any("HF_TOKEN is not set" in warning for warning in warnings), warnings
        normalised = " ".join(README.split())
        # The input-directory warning ends in the config's own path, so it is quoted by its
        # stable prefix plus the value the shipped example actually holds.
        unquoted = [w for w in warnings
                    if w not in normalised
                    and not (w.startswith("input directory does not exist")
                             and "/data/videos" in README)]
        assert not unquoted, f"warnings the shipped config produces but the README omits: {unquoted}"


class TestDenseTableHonesty:
    """The single most misleading sentence in this file used to be "one row per 25 FPS frame".

    Measured on the KABC clip: 126 source frames, 105 rows in
    `speaker/active_speaker_frames.parquet`, and only 1 row where the grid second equals the
    source second. A reader who took "frame" for a source frame joined pose to the wrong
    frames on 102 of 105 rows and got numbers that looked fine. So the qualifier is asserted
    next to the claim, in both places that make it.
    """

    def test_the_grid_is_named_wherever_the_dense_claim_is_made(self) -> None:
        """Every 25 FPS claim *about rows* must say the grid is not the source frames.

        A bare "at 25 FPS" sentence about TalkNet's sampling rate is fine; "one row per
        25 FPS frame" without the qualifier is the sentence that cost a reader a wrong join.
        """
        mentions = [line for line in README.splitlines() if "25 FPS" in line]
        assert mentions, "the active-speaker table is no longer described as 25 FPS"
        claims = [line for line in mentions if "row" in line or "dense" in line]
        assert claims, "no per-row 25 FPS claim left to police"
        unqualified = [line.strip()
                       for line in claims
                       if "grid" not in line and "source" not in line
                       and "working timeline" not in line]
        assert not unqualified, (
            "a 25 FPS claim with no note that the grid is not the source frames: "
            f"{unqualified}"
        )

    def test_the_join_rule_is_stated(self) -> None:
        """The rule must be stated in the direction that is true, not merely mentioned.

        A substring test on `never on `frame_number`` passes when the sentence is inverted
        to "join on `frame_number`, never on `source_timestamp`" -- both phrases are then
        present and the README tells a consumer the exact wrong thing. So the assertion is
        on the sentence shape: somewhere the README must pair "join on" with
        `source_timestamp` and forbid `frame_number`, and must never state the reverse.
        """
        join_forward = re.search(
            r"join on `source_timestamp`,? never (?:on )?`frame_number`", README)
        assert join_forward, (
            "README no states the pose/active-speaker join rule in the true direction "
            "(join on `source_timestamp`, never `frame_number`)")
        join_backward = re.search(
            r"join on `frame_number`,? never (?:on )?`source_timestamp`", README)
        assert join_backward is None, (
            "README states the inverted join rule: `frame_number` is the ASD stage's 25 FPS "
            "grid index, not a source frame, so joining pose to it on `frame_number` "
            "silently misaligns every non-25 fps clip")
        assert "source_timestamp" in README

    def test_the_frame_columns_come_from_the_grid_not_the_source(self) -> None:
        """Pinned against the code, not the prose: frame_number *is* the 25 FPS index."""
        source = (ROOT / "src" / "multimodal_pipeline" / "stages" / "activespeaker.py").read_text(encoding="utf-8")
        assert '"frame_number": row.get("frame_25fps")' in source, (
            "the stage no longer maps frame_number onto the worker's 25 FPS index; if the "
            "grid became the source frame, the README's join rule and this test both change"
        )
        worker = (ROOT / "workers" / "activespeaker_worker.py").read_text(encoding="utf-8")
        assert "OUTPUT_FPS = 25" in worker


class TestWorkedExampleMatchesTheManifest:
    """The §20.6 worked example must not promise a file the datasets do not have.

    `data/processed/` is gitignored, so this checks the *claim* rather than the bytes: the
    README asserts an exact artifact count per dataset and says which registered artifacts
    are absent from those datasets. Both halves are cheap to state and expensive to get
    wrong, because a reader follows the README's file list literally.
    """

    def test_the_artifact_count_it_quotes_is_the_manifest_key_count(self) -> None:
        # The registry declares 43 manifest candidates; exactly one is opt-in
        # (`pose_images_raw`, gated on `openpose.write_images`), and the README's corpus
        # sentence says every dataset on this disk lists the remaining 42. The old shape of
        # this test hardcoded a `never_produced` set of seven; a full batch run turned that
        # set into fiction, which is why the prose guard below now parses the sentence and
        # re-checks it against every manifest instead of trusting a constant.
        from multimodal_pipeline.artifacts import MANIFEST_ARTIFACTS

        assert len(MANIFEST_ARTIFACTS) == 43
        assert "pose_images_raw" in MANIFEST_ARTIFACTS

    def test_the_corpus_counts_the_prose_states_match_the_manifests_on_disk(self) -> None:
        """The README states per-dataset artifact counts for *this* machine's corpus, in two
        halves: arithmetic the sentence owes itself, and arithmetic it owes the bytes.

        The prose half runs everywhere and needs no corpus — it parses every number out of the
        sentences and checks them against each other (plain + other == registry, the re-run
        converting only some of the declared-absent, the dataset counts adding up), with the
        registry constant consulted exactly once, to confirm the README's declared total is the
        registry's. The byte half then re-checks those same parsed numbers against every manifest
        under `data/processed/`, and skips without one — the shape that previously let the guard
        skip past a fresh clone entirely (R3-corpus-skip).

        Drift therefore dies from either side: edit the sentence and it contradicts the registry
        or disk; re-run the pipeline over the corpus and disk contradicts the sentence. What it
        guards against is not hypothetical — the prose previously asserted "absent from every
        dataset under data/processed/", a universal claim already false when read, because the
        KABC manifest declared two of the supposedly-absent artifacts and no test could see a
        manifest that disagreed with a sentence.
        """
        root = ROOT / "data" / "processed"
        manifests = sorted(root.glob("*/manifest.json"))

        # 1. What the prose says, extracted rather than restated. This half needs no disk,
        # so it runs everywhere — the corpus is gitignored, and a guard that skipped with it
        # would guard nothing on a fresh clone or CI (advisory R3-corpus-skip).
        stated_corpus = re.search(
            r"so all (\w+) now list \*\*(\d+)\*\*\s*\n?artifacts and declare\s*\n?"
            r"the same one absence", README)
        stated_stages = re.search(
            r"Why (\d+) stages and (\d+) artifacts when the registry declares (\d+) keys", README)
        assert stated_corpus and stated_stages, (
            "the corpus-count sentences the README makes about this machine changed shape; "
            "re-point this test at them instead of deleting the check")
        words = {"seven": 7, "six": 6, "eight": 8}
        n_total = words.get(stated_corpus.group(1).lower())
        n_count = int(stated_corpus.group(2))
        n_stages, n_stated_count, n_registry = (int(stated_stages.group(i)) for i in (1, 2, 3))
        assert n_total and n_count, (
            f"unparsed number in: {stated_corpus.group(0)!r}")

        # The prose has to add up on its own terms before disk or code is consulted. The
        # count appears twice (corpus sentence and stage sentence) and must agree with
        # itself; the registry total is the per-dataset count plus the one opt-in absence;
        # and the stage number is STAGE_ORDER's length, not a remembered number. The only
        # comparison against the registry constant is the one that says the declared 43 is
        # the registry's (advisory R3-rerun-upper-bound's lesson: derive bounds from the
        # prose, consult the constant once).
        from multimodal_pipeline.artifacts import MANIFEST_ARTIFACTS
        from multimodal_pipeline.stages.base import STAGE_ORDER

        assert n_count == n_stated_count, (
            f"the README states the per-dataset count twice and disagrees with itself: "
            f"{n_stated_count} vs {n_count}")
        assert n_count + 1 == n_registry, (
            f"the prose says {n_count} listed per dataset against {n_registry} registry "
            "keys, which only adds up if exactly one artifact is the declared opt-in absence")
        assert n_registry == len(MANIFEST_ARTIFACTS), (
            f"the prose says the registry declares {n_registry}; "
            f"MANIFEST_ARTIFACTS has {len(MANIFEST_ARTIFACTS)}")
        assert n_stages == len(STAGE_ORDER), (
            f"the prose says {n_stages} stages; STAGE_ORDER has {len(STAGE_ORDER)}")
        if len(manifests) < 2:
            pytest.skip("byte-level half needs the corpus under data/processed/")

        # 2. What the bytes say.
        docs = {p.parent.name: json.loads(p.read_text()) for p in manifests}
        assert len(docs) == n_total, (
            f"the prose says {n_total} datasets on this disk; {len(docs)} manifests found")
        # The prose claims uniformity: every dataset lists n_count artifacts and declares
        # exactly one absence, and that absence is the opt-in renderings.
        for name, doc in sorted(docs.items()):
            assert len(doc["artifacts"]) == n_count, (
                f"the prose says all {n_total} datasets list {n_count} artifacts; "
                f"{name} lists {len(doc['artifacts'])}")
            assert set(doc["artifacts_not_generated"]) == {"pose_images_raw"}, (
                f"the prose says every dataset declares the same one absence "
                f"(pose_images_raw); {name} declares "
                f"{sorted(doc['artifacts_not_generated'])}")
        # 3. The opt-in gate the prose names is the config default, not a wish.
        import inspect

        from multimodal_pipeline.config import OpenPoseConfig

        assert "write_images: bool = False" in inspect.getsource(OpenPoseConfig), (
            "the README explains pose_images_raw's absence as an opt-in defaulting to "
            "false; OpenPoseConfig no longer says that")
        # 4. A manifest that lists persons_raw must have the file behind it.
        for name, doc in sorted(docs.items()):
            if "persons_raw" in doc["artifacts"]:
                assert (root / name / "persons" / "raw" / "yolo_track.json").is_file(), (
                    f"{name}'s manifest lists persons_raw but the file is absent")

    def test_the_example_never_walks_an_absent_artifact_as_a_file_that_exists(self) -> None:
        """No row of the walked table may present a never-produced artifact as present.

        Narrower than "do not mention the name": the README is allowed (and required) to
        say these four are absent, which it does in prose. What it may not do is hand the
        reader a file-by-file table with a row that implies the file is in the dataset.
        """
        # Absence is measured from the corpus, not remembered from a run that aged out.
        # The old shape of this test hardcoded the seven artifacts of the newer stages; a
        # full batch run produced six of them everywhere and the hardcoded list walked the
        # guard past its own point. What no dataset lists IS the absent set.
        root = ROOT / "data" / "processed"
        manifests = sorted(root.glob("*/manifest.json"))
        if len(manifests) < 2:
            pytest.skip("byte-level absence needs the corpus under data/processed/")
        from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT, MANIFEST_ARTIFACTS

        docs = {p.parent.name: json.loads(p.read_text()) for p in manifests}
        absent = {
            key for key in MANIFEST_ARTIFACTS
            if all(key not in d["artifacts"] for d in docs.values())
        }
        assert absent == {"pose_images_raw"}, (
            f"the guard's promise was that exactly the opt-in renderings are absent "
            f"everywhere; disk disagrees: {sorted(absent)}")
        absent_paths = {ARTIFACT_LAYOUT[key] for key in absent}
        section = README[README.index("### One dataset, file by file"):]
        section = section[:section.index("### How to consume it")]
        walked = []
        for line in section.splitlines():
            stripped = line.strip()
            if stripped.startswith("| `"):
                first = stripped[3:].split("`")[0]
                walked.append(first)
        assert walked, "the worked example no longer walks any file in a table"
        promised = sorted(set(walked) & absent_paths)
        assert not promised, (
            f"the worked example walks {promised}, which no dataset under data/processed/ "
            "contains — the stage that writes it is newer than every run on this disk"
        )

    def test_the_worked_example_is_the_regenerable_dataset(self) -> None:
        # §20.6 asks for real values; the *reproducible* half only holds for the fixture
        # corpus, which is why that is the one walked file by file.
        section = README[README.index("### One dataset, file by file"):]
        assert "pipeline_demo" in section
        assert "make_fixtures.sh" in section

    def test_the_committed_fixture_durations_agree_with_the_prose(self) -> None:
        # 9.985 s is quoted in the quick start and in the worked example; make_fixtures.sh
        # asks for 14 s and -shortest trims it. Both sentences must keep agreeing.
        script = (ROOT / "scripts" / "make_fixtures.sh").read_text(encoding="utf-8")
        assert "-shortest" in script
        assert README.count("9.985") >= 2

    def test_the_verbatim_snippet_output_is_actually_verbatim(self) -> None:
        """"Output, verbatim" must survive the next column rename.

        The worked example prints `manifest.json` + `status.json` and claims the block
        below it is verbatim. Every other claim here is checked against the tree; this one
        can only be checked by running the snippet, and a stage renamed or a row-count key
        added makes the block wrong in a way no static assertion sees. It skips without the
        corpus like the other byte-level guards, and that is honest: the README scopes the
        claim to "this machine", so the machine with the dataset is the one that has to
        prove it — which it just did, by dying when a batch run moved the counts the block
        printed. A fresh clone cannot check a claim that was never made about it.
        """
        dataset = ROOT / "data" / "processed" / "pipeline_demo"
        if not (dataset / "manifest.json").is_file():
            pytest.skip("the verbatim claim is scoped to this machine's dataset, absent here")
        section = README[README.index("### One dataset, file by file"):]
        section = section[:section.index("### How to consume it")]
        snippet = re.search(r"^```python\n(.*?)^```", section, re.DOTALL | re.MULTILINE)
        claimed = re.search(r"^```text\n(.*?)^```", section, re.DOTALL | re.MULTILINE)
        assert snippet and claimed, "the worked example lost its snippet or its claimed output"
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", snippet.group(1)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        got = result.stdout.strip().splitlines()
        want = claimed.group(1).strip().splitlines()
        assert got == want, "\n".join(
            [f"README claims {len(want)} lines, the snippet prints {len(got)}:"]
            + [f"  claim: {line}" for line in want]
            + [f"  actual: {line}" for line in got])


def test_pytest_is_available_for_the_documented_test_command() -> None:
    """The README tells you to run `uv run --with pytest pytest tests/unit`."""
    assert (ROOT / "tests" / "unit").is_dir()
    assert (ROOT / "tests" / "e2e").is_dir()
    assert sys.version_info >= (3, 10)
