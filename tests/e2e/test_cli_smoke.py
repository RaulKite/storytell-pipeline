"""End-to-end smoke test through the real CLI.

Unit tests cover each stage in isolation; this file checks the only thing an
operator actually touches — the ``multimodal-pipeline`` command line against a
real on-disk project with ffmpeg, real uv workers for the light stages and the
mock translator. Heavy GPU stages are switched off in the config, exactly as a
CPU-only machine would run them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from tests.conftest import make_test_video

PROJECT_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def cli(*args: str, cwd: Path | None = None,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Invoke the console script as a subprocess — exit codes are part of the API."""
    argv = [sys.executable, "-m", "multimodal_pipeline.cli", *args]
    return subprocess.run(argv, cwd=str(cwd or PROJECT_ROOT), capture_output=True, text=True,
                          env=env if env is not None else _env(), timeout=1800)


def _env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def config_text(root: Path, video_dir: Path) -> str:
    """A CPU-only project: GPU stages disabled, mock translation, no diarization token."""
    return yaml.safe_dump({
        "project_root": str(root),
        "input": {"directory": str(video_dir), "recursive": False},
        "output": {"directory": str(root / "out")},
        "logging": {"level": "INFO", "console": False},
        "whisperx": {"enabled": False},
        "diarization": {"enabled": False},
        "translation": {"enabled": False},
        "spacy": {"enabled": False},
        "acoustic": {"enabled": False},
        "openpose": {"enabled": False},
    })


def parse_ffmpeg_major(version_line: str) -> int:
    """The major version from a ``ffmpeg -version`` first line.

    Extracted so the skip-versus-fail decision can be tested with a version this machine
    does not have. Asserting "ffmpeg must be 7" on a machine running 6 tests the machine
    rather than the pipeline, and a clean install then cannot use this suite to check
    itself.
    """
    match = re.search(r"version\s+n?(\d+)", version_line)
    if not match:
        raise AssertionError(f"unparseable ffmpeg version string: {version_line!r}")
    return int(match.group(1))


def openpose_or_skip(config_path: Path) -> str:
    """The discovered OpenPose executable for ``config_path``, or skip the calling test.

    Skipping is the right outcome when OpenPose is simply not installed here: an e2e test
    that fails because a third-party binary is absent tells an operator nothing about
    whether *their* install is correct, and this file is the closest the repository has to
    install verification. A discovered-but-unreadable path is a real defect and must still
    fail, so the caller asserts on the returned value.
    """
    payload = stdout_json(cli("inspect-environment", "-c", str(config_path)))
    discovered = payload["tools"]["openpose"].get("executable")
    if not discovered:
        pytest.skip(
            "no OpenPose binary discovered for "
            f"{config_path} (default root /opt/openpose present: {Path('/opt/openpose').is_dir()})"
        )
    return str(discovered)


class TestSkipInsteadOfFail:
    """The two guards that make this suite usable as install verification.

    Both exist because this file *failed* on a machine without OpenPose, and would have
    failed on ffmpeg 6, when neither is a defect in the pipeline. The helpers are called
    with a foreign version string and a foreign root so the skip paths are exercised on a
    machine that would otherwise never reach them.
    """

    @pytest.mark.parametrize(
        ("line", "major"),
        [
            ("ffmpeg version 7.1.1 Copyright (c) 2000-2025", 7),
            ("ffmpeg version 6.1.1-3ubuntu5 Copyright", 6),
            ("ffmpeg version n7.0.2", 7),
            ("ffmpeg version 8.0", 8),
        ],
    )
    def test_ffmpeg_major_parses_from_the_reported_string(self, line: str, major: int) -> None:
        assert parse_ffmpeg_major(line) == major

    def test_an_unparseable_version_fails_loudly(self) -> None:
        with pytest.raises(AssertionError, match="unparseable ffmpeg version"):
            parse_ffmpeg_major("ffmpeg: command not found")

    def test_a_machine_without_openpose_skips_instead_of_failing(self, tmp_path: Path) -> None:
        example = yaml.safe_load((PROJECT_ROOT / "config" / "config.example.yaml").read_text())
        example["openpose"]["root"] = str(tmp_path / "definitely-not-openpose")
        stray = tmp_path / "config"
        stray.mkdir(parents=True)
        broken = stray / "config.example.yaml"
        broken.write_text(yaml.safe_dump(example))

        with pytest.raises(pytest.skip.Exception) as raised:
            openpose_or_skip(broken)
        assert "no OpenPose binary discovered" in str(raised.value)


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("e2e")
    video_dir = root / "in"
    video_dir.mkdir()
    make_test_video(video_dir / "alpha.mp4", seconds=1.0, fps=25, size="96x64")
    make_test_video(video_dir / "beta.mp4", seconds=1.0, fps=30, size="96x64", rate="30000/1001")
    (root / "config").mkdir()
    (root / "config" / "config.yaml").write_text(config_text(root, video_dir), encoding="utf-8")
    return root


@pytest.fixture()
def fresh(project: Path) -> Path:
    """Each test starts from a pristine project: no output and no stray inputs.

    Tests that add a broken file must not leak it into the next test's batch —
    otherwise every later ``run`` exits nonzero for reasons the test never asked
    for, and the failure looks like a pipeline bug.
    """
    shutil.rmtree(project / "out", ignore_errors=True)
    wanted = {"alpha.mp4", "beta.mp4"}
    for stray in (project / "in").iterdir():
        if stray.name not in wanted:
            stray.unlink()
    return project


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stdout_json(result: subprocess.CompletedProcess) -> dict:
    """Machine-readable output must be parseable from stdout alone.

    If diagnostics share stdout, ``--json | jq`` breaks, so parsing stdout *is*
    the assertion.
    """
    return json.loads(result.stdout)


class TestHelpAndEnvironment:
    def test_help_lists_every_documented_command(self) -> None:
        result = cli("--help")
        assert result.returncode == 0
        for command in ("run", "status", "resume", "retry-failed", "process-video", "validate",
                        "inspect-environment"):
            assert command in result.stdout

    def test_unknown_stage_is_rejected_before_any_work(self, fresh: Path) -> None:
        result = cli("run", "-c", str(fresh / "config" / "config.yaml"), "--only-stage", "nonsense")
        assert result.returncode != 0
        assert "unknown stage" in (result.stdout + result.stderr).lower()
        assert not (fresh / "out").exists() or not list((fresh / "out").iterdir())

    def test_missing_config_fails_clearly(self, fresh: Path) -> None:
        result = cli("run", "-c", str(fresh / "config" / "absent.yaml"))
        assert result.returncode != 0
        assert "not found" in (result.stdout + result.stderr).lower() or \
            "no such file" in (result.stdout + result.stderr).lower()

    @staticmethod
    def _dotenv_env() -> dict[str, str]:
        """Environment with the implicit .env read switched back on for these tests."""
        env = _env()
        env.pop("MULTIMODAL_PIPELINE_NO_DOTENV", None)
        return env

    def test_a_malformed_dotenv_is_reported_not_traced(self, fresh: Path) -> None:
        """A half-edited credential file must name the line, not dump a traceback."""
        (fresh / ".env").write_text("HF_TOKEN=ok\nthis is not an assignment\n", encoding="utf-8")
        config = fresh / "config" / "config.yaml"
        result = cli("inspect-environment", "-c", str(config), cwd=fresh, env=self._dotenv_env())
        combined = result.stdout + result.stderr
        assert result.returncode == 2, combined[-1500:]
        assert ".env:2" in combined and "Traceback" not in combined

    def test_dotenv_credentials_reach_the_run_without_export(self, fresh: Path) -> None:
        """The documented workflow: write .env, run. No shell wrapper, no `source`."""
        env = self._dotenv_env()
        env.pop("HF_TOKEN", None)
        (fresh / ".env").write_text("HF_TOKEN=hf_fromdotenvfile\n", encoding="utf-8")
        config = fresh / "config" / "config.yaml"
        payload = yaml.safe_load(config.read_text())
        payload["diarization"] = {"enabled": True, "hf_token_env": "HF_TOKEN"}
        config.write_text(yaml.safe_dump(payload), encoding="utf-8")
        result = cli("inspect-environment", "-c", str(config), cwd=fresh, env=env)
        warnings = stdout_json(result)["environment_warnings"]
        # The token now exists, so the "will be skipped" warning must be gone.
        assert not [w for w in warnings if "HF_TOKEN" in w], warnings
        # And its value must not be echoed anywhere in the report.
        assert "hf_fromdotenvfile" not in (result.stdout + result.stderr)

    def test_inspect_environment_reports_the_toolchain(self) -> None:
        """Runs on any machine with ffmpeg: what is *found* is reported, honestly.

        Version- and install-specific claims live in their own tests below, so a machine
        that differs from the one this pipeline was verified on cannot fail its way past
        its own install check. That distinction is the point: this file is the closest
        thing the repository has to install verification, and a test that fails because
        OpenPose is not installed tells an operator nothing about whether they installed
        the pipeline correctly.
        """
        example = PROJECT_ROOT / "config" / "config.example.yaml"
        result = cli("inspect-environment", "-c", str(example))
        assert result.returncode == 0, result.stderr[-2000:]
        payload = stdout_json(result)
        assert payload["system"]["python"]
        assert payload["tools"]["ffmpeg"]["ffmpeg"], "ffmpeg version was not probed"
        assert payload["tools"]["ffmpeg"]["ffprobe"], "ffprobe version was not probed"
        assert payload["tools"]["ffmpeg"]["resolved_ffmpeg"], "ffmpeg is not resolvable"
        assert "openpose" in payload["tools"]
        assert "uv_projects" in payload["tools"]

    def test_the_ffmpeg_major_this_pipeline_was_verified_against(self) -> None:
        """Parselmouth-free behaviour was characterised against ffmpeg 7.1.1.

        Skipped rather than failed on another major: the claim under test is "the version
        we verified against is what gets reported", which a machine on a different major
        cannot answer. The README names 7.x as the verified pin.
        """
        result = cli("inspect-environment", "-c", str(PROJECT_ROOT / "config" / "config.example.yaml"))
        payload = stdout_json(result)
        reported = payload["tools"]["ffmpeg"]["ffmpeg"]
        major = parse_ffmpeg_major(reported)
        if major != 7:
            pytest.skip(f"ffmpeg major {major} here; the verified pin is 7.x")
        assert "7." in reported and "7." in payload["tools"]["ffmpeg"]["ffprobe"]

    def test_openpose_is_discovered_where_it_is_installed(self) -> None:
        """Skipped when this machine has no OpenPose; failing when it has a broken install."""
        discovered = openpose_or_skip(PROJECT_ROOT / "config" / "config.example.yaml")
        assert Path(discovered).is_file(), f"discovered OpenPose executable is not a file: {discovered}"

    def test_the_example_config_matches_the_schema(self) -> None:
        """A shipped example that does not load is worse than no example."""
        result = cli("inspect-environment", "-c", str(PROJECT_ROOT / "config" / "config.example.yaml"))
        assert result.returncode == 0, result.stderr[-1500:]
        assert "invalid configuration" not in result.stderr


class TestFullRun:
    def test_run_produces_a_valid_dataset_per_video(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        result = cli("run", "-c", config)
        assert result.returncode == 0, result.stderr[-3000:]
        for video_id in ("alpha", "beta"):
            dataset = fresh / "out" / video_id
            manifest = load(dataset / "manifest.json")
            assert manifest["processing"]["status"] == "completed", manifest["processing"]
            assert manifest["source"]["duration_seconds"] == pytest.approx(1.0, abs=0.15)
            assert (dataset / "source" / "metadata.json").is_file()
            assert (dataset / "audio" / "audio.wav").is_file()
            assert (dataset / "source" / "frame_index.parquet").is_file()
            validate = cli("validate", "-c", config)
            assert validate.returncode == 0, validate.stdout[-2000:]

    def test_disabled_stages_are_skipped_not_failed(self, fresh: Path) -> None:
        cli("run", "-c", str(fresh / "config" / "config.yaml"))
        stages = load(fresh / "out" / "alpha" / "manifest.json")["processing"]["stages"]
        assert stages["whisperx"] == "skipped"
        assert stages["openpose"] == "skipped"
        assert stages["metadata"] == "completed"

    # --- the second diarization engine must be inert until it is asked for -----------

    def test_second_diarizer_skips_and_is_declared_absent(self, fresh: Path) -> None:
        """With the engine off, the run completes and the extra table is *declared* absent.

        This is the guarantee the operator's request depends on: adding the second engine
        may not break a corpus that does not want it, and its artifacts may not be silently
        missing (the manifest is the contract that says why a file is not there).
        """
        config = str(fresh / "config" / "config.yaml")
        result = cli("run", "-c", config)
        assert result.returncode == 0, result.stderr[-3000:]
        manifest = load(fresh / "out" / "alpha" / "manifest.json")
        assert manifest["processing"]["stages"]["diarization_nemotron"] == "skipped"
        assert "speaker_turns_nemotron" in manifest["artifacts_not_generated"]
        assert not (fresh / "out" / "alpha" / "speech" / "speaker_turns_nemotron.parquet").exists()
        validate = cli("validate", "-c", config)
        assert validate.returncode == 0, validate.stdout[-2000:]

    def test_enabled_second_diarizer_without_environment_skips_with_the_fix(self, fresh: Path) -> None:
        """Enabled but not installed: skip, name the fix, and still complete the dataset."""
        config_path = fresh / "config" / "config.yaml"
        payload = yaml.safe_load(config_path.read_text())
        payload["diarization_nemotron"] = {
            "enabled": True,
            "uv_project": str(fresh / "environments" / "diarization_nemotron"),
        }
        config_path.write_text(yaml.safe_dump(payload))
        result = cli("run", "-c", str(config_path))
        assert result.returncode == 0, result.stderr[-3000:]
        stages = load(fresh / "out" / "alpha" / "manifest.json")["processing"]["stages"]
        assert stages["diarization_nemotron"] == "skipped"
        status = load(fresh / "out" / "alpha" / "status.json")
        reason = json.dumps(status)
        assert "uv sync" in reason, "the skip reason must carry the remedy"

    # --- the real thing: run the real worker over a real clip -------------------------

    def nemotron_or_skip(self, fresh: Path) -> None:
        """Skip unless this machine can actually run the second engine.

        Two hard prerequisites, both checked rather than assumed:

        * a working CUDA in the stage's own environment. A CPU run of this test would still
          be a real run, but it turns a 2-second stage into minutes on a shared CI runner,
          so the GPU is what makes it fair to leave in the suite;
        * the checkpoint already in the local Hugging Face cache. Without it the worker
          starts a ~500 MB download, which is not something a test may do silently and is
          impossible where there is no network.

        Skipping is the honest outcome when either is missing. Failing would punish a clean
        install for not having a GPU, exactly the defect the OpenPose and ffmpeg guards in
        this file were added to remove.
        """
        project = PROJECT_ROOT / "environments" / "diarization_nemotron"
        if not project.is_dir():
            pytest.skip(f"no nemotron environment at {project}")
        probe = subprocess.run(
            ["uv", "run", "--project", str(project), "python", "-c",
             "import torch; print(int(torch.cuda.is_available()))"],
            capture_output=True, text=True, cwd=PROJECT_ROOT,
        )
        if probe.stdout.strip() != "1":
            pytest.skip("nemotron environment reports no usable CUDA on this machine")
        cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--nvidia--Nemotron-3-Diarization"
        if not cache.is_dir():
            pytest.skip(
                f"nvidia/Nemotron-3-Diarization is not in {cache}; run the stage once "
                "manually to fetch it (~500 MB) instead of letting a test download it"
            )

    def test_the_real_worker_produces_a_comparable_table(self, fresh: Path) -> None:
        """End to end with the committed worker and the committed environment.

        This is the only test in the suite that proves the pin in
        ``environments/diarization_nemotron/pyproject.toml`` still loads this checkpoint.
        Everything else in this file would stay green if the pin were wrong, because the
        stage skips politely when its environment is absent.
        """
        self.nemotron_or_skip(fresh)
        config_path = fresh / "config" / "config.yaml"
        payload = yaml.safe_load(config_path.read_text())
        payload["whisperx"] = {"enabled": False}
        payload["diarization"] = {"enabled": False}
        payload["translation"] = {"enabled": False}
        payload["diarization_nemotron"] = {
            "enabled": True,
            # Absolute on purpose: this project's project_root is a temporary directory, so
            # the repository-relative defaults would resolve to a worker and an environment
            # that do not exist there — and the stage would skip, testing nothing.
            "uv_project": str(PROJECT_ROOT / "environments" / "diarization_nemotron"),
            "worker": str(PROJECT_ROOT / "workers" / "nemotron_diarization_worker.py"),
            "device": "cuda",
        }
        payload["spacy"] = {"enabled": False}
        payload["acoustic"] = {"enabled": False}
        config_path.write_text(yaml.safe_dump(payload))

        # A full run, not --only-stage: the manifest is written by finalization, and the
        # manifest is the artifact contract this test is checking.
        result = cli("run", "-c", str(config_path))
        assert result.returncode == 0, result.stderr[-4000:]

        dataset = fresh / "out" / "alpha"
        stages = load(dataset / "manifest.json")["processing"]["stages"]
        assert stages["diarization_nemotron"] == "completed", stages

        table = pq.read_table(dataset / "speech" / "speaker_turns_nemotron.parquet")
        assert [f.name for f in table.schema] == [
            "schema_version", "video_id", "turn_id", "speaker_id", "start_time",
            "end_time", "duration", "diarization_type", "overlap_s"], table.schema.names
        rows = table.to_pylist()
        # alpha is a 1-second synthetic tone: speech is whatever the model decides, so the
        # assertions are invariants rather than content. Content is what a human compares.
        for row in rows:
            assert row["diarization_type"] == "overlapping"
            assert row["end_time"] > row["start_time"]
            assert row["overlap_s"] >= 0.0
            assert row["speaker_id"].startswith("speaker_")
            assert row["start_time"] >= 0.0
            assert row["end_time"] <= 1.35, "a turn ran past the end of the clip"

        # The raw output is preserved and is the ground truth the table was derived from.
        raw = load(dataset / "speech" / "raw" / "nemotron_diarization.json")
        assert raw["model_id"] == "nvidia/Nemotron-3-Diarization"
        assert raw["device"] == "cuda"
        assert raw["video_id"] == "alpha"
        assert raw["logits_shape"][2] == 8, "expected the model's 8 speaker channels"
        assert len(raw["segments"]) >= len(rows)
        # The offline geometry the README documents is the geometry actually used.
        assert raw["offline_geometry"]["chunk_length"] == 340
        assert raw["offline_geometry"]["frame_stride_ms"] == pytest.approx(10.0)

        validate = cli("validate", "-c", str(config_path))
        assert validate.returncode == 0, validate.stdout[-3000:]

    def test_rerunning_the_second_engine_is_a_no_op(self, fresh: Path) -> None:
        """A completed second engine must not re-run, the same way every other stage works.

        It costs a model load per video, so a fingerprint that failed to stabilise would be
        paid for on every batch, on every video, forever.
        """
        self.nemotron_or_skip(fresh)
        config_path = fresh / "config" / "config.yaml"
        payload = yaml.safe_load(config_path.read_text())
        payload["diarization_nemotron"] = {
            "enabled": True,
            "uv_project": str(PROJECT_ROOT / "environments" / "diarization_nemotron"),
            "worker": str(PROJECT_ROOT / "workers" / "nemotron_diarization_worker.py"),
            "device": "cuda",
        }
        config_path.write_text(yaml.safe_dump(payload))
        first = cli("run", "-c", str(config_path))
        assert first.returncode == 0, first.stderr[-3000:]
        status = load(fresh / "out" / "alpha" / "status.json")
        record = status["stages"]["diarization_nemotron"]
        assert record["status"] == "completed"
        assert record["config_hash"], "a completed stage must record its fingerprint"

        # The plan is written to stderr in this CLI (see test_status_plan_explains_reuse).
        plan = cli("status", "-c", str(config_path), "--plan")
        assert plan.returncode == 0, plan.stderr[-2000:]
        assert "diarization_nemotron" in plan.stderr, plan.stderr[-2000:]
        # The plan must say reuse, not recompute: that is the whole point of the fingerprint.
        plan_line = [line for line in plan.stderr.splitlines() if "diarization_nemotron" in line]
        assert any("valid previous result" in line for line in plan_line), plan_line
        second = cli("run", "-c", str(config_path))
        assert second.returncode == 0, second.stderr[-3000:]
        after = load(fresh / "out" / "alpha" / "status.json")
        assert after["stages"]["diarization_nemotron"]["config_hash"] == record["config_hash"], (
            "the fingerprint moved between two identical runs")
        stages = load(fresh / "out" / "alpha" / "manifest.json")["processing"]["stages"]
        assert stages["diarization_nemotron"] in ("completed", "reused"), stages

    def test_status_reads_the_dataset_without_recomputing(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config)
        result = cli("status", "-c", config)
        assert result.returncode == 0, result.stderr[-2000:]
        # Human tables are diagnostics: they belong on stderr, out of pipeable output.
        assert "alpha" in result.stderr and "beta" in result.stderr
        assert result.stdout.strip() == ""

    def test_status_json_is_machine_readable(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config)
        payload = stdout_json(cli("status", "-c", config, "--json"))
        assert {video["video_id"] for video in payload["videos"]} == {"alpha", "beta"}
        assert payload["stages"][0] == "metadata"

    def test_status_plan_explains_reuse(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config)
        result = cli("status", "-c", config, "--plan")
        assert result.returncode == 0, result.stderr[-2000:]
        assert "metadata" in result.stderr
        assert "valid previous result" in result.stderr

    def test_batch_report_counts_the_run(self, fresh: Path) -> None:
        cli("run", "-c", str(fresh / "config" / "config.yaml"))
        report = load(fresh / "out" / "batch_report.json")
        assert report["summary"]["completed"] == 2
        assert report["summary"]["failed"] == 0
        assert [video["video_id"] for video in report["videos"]] == ["alpha", "beta"]

    def test_second_run_reuses_everything(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config)
        fingerprint = {path.name: (path.stat().st_mtime_ns, path.read_bytes()[:64])
                       for path in (fresh / "out" / "alpha").rglob("*.parquet")}
        result = cli("run", "-c", config)
        assert result.returncode == 0, result.stderr[-3000:]
        after = {path.name: (path.stat().st_mtime_ns, path.read_bytes()[:64])
                 for path in (fresh / "out" / "alpha").rglob("*.parquet")}
        assert after == fingerprint, "an unchanged second run must not rewrite artifacts"

    def test_a_batch_survives_one_broken_video(self, fresh: Path) -> None:
        """A corrupt file must not stop the batch or hide the good results."""
        (fresh / "in" / "broken.mp4").write_bytes(b"not a video at all")
        config = str(fresh / "config" / "config.yaml")
        result = cli("run", "-c", config)
        report = load(fresh / "out" / "batch_report.json")
        by_id = {video["video_id"]: video for video in report["videos"]}
        assert by_id["alpha"]["status"] == "completed"
        # metadata is the first stage: with nothing produced the video is a failure,
        # not a partial result someone could use.
        assert by_id["broken"]["status"] == "failed"
        assert by_id["broken"]["stages"]["metadata"] == "failed"
        assert report["summary"]["failed"] == 1
        assert result.returncode != 0, "a failed video must be visible to CI"
        assert (fresh / "out" / "broken" / "status.json").is_file()

    def test_failure_reason_is_recorded_not_silently_swallowed(self, fresh: Path) -> None:
        (fresh / "in" / "broken.mp4").write_bytes(b"not a video at all")
        cli("run", "-c", str(fresh / "config" / "config.yaml"))
        entry = next(video for video in load(fresh / "out" / "batch_report.json")["videos"]
                     if video["video_id"] == "broken")
        assert entry["errors"], entry
        assert set(entry["errors"]) & {"metadata", "audio"}


class TestStageSelection:
    def test_only_stage_runs_prerequisites(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        result = cli("run", "-c", config, "--video", "beta.mp4", "--only-stage", "audio")
        assert result.returncode == 0, result.stderr[-3000:]
        stages = load(fresh / "out" / "beta" / "status.json")["stages"]
        assert stages["metadata"]["status"] == "completed"
        assert stages["audio"]["status"] == "completed"

    def test_force_stage_recomputes(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config, "--video", "alpha.mp4")
        before = (fresh / "out" / "alpha" / "audio" / "audio.wav").stat().st_mtime_ns
        result = cli("run", "-c", config, "--video", "alpha.mp4", "--only-stage", "audio",
                     "--force-stage", "audio")
        assert result.returncode == 0, result.stderr[-3000:]
        assert (fresh / "out" / "alpha" / "audio" / "audio.wav").stat().st_mtime_ns >= before

    def test_video_by_bare_name(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        assert cli("run", "-c", config, "--video", "alpha.mp4").returncode == 0

    def test_unknown_video_is_reported(self, fresh: Path) -> None:
        result = cli("run", "-c", str(fresh / "config" / "config.yaml"), "--video", "ghost.mp4")
        assert result.returncode != 0
        assert "ghost.mp4" in (result.stdout + result.stderr)


class TestResumeAndRetry:
    def test_resume_finishes_an_interrupted_video(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        # Simulate an interruption: run only through audio, then resume.
        assert cli("run", "-c", config, "--video", "alpha.mp4", "--to-stage", "audio").returncode == 0
        assert cli("resume", "-c", config, "--video", "alpha.mp4").returncode == 0
        final = load(fresh / "out" / "alpha" / "status.json")
        assert final["stages"]["finalization"]["status"] == "completed"
        assert final["stages"]["metadata"]["status"] == "completed"

    def test_resume_reports_nothing_to_do(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config)
        assert cli("resume", "-c", config).returncode == 0

    def test_retry_failed_only_touches_the_failure(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        (fresh / "in" / "broken.mp4").write_bytes(b"still not a video")
        cli("run", "-c", config)
        audio_before = (fresh / "out" / "alpha" / "audio" / "audio.wav").stat().st_mtime_ns
        result = cli("retry-failed", "-c", config)
        assert result.returncode != 0  # the video is still broken
        assert (fresh / "out" / "alpha" / "audio" / "audio.wav").stat().st_mtime_ns == audio_before

    def test_interrupted_stage_is_never_reported_as_done(self, fresh: Path) -> None:
        """Kill mid-run; the stage in flight must come back as failed/pending, not completed."""
        config_path = fresh / "config" / "config.yaml"
        payload = yaml.safe_load(config_path.read_text())
        payload["openpose"]["enabled"] = True  # the slowest stage: a reliable kill window
        payload["openpose"]["root"] = "/opt/openpose"
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-m", "multimodal_pipeline.cli", "run",
                                    "-c", str(config_path), "--video", "alpha.mp4"],
                                   cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=_env())
        status_path = fresh / "out" / "alpha" / "status.json"
        deadline = __import__("time").time() + 240
        killed = False
        while __import__("time").time() < deadline:
            if status_path.is_file():
                try:
                    stages = load(status_path)["stages"]
                except (json.JSONDecodeError, OSError):
                    stages = {}
                if any(record.get("status") == "running" for record in stages.values()):
                    process.kill()
                    killed = True
                    break
            if process.poll() is not None:
                break
            __import__("time").sleep(0.5)
        process.wait(timeout=60)
        if not killed:
            pytest.skip("could not catch a running stage in time")
        stages = load(status_path)["stages"]
        assert not any(record.get("status") == "completed"
                       for record in stages.values() if record.get("status") == "running")
        assert any(record.get("status") in {"failed", "pending", "interrupted"}
                   for record in stages.values()), stages
        # And the pipeline must be resumable after the crash.
        assert cli("resume", "-c", str(config_path), "--video", "alpha.mp4").returncode == 0


class TestProcessVideoAndValidation:
    def test_process_video_handles_one_file(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        result = cli("process-video", "-c", config, str(fresh / "in" / "alpha.mp4"))
        assert result.returncode == 0, result.stderr[-3000:]
        assert (fresh / "out" / "alpha" / "manifest.json").is_file()

    def test_validate_detects_a_deleted_artifact(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config)
        (fresh / "out" / "alpha" / "audio" / "audio.wav").unlink()
        result = cli("validate", "-c", config)
        assert result.returncode != 0
        assert "audio" in (result.stdout + result.stderr)

    def test_validate_json_reports_per_video(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config)
        payload = stdout_json(cli("validate", "-c", config, "--json"))
        assert payload["ok"] is True
        assert {entry["video_id"] for entry in payload["results"]} == {"alpha", "beta"}

    def test_a_rewritten_frame_index_is_detected(self, fresh: Path) -> None:
        config = str(fresh / "config" / "config.yaml")
        cli("run", "-c", config)
        index = fresh / "out" / "alpha" / "source" / "frame_index.parquet"
        table = pq.read_table(index)
        pq.write_table(table.slice(0, max(table.num_rows - 1, 1)), index)
        result = cli("validate", "-c", config)
        assert result.returncode != 0


class TestSecrets:
    def test_no_credential_appears_in_any_written_file(self, fresh: Path) -> None:
        config_path = fresh / "config" / "config.yaml"
        payload = yaml.safe_load(config_path.read_text())
        payload["translation"] = {"enabled": False, "provider": "openai-compatible",
                                 "base_url": "https://litellm.invalid/v1",
                                 "api_key": "sk-E2E-secret-do-not-write", "model": "m"}
        payload["diarization"] = {"enabled": True, "hf_token_env": "HF_TOKEN"}
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        env = _env()
        env["HF_TOKEN"] = "hf_E2Esecrettoken"
        subprocess.run([sys.executable, "-m", "multimodal_pipeline.cli", "run", "-c",
                        str(config_path)], cwd=str(PROJECT_ROOT), capture_output=True, text=True,
                       env=env, timeout=1800)
        offenders = []
        for path in sorted((fresh / "out").rglob("*")):
            if not path.is_file() or path.suffix in {".wav", ".parquet"}:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for secret in ("sk-E2E-secret-do-not-write", "hf_E2Esecrettoken"):
                if secret in text:
                    offenders.append(f"{path.relative_to(fresh)}: {secret}")
        assert offenders == []

    def test_the_token_variable_name_survives_but_its_value_does_not(self, fresh: Path) -> None:
        config_path = fresh / "config" / "config.yaml"
        payload = yaml.safe_load(config_path.read_text())
        payload["diarization"] = {"enabled": True, "hf_token_env": "HF_TOKEN"}
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        env = _env()
        env["HF_TOKEN"] = "hf_E2Esecrettoken"
        subprocess.run([sys.executable, "-m", "multimodal_pipeline.cli", "run", "-c",
                        str(config_path), "--video", "alpha.mp4"], cwd=str(PROJECT_ROOT),
                       capture_output=True, text=True, env=env, timeout=1800)
        written = "\n".join(path.read_text(encoding="utf-8", errors="replace")
                            for path in (fresh / "out" / "alpha").rglob("*") if path.is_file())
        assert "hf_E2Esecrettoken" not in written
        # The operator still has to learn *which* variable to set; naming it is
        # documentation, echoing its value would be a leak.
        warnings = cli("inspect-environment", "-c", str(config_path))
        assert "HF_TOKEN" in stdout_json(warnings)["environment_warnings"][0]
        assert "hf_E2Esecrettoken" not in stdout_json(warnings)["environment_warnings"][0]
