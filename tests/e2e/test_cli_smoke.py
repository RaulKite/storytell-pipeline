"""End-to-end smoke test through the real CLI.

Unit tests cover each stage in isolation; this file checks the only thing an
operator actually touches — the ``multimodal-pipeline`` command line against a
real on-disk project with ffmpeg, real uv workers for the light stages and the
mock translator. Heavy GPU stages are switched off in the config, exactly as a
CPU-only machine would run them.
"""

from __future__ import annotations

import json
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


def cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Invoke the console script as a subprocess — exit codes are part of the API."""
    argv = [sys.executable, "-m", "multimodal_pipeline.cli", *args]
    return subprocess.run(argv, cwd=str(cwd or PROJECT_ROOT), capture_output=True, text=True,
                          env=_env(), timeout=1800)


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

    def test_inspect_environment_reports_the_toolchain(self) -> None:
        example = PROJECT_ROOT / "config" / "config.example.yaml"
        result = cli("inspect-environment", "-c", str(example))
        assert result.returncode == 0, result.stderr[-2000:]
        payload = stdout_json(result)
        assert payload["system"]["python"]
        assert "7." in payload["tools"]["ffmpeg"]["ffmpeg"]
        assert "7." in payload["tools"]["ffmpeg"]["ffprobe"]
        assert payload["tools"]["openpose"]["executable"], "OpenPose must be discovered"

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
