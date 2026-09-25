"""Unit tests for the Nemotron 3 Diarization stage — the second engine.

The stage exists so two diarizers can be compared, so the guarantees worth testing are the
ones that keep a comparison honest: the pyannote artifacts must be untouched, the two
speaker-id namespaces must not merge, the overlap that motivates the engine must survive
normalisation, and an unavailable optional engine must skip rather than fail a corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from multimodal_pipeline.exceptions import ValidationError
from multimodal_pipeline.normalization import nemotron_turn_rows
from multimodal_pipeline.schemas import SPEAKER_TURNS_NEMOTRON_SCHEMA, read_table
from multimodal_pipeline.stages.base import stamp_raw
from multimodal_pipeline.stages.diarization_nemotron import DiarizationNemotronStage

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_nemotron_document(**overrides: Any) -> dict[str, Any]:
    """A structurally real worker document, measured shape from the 2026-09-25 probe.

    Segment times and the two-channel overlap come from the La1 clip: pyannote reported one
    speaker there and Nemotron reported two with real overlap, so this fixture reproduces
    the disagreement that is the reason the stage exists.
    """
    document = {
        "schema_version": "1.0",
        "video_id": "conversation_001",
        "model_id": "nvidia/Nemotron-3-Diarization",
        "runtime": "transformers",
        "runtime_version": "5.18.0.dev0",
        "torch_version": "2.8.0+cu128",
        "device": "cuda",
        "requested_device": "cuda",
        "max_speakers": 8,
        "threshold": 0.5,
        "sample_rate": 16000,
        "duration_seconds": 8.011,
        "logits_shape": [1, 802, 8],
        "offline_geometry": {
            "chunk_length": 340, "chunk_right_context": 40, "fifo_length": 40,
            "speaker_cache_update_period": 300, "speaker_cache_length": 264,
            "prediction_score_threshold": 0.25, "frame_stride_ms": 10.0, "channels": 8,
        },
        "segments": [
            {"start": 0.0, "end": 0.91, "speaker_index": 0, "speaker_id": "speaker_0"},
            {"start": 3.26, "end": 5.68, "speaker_index": 1, "speaker_id": "speaker_1"},
            {"start": 5.35, "end": 8.01, "speaker_index": 0, "speaker_id": "speaker_0"},
            {"start": 6.9, "end": 7.75, "speaker_index": 1, "speaker_id": "speaker_1"},
        ],
        "speakers": ["speaker_0", "speaker_1"],
        "dropped_over_max_speakers": {},
        "load_seconds": 1.1,
        "inference_seconds": 0.15,
    }
    document.update(overrides)
    return document


@pytest.fixture
def seeded(context):
    """A context with audio, metadata, and a valid raw document plus its sidecar."""
    stage = DiarizationNemotronStage()
    audio = context.artifact("audio")
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"stub-16k-mono-wav")
    raw = context.artifact("nemotron_diarization_raw")
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(make_nemotron_document()), encoding="utf-8")
    stamp_raw(raw, request=stage.request(context), digest=stage.request_digest(context),
              worker=None)
    context.artifact("metadata").parent.mkdir(parents=True, exist_ok=True)
    context.artifact("metadata").write_text(json.dumps({"duration_seconds": 8.011}),
                                            encoding="utf-8")
    return context


def restamp(context, stage) -> None:
    """Re-stamp after the caller mutated config or the raw document."""
    raw = context.artifact("nemotron_diarization_raw")
    stamp_raw(raw, request=stage.request(context), digest=stage.request_digest(context),
              worker=None)


def enable_environment(context, tmp_path: Path) -> Path:
    """Point the stage at a uv project directory and the real worker script.

    The worker is the repository's own file rather than a stub: the gating test then
    checks the same paths a real run would use, and a renamed worker is caught instead of
    being quietly accepted by a fake.
    """
    project = tmp_path / "environments" / "diarization_nemotron"
    project.mkdir(parents=True)
    context.config.diarization_nemotron.enabled = True
    context.config.diarization_nemotron.uv_project = project
    context.config.diarization_nemotron.worker = PROJECT_ROOT / "workers" / "nemotron_diarization_worker.py"
    return project


class TestGating:
    """An optional second engine must skip, and say exactly why."""

    def test_disabled_by_default(self, context):
        enabled, reason = DiarizationNemotronStage().enabled(context)
        assert enabled is False
        assert reason == "diarization_nemotron.enabled = false"

    def test_enabled_but_environment_missing_skips_with_the_fix(self, context, tmp_path):
        cfg = context.config.diarization_nemotron
        cfg.enabled = True
        cfg.uv_project = tmp_path / "environments" / "diarization_nemotron"
        enabled, reason = DiarizationNemotronStage().enabled(context)
        assert enabled is False
        # The reason must carry the remedy: this is the number-one setup failure class in
        # this repository (a stage that dies with "uv project not found" and no next step).
        assert "uv sync" in reason
        assert str(cfg.uv_project) in reason

    def test_enabled_with_environment_present(self, context, tmp_path):
        enable_environment(context, tmp_path)
        enabled, reason = DiarizationNemotronStage().enabled(context)
        assert enabled is True
        assert reason == ""

    def test_missing_token_does_not_disable_it(self, context, tmp_path, monkeypatch):
        """Unlike pyannote community-1, this model is not gated: no token, still runs."""
        enable_environment(context, tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        enabled, _ = DiarizationNemotronStage().enabled(context)
        assert enabled is True

    def test_missing_worker_script_skips(self, context, tmp_path):
        cfg = context.config.diarization_nemotron
        cfg.enabled = True
        cfg.uv_project = tmp_path
        cfg.worker = tmp_path / "absent_worker.py"
        enabled, reason = DiarizationNemotronStage().enabled(context)
        assert enabled is False
        assert "worker script missing" in reason


class TestNoPyannoteCoupling:
    """The comparison is only valid if the first engine is genuinely untouched."""

    def test_declares_no_pyannote_artifact_as_output(self):
        stage = DiarizationNemotronStage()
        assert "speaker_turns" not in stage.outputs
        assert "diarization_raw" not in stage.outputs
        assert set(stage.outputs) == {"nemotron_diarization_raw", "speaker_turns_nemotron"}

    def test_normalize_does_not_touch_the_pyannote_table(self, seeded):
        stage = DiarizationNemotronStage()
        pyannote_table = seeded.artifact("speaker_turns")
        pyannote_table.parent.mkdir(parents=True, exist_ok=True)
        original = b"pyannote-owned-bytes-must-survive"
        pyannote_table.write_bytes(original)
        stage.normalize(seeded)
        assert pyannote_table.read_bytes() == original

    def test_dependencies_exclude_the_other_diarizer(self):
        from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES

        assert STAGE_DEPENDENCIES["diarization_nemotron"] == ("audio",)
        # And the reverse: speaker assignment must not start consuming the new engine
        # silently, because speaker labels across the whole dataset would change meaning.
        assert "diarization_nemotron" not in STAGE_DEPENDENCIES["speaker_assignment"]


class TestEnvironmentIsolation:
    def test_environment_path_is_configurable(self, context):
        from pathlib import Path as P

        assert context.config.diarization_nemotron.uv_project == P("environments/diarization_nemotron")

    def test_its_own_environment_is_a_real_uv_project(self):
        """The pin file exists and is parseable TOML with the measured constraints."""
        import tomllib

        path = PROJECT_ROOT / "environments" / "diarization_nemotron" / "pyproject.toml"
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        deps = " ".join(data["project"]["dependencies"])
        assert "transformers" in deps
        assert "librosa" in deps
        assert "accelerate" in deps
        assert "torch==2.8.0" in deps
        # A git source must be an exact commit, never a branch name.
        assert "transformers @ git+" in deps
        assert "@main" not in deps
        assert "@5880561ab3ea92cb2d8943ffd58891d0bc085fe1" in deps


class TestFingerprint:
    def test_audio_digest_participates(self, seeded):
        stage = DiarizationNemotronStage()
        before = stage.request_digest(seeded)
        seeded.artifact("audio").write_bytes(b"different-audio")
        # The stage memoises the digest in ctx.scratch for the run, exactly as the pyannote
        # stage does; clearing it is what a fresh process would bring.
        seeded.scratch.pop("audio_sha256_nemotron", None)
        assert stage.request_digest(seeded) != before

    def test_model_change_invalidates(self, seeded):
        stage = DiarizationNemotronStage()
        before = stage.request_digest(seeded)
        seeded.config.diarization_nemotron.model = "nvidia/Some-Other-Model"
        assert stage.request_digest(seeded) != before

    def test_threshold_change_invalidates(self, seeded):
        """threshold changes which frames become segments, so it must rerun the model."""
        stage = DiarizationNemotronStage()
        before = stage.request_digest(seeded)
        seeded.config.diarization_nemotron.threshold = 0.35
        assert stage.request_digest(seeded) != before

    def test_max_speakers_change_invalidates(self, seeded):
        stage = DiarizationNemotronStage()
        before = stage.request_digest(seeded)
        seeded.config.diarization_nemotron.max_speakers = 4
        assert stage.request_digest(seeded) != before

    def test_device_change_invalidates(self, seeded):
        stage = DiarizationNemotronStage()
        before = stage.request_digest(seeded)
        seeded.config.diarization_nemotron.device = "cpu"
        assert stage.request_digest(seeded) != before

    def test_identical_config_reproduces_the_digest(self, seeded):
        stage = DiarizationNemotronStage()
        assert stage.request_digest(seeded) == DiarizationNemotronStage().request_digest(seeded)

    def test_worker_source_change_invalidates(self, seeded):
        """Cached raw output must not survive a worker bug fix."""
        stage = DiarizationNemotronStage()
        payload = stage.digest_payload(seeded)
        assert "_worker_code_sha256" in payload

    def test_environment_absence_participates(self, seeded, tmp_path):
        """Same path, different installed content must not claim each other's cache."""
        stage = DiarizationNemotronStage()
        assert stage.request(seeded)["uv_project_present"] is False
        seeded.config.diarization_nemotron.uv_project = tmp_path
        assert stage.request(seeded)["uv_project_present"] is True


class TestOverlapNormalisation:
    """The overlap is the product. If it is flattened, the engine bought nothing."""

    def test_overlapping_pair_is_measured_per_row(self):
        rows = nemotron_turn_rows(make_nemotron_document(), "v1")
        by_span = {(row["start_time"], row["end_time"]): row for row in rows}
        # 6.90-7.75 sits entirely inside speaker_0's 5.35-8.01.
        assert by_span[(6.9, 7.75)]["overlap_s"] == pytest.approx(0.85, abs=1e-6)
        # The enclosing segment overlaps TWO foreign segments, and overlap_s is the sum:
        # 0.33 s against 3.26-5.68 plus 0.85 s against 6.90-7.75. A reviewer checking the
        # column must expect a sum, not the single largest overlap.
        assert by_span[(5.35, 8.01)]["overlap_s"] == pytest.approx(0.33 + 0.85, abs=1e-6)
        assert by_span[(3.26, 5.68)]["overlap_s"] == pytest.approx(0.33, abs=1e-6)
        # A lone segment overlaps nothing.
        assert by_span[(0.0, 0.91)]["overlap_s"] == pytest.approx(0.0)

    def test_same_speaker_never_counts_as_overlap(self):
        """Two adjacent segments of one speaker are bookkeeping, not overlapping speech."""
        doc = make_nemotron_document(segments=[
            {"start": 0.0, "end": 2.0, "speaker_index": 0, "speaker_id": "speaker_0"},
            {"start": 1.0, "end": 3.0, "speaker_index": 0, "speaker_id": "speaker_0"},
        ], speakers=["speaker_0"])
        rows = nemotron_turn_rows(doc, "v1")
        assert [row["overlap_s"] for row in rows] == [0.0, 0.0]

    def test_three_way_overlap_accumulates(self):
        doc = make_nemotron_document(segments=[
            {"start": 0.0, "end": 10.0, "speaker_index": 0, "speaker_id": "speaker_0"},
            {"start": 1.0, "end": 2.0, "speaker_index": 1, "speaker_id": "speaker_1"},
            {"start": 3.0, "end": 5.0, "speaker_index": 2, "speaker_id": "speaker_2"},
        ], speakers=["speaker_0", "speaker_1", "speaker_2"])
        rows = nemotron_turn_rows(doc, "v1")
        assert rows[0]["overlap_s"] == pytest.approx(3.0, abs=1e-6)

    def test_segments_sorted_and_turn_ids_dense(self):
        doc = make_nemotron_document(segments=list(reversed(make_nemotron_document()["segments"])))
        rows = nemotron_turn_rows(doc, "v1")
        assert [row["turn_id"] for row in rows] == [f"turn{i + 1:06d}" for i in range(len(rows))]
        starts = [row["start_time"] for row in rows]
        assert starts == sorted(starts)

    def test_diarization_type_names_this_engine(self):
        rows = nemotron_turn_rows(make_nemotron_document(), "v1")
        assert {row["diarization_type"] for row in rows} == {"overlapping"}

    def test_speaker_ids_are_kept_verbatim(self):
        """Nemotron's namespace must stay distinguishable from pyannote's SPEAKER_00."""
        rows = nemotron_turn_rows(make_nemotron_document(), "v1")
        assert {row["speaker_id"] for row in rows} == {"speaker_0", "speaker_1"}
        assert not any(row["speaker_id"].startswith("SPEAKER_") for row in rows)

    def test_zero_length_segment_is_dropped_not_negative(self):
        doc = make_nemotron_document(segments=[
            {"start": 1.0, "end": 1.0, "speaker_index": 0, "speaker_id": "speaker_0"},
            {"start": 2.0, "end": 3.0, "speaker_index": 1, "speaker_id": "speaker_1"},
        ])
        rows = nemotron_turn_rows(doc, "v1")
        assert [row["start_time"] for row in rows] == [2.0]

    def test_malformed_segment_rows_are_skipped(self):
        doc = make_nemotron_document(segments=[
            {"start": None, "end": 1.0, "speaker_id": "speaker_0"},
            {"start": 0.0, "end": 1.0, "speaker_id": ""},
            {"start": 0.0, "end": 2.0, "speaker_id": "speaker_1"},
        ])
        rows = nemotron_turn_rows(doc, "v1")
        assert [row["speaker_id"] for row in rows] == ["speaker_1"]

    def test_empty_document_yields_no_rows(self):
        assert nemotron_turn_rows(make_nemotron_document(segments=[]), "v1") == []


class TestNormalizeAndValidate:
    def test_writes_the_declared_table_with_its_own_schema(self, seeded):
        stage = DiarizationNemotronStage()
        summary = stage.normalize(seeded)
        table = read_table(seeded.artifact("speaker_turns_nemotron"))
        assert table.schema == SPEAKER_TURNS_NEMOTRON_SCHEMA
        assert table.num_rows == 4
        assert summary["turns"] == 4
        assert summary["speakers"] == 2
        assert summary["overlapping_turns"] == 3

    def test_validate_accepts_a_seeded_run(self, seeded):
        stage = DiarizationNemotronStage()
        stage.normalize(seeded)
        result = stage.validate(seeded)
        assert result["turns"] == 4
        assert result["overlapping_turns"] == 3

    def test_validate_rejects_a_table_from_a_different_model(self, seeded):
        stage = DiarizationNemotronStage()
        stage.normalize(seeded)
        seeded.config.diarization_nemotron.model = "nvidia/Nemotron-4-Diarization"
        with pytest.raises(ValidationError, match="raw result came from nvidia/Nemotron-3"):
            stage.validate(seeded)

    def test_validate_rejects_a_negative_overlap(self, seeded):
        stage = DiarizationNemotronStage()
        stage.normalize(seeded)
        import pyarrow.parquet as pq

        path = seeded.artifact("speaker_turns_nemotron")
        table = pq.read_table(path)
        col = [float(v) for v in table.column("overlap_s").to_pylist()]
        col[0] = -1.0
        pq.write_table(table.set_column(
            table.schema.get_field_index("overlap_s"), "overlap_s",
            pa.array(col, type=pa.float64())), path)
        with pytest.raises(ValidationError, match="negative overlap"):
            stage.validate(seeded)

    def test_validate_rejects_a_flattened_overlap_column(self, seeded):
        """A normalisation bug that zeroed the overlap must be caught, not admired."""
        stage = DiarizationNemotronStage()
        stage.normalize(seeded)
        import pyarrow.parquet as pq

        path = seeded.artifact("speaker_turns_nemotron")
        table = pq.read_table(path)
        col = [float(v) for v in table.column("overlap_s").to_pylist()]
        col[0] = -0.5  # still an interval-shaped table; only the invariant catches it
        pq.write_table(table.set_column(
            table.schema.get_field_index("overlap_s"), "overlap_s",
            pa.array(col, type=pa.float64())), path)
        with pytest.raises(ValidationError):
            stage.validate(seeded)

    def test_validate_rejects_more_speakers_than_allowed(self, seeded):
        stage = DiarizationNemotronStage()
        stage.normalize(seeded)
        seeded.config.diarization_nemotron.max_speakers = 1
        restamp(seeded, stage)
        with pytest.raises(ValidationError, match="at most 1"):
            stage.validate(seeded)

    def test_validate_rejects_a_stale_request(self, seeded):
        stage = DiarizationNemotronStage()
        stage.normalize(seeded)
        seeded.config.diarization_nemotron.threshold = 0.4
        with pytest.raises(ValidationError, match="different configuration"):
            stage.validate(seeded)

    def test_validate_rejects_a_missing_table(self, seeded):
        stage = DiarizationNemotronStage()
        seeded.artifact("nemotron_diarization_raw").write_text(
            json.dumps(make_nemotron_document()), encoding="utf-8")
        restamp(seeded, stage)
        with pytest.raises(ValidationError, match="speaker_turns_nemotron.parquet missing"):
            stage.validate(seeded)

    def test_silent_audio_still_produces_its_artifacts(self, seeded):
        """Zero segments is a completed stage, not a failure: both files must exist."""
        stage = DiarizationNemotronStage()
        raw = seeded.artifact("nemotron_diarization_raw")
        raw.write_text(json.dumps(make_nemotron_document(segments=[], speakers=[])),
                       encoding="utf-8")
        restamp(seeded, stage)
        summary = stage.normalize(seeded)
        assert summary == {"turns": 0, "speakers": 0, "overlapping_turns": 0,
                           "diarization_type": None}
        assert seeded.artifact("speaker_turns_nemotron").is_file()
        assert stage.validate(seeded)["turns"] == 0


class TestWorkerContract:
    @pytest.fixture
    def worker(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "nemotron_worker", PROJECT_ROOT / "workers" / "nemotron_diarization_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_result_json_default_matches_the_harness_convention(self, worker):
        """The worker's fallback result filename must be exactly what the harness reads.

        ``WorkerStage.run_model`` computes ``{stage_name}_worker_result.json`` and never
        passes ``--result-json``, so each worker's own default has to agree with it. Mine
        said ``nemotron_diarization_worker_result.json`` and every real run failed with
        "worker produced no result JSON" *after diarizing successfully* — found by the e2e
        test, invisible to every unit test, because nothing compared the two names.
        """
        from multimodal_pipeline.uv_worker import worker_result_path

        expected = worker_result_path(Path("/tmp/whatever"),
                                      "diarization_nemotron_worker_result.json").name
        parsed = worker.parse_args(["--audio", "/tmp/a.wav", "--output", "/tmp/whatever/o.json"])
        default = (parsed.result_json or
                   (parsed.output.parent / "diarization_nemotron_worker_result.json")).name
        assert default == expected

    def test_argv_matches_the_worker_parser(self, context, tmp_path, worker):
        """Every flag the stage passes must exist in the worker, and vice versa.

        Two hand-written lists drift silently: a flag only in the stage makes the worker
        exit with argparse code 2, and a flag only in the worker is unreachable config.
        """
        enable_environment(context, tmp_path)
        audio = context.artifact("audio")
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"x")
        argv = DiarizationNemotronStage().worker_argv(
            context, context.artifact("nemotron_diarization_raw"), "hash123")
        parsed = worker.parse_args(argv)
        assert parsed.model == context.config.diarization_nemotron.model
        assert parsed.threshold == context.config.diarization_nemotron.threshold
        assert parsed.max_speakers == context.config.diarization_nemotron.max_speakers
        assert parsed.cpu_fallback is True
        # BooleanOptionalAction gives --no-cpu-fallback for a False config value.
        context.config.diarization_nemotron.fallback_to_cpu = False
        argv = DiarizationNemotronStage().worker_argv(
            context, context.artifact("nemotron_diarization_raw"), "hash123")
        assert "--no-cpu-fallback" in argv
        assert worker.parse_args(argv).cpu_fallback is False

    def test_token_never_reaches_argv(self, context, tmp_path):
        import os

        enable_environment(context, tmp_path)
        os.environ["HF_TOKEN"] = "sk-super-secret-token"
        audio = context.artifact("audio")
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"x")
        argv = DiarizationNemotronStage().worker_argv(
            context, context.artifact("nemotron_diarization_raw"), "hash")
        assert "sk-super-secret-token" not in " ".join(argv)
        env = DiarizationNemotronStage().worker_environment(context)
        assert env["HF_TOKEN"] == "sk-super-secret-token"

    def test_token_absent_yields_no_env_entry(self, context, tmp_path, monkeypatch):
        enable_environment(context, tmp_path)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        assert "HF_TOKEN" not in DiarizationNemotronStage().worker_environment(context)

    def test_gpu_requirement_refuses_to_crawl_silently(self, worker):
        """The fallback policy is a pure function, so it is tested directly, not mocked.

        Mocking torch.cuda to test a cuda check would leave the branch green no matter what
        it returned, which is the kind of test AGENTS.md calls worse than no test.
        """
        assert worker.resolve_device(requested="cuda", cpu_fallback=True,
                                     cuda_available=False) == (
            "cpu", "torch.cuda.is_available() was False")
        assert worker.resolve_device(requested="cuda", cpu_fallback=True,
                                     cuda_available=True) == ("cuda", None)
        assert worker.resolve_device(requested="cpu", cpu_fallback=False,
                                     cuda_available=False) == ("cpu", None)
        with pytest.raises(RuntimeError, match="--no-cpu-fallback forbids degrading"):
            worker.resolve_device(requested="cuda", cpu_fallback=False, cuda_available=False)

    def test_bad_arguments_are_reported_before_any_import(self, worker, tmp_path):
        """The orchestrator environment has no torch; a bad flag must not look like that."""
        audio = tmp_path / "a.wav"
        audio.write_bytes(b"x")
        assert worker.main(["--audio", str(audio), "--output", str(tmp_path / "o.json"),
                            "--threshold", "0"]) == 1
        payload = json.loads((tmp_path / "diarization_nemotron_worker_result.json").read_text())
        assert payload["status"] == "error"
        assert "threshold" in payload["error"]

    def test_missing_audio_is_reported_as_missing_audio(self, worker, tmp_path):
        result = worker.main(["--audio", str(tmp_path / "nope.wav"),
                              "--output", str(tmp_path / "o.json")])
        assert result == 1
        payload = json.loads((tmp_path / "diarization_nemotron_worker_result.json").read_text())
        assert "audio file not found" in payload["error"]


class TestConfigParsing:
    def test_max_speakers_above_the_model_channel_count_is_refused(self):
        from multimodal_pipeline.config import DiarizationNemotronConfig

        with pytest.raises(ValueError, match="8 speaker channels"):
            DiarizationNemotronConfig(max_speakers=9)

    def test_threshold_outside_zero_one_is_refused(self):
        from multimodal_pipeline.config import DiarizationNemotronConfig

        with pytest.raises(ValueError, match="probability"):
            DiarizationNemotronConfig(threshold=1.5)

    def test_unknown_device_is_refused(self):
        from multimodal_pipeline.config import DiarizationNemotronConfig

        with pytest.raises(ValueError, match="cuda"):
            DiarizationNemotronConfig(device="tpu")

    def test_a_dead_knob_stays_out(self):
        """There is no retry plumbing in run_worker, so no retry knob is offered."""
        from multimodal_pipeline.config import DiarizationNemotronConfig

        assert "max_retries" not in DiarizationNemotronConfig.model_fields
        assert "operating_point" not in DiarizationNemotronConfig.model_fields
