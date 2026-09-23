"""Configuration: YAML parsing, validation, defaults, hashing, masking."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from multimodal_pipeline.config import load_config, mask_command, mask_secrets, stable_hash


def write_config(root: Path, payload: dict, name: str = "c.yaml") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def base_payload(root: Path) -> dict:
    return {
        "project_root": str(root),
        "input": {"directory": str(root / "in")},
        "output": {"directory": str(root / "out")},
    }


class TestLoading:
    def test_minimal_config_gets_defaults(self, tmp_path: Path) -> None:
        config = load_config(write_config(tmp_path, base_payload(tmp_path)))
        assert config.whisperx.model == "large-v3"
        assert config.diarization.pipeline == "pyannote/speaker-diarization-community-1"
        assert config.openpose.root == Path("/opt/openpose")
        assert config.acoustic.time_step == 0.01
        assert config.execution.gpu == 0
        assert config.input.extensions == [".avi", ".mkv", ".mov", ".mp4", ".webm"]

    def test_project_root_defaults_to_config_parent(self, tmp_path: Path) -> None:
        path = tmp_path / "proj" / "config" / "c.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({k: v for k, v in base_payload(tmp_path).items() if k != "project_root"}))
        config = load_config(path)
        assert config.project_root == tmp_path / "proj"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_non_mapping_root_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- one\n- two\n")
        with pytest.raises(ValueError):
            load_config(path)

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["whisperx"] = {"modle": "large-v3"}
        with pytest.raises(Exception) as excinfo:
            load_config(write_config(tmp_path, payload))
        assert "modle" in str(excinfo.value)

    def test_extension_normalisation(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["input"] = {"directory": str(tmp_path / "in"), "extensions": ["MP4", ".mov", "mkv", " "]}
        config = load_config(write_config(tmp_path, payload))
        assert config.input.extensions == [".mkv", ".mov", ".mp4"]

    def test_empty_extensions_rejected(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["input"] = {"directory": str(tmp_path / "in"), "extensions": ["  "]}
        with pytest.raises(Exception):
            load_config(write_config(tmp_path, payload))


class TestValidationRules:
    def test_only_sequential_execution(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["execution"] = {"mode": "parallel"}
        with pytest.raises(Exception):
            load_config(write_config(tmp_path, payload))

    def test_acoustic_pitch_floor_must_be_below_ceiling(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["acoustic"] = {"pitch_floor": 600, "pitch_ceiling": 500}
        with pytest.raises(Exception):
            load_config(write_config(tmp_path, payload))

    def test_formant_count_bounds(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["acoustic"] = {"number_of_formants": 9}
        with pytest.raises(Exception):
            load_config(write_config(tmp_path, payload))

    def test_unknown_diarization_provider_rejected(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["diarization"] = {"provider": "silero"}
        with pytest.raises(Exception):
            load_config(write_config(tmp_path, payload))


class TestEnvironmentInterpolation:
    def test_env_var_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MP_LITELLM_KEY", "sk-from-env")
        payload = base_payload(tmp_path)
        payload["translation"] = {"api_key": "${MP_LITELLM_KEY}", "base_url": "http://h:1/v1", "model": "m"}
        config = load_config(write_config(tmp_path, payload))
        assert config.translation.api_key == "sk-from-env"

    def test_env_default_used_when_unset(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["input"]["directory"] = "${MP_DEFINITELY_UNSET_DIR:-/fallback/videos}"
        config = load_config(write_config(tmp_path, payload))
        assert config.input.directory == Path("/fallback/videos")

    def test_unresolved_placeholder_left_alone(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload.setdefault("logging", {})["level"] = "${MP_NOPE}"
        config = load_config(write_config(tmp_path, payload))
        assert config.logging.level == "${MP_NOPE}"


class TestTranslationEndpointReadiness:
    def test_example_config_counts_as_unconfigured(self, tmp_path: Path) -> None:
        config = load_config(write_config(tmp_path, base_payload(tmp_path)))
        assert config.translation.endpoint_configured is False

    def test_mock_provider_always_ready(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["translation"] = {"provider": "mock"}
        config = load_config(write_config(tmp_path, payload))
        assert config.translation.endpoint_configured is True

    def test_real_endpoint_configured(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["translation"] = {"base_url": "http://10.0.0.5:4000/v1", "api_key": "sk-real", "model": "claude"}
        config = load_config(write_config(tmp_path, payload))
        assert config.translation.endpoint_configured is True


class TestMasking:
    def test_api_key_masked_in_dict(self) -> None:
        masked = mask_secrets({"translation": {"api_key": "sk-very-secret", "model": "m"},
                               "diarization": {"hf_token_env": "HF_TOKEN"}})
        assert masked["translation"]["api_key"] == "***masked***"
        assert masked["translation"]["model"] == "m"
        # The *name* of the env var is not a secret and must survive.
        assert masked["diarization"]["hf_token_env"] == "HF_TOKEN"

    def test_masked_config_dict_hides_key(self, tmp_path: Path) -> None:
        payload = base_payload(tmp_path)
        payload["translation"] = {"api_key": "sk-super-secret", "model": "m", "base_url": "http://h:1/v1"}
        config = load_config(write_config(tmp_path, payload))
        dumped = yaml.safe_dump(config.masked_dict())
        assert "sk-super-secret" not in dumped

    def test_command_token_masked(self) -> None:
        argv = ["worker.py", "--hf-token", "hf_real_token", "--model", "large-v3"]
        masked = mask_command(argv)
        assert masked[2] == "***masked***"
        assert masked[4] == "large-v3"

    def test_api_key_style_flag_masked(self) -> None:
        assert mask_command(["x", "--api-key", "abc"])[2] == "***masked***"


class TestHashing:
    def test_hash_stable_and_order_insensitive(self) -> None:
        assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})

    def test_hash_changes_with_value(self) -> None:
        assert stable_hash({"model": "large-v3"}) != stable_hash({"model": "large-v2"})

    def test_config_hash_changes_when_model_changes(self, tmp_path: Path) -> None:
        from multimodal_pipeline.provenance import configuration_hash

        first = load_config(write_config(tmp_path, base_payload(tmp_path)))
        payload = base_payload(tmp_path)
        payload["whisperx"] = {"model": "medium"}
        second_path = tmp_path / "second.yaml"
        second_path.write_text(yaml.safe_dump(payload))
        second = load_config(second_path)
        assert configuration_hash(first) != configuration_hash(second)

    def test_config_hash_ignores_path_but_not_content(self, tmp_path: Path) -> None:
        from multimodal_pipeline.provenance import configuration_hash

        first = load_config(write_config(tmp_path, base_payload(tmp_path), "a.yaml"))
        second = load_config(write_config(tmp_path, base_payload(tmp_path), "b.yaml"))
        assert first.config_path != second.config_path
        assert configuration_hash(first) == configuration_hash(second)
