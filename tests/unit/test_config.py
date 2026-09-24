"""Configuration: YAML parsing, validation, defaults, hashing, masking."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from multimodal_pipeline.config import (
    load_config,
    load_dotenv,
    mask_command,
    mask_secrets,
    stable_hash,
)
from multimodal_pipeline.exceptions import ConfigError


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


class TestLoadDotenv:
    """.env is the only place credentials live; the YAML never holds one."""

    def test_sets_variables_from_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("LITELLM_MODEL", raising=False)
        dotenv = tmp_path / ".env"
        dotenv.write_text("HF_TOKEN=abc123\nLITELLM_MODEL=chat\n")
        keys = load_dotenv(dotenv)
        assert sorted(keys) == ["HF_TOKEN", "LITELLM_MODEL"]
        assert os.environ["HF_TOKEN"] == "abc123"

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_dotenv(tmp_path / ".env") == []

    def test_existing_environment_wins(self, tmp_path):
        """A stale .env must not shadow an explicit override or a CI secret."""
        dotenv = tmp_path / ".env"
        dotenv.write_text("HF_TOKEN=from-file\n")
        os.environ["HF_TOKEN"] = "from-shell"
        try:
            assert load_dotenv(dotenv) == []
            assert os.environ["HF_TOKEN"] == "from-shell"
        finally:
            os.environ.pop("HF_TOKEN", None)

    def test_override_flag_replaces(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("HF_TOKEN=from-file\n")
        os.environ["HF_TOKEN"] = "from-shell"
        try:
            assert load_dotenv(dotenv, override=True) == ["HF_TOKEN"]
            assert os.environ["HF_TOKEN"] == "from-file"
        finally:
            os.environ.pop("HF_TOKEN", None)

    def test_quotes_and_comments(self, tmp_path, monkeypatch):
        for key in ("HF_TOKEN", "HF_HASH_TOKEN", "LITELLM_BASE_URL", "LITELLM_MODEL"):
            monkeypatch.delenv(key, raising=False)
        dotenv = tmp_path / ".env"
        dotenv.write_text(
            "# a comment line\n"
            'export HF_TOKEN="quoted token"  # trailing\n'
            'HF_HASH_TOKEN="a#b"  # hash inside quotes is data\n'
            "LITELLM_BASE_URL=https://host/v1 # inline comment\n"
            "LITELLM_MODEL='single'\n"
        )
        load_dotenv(dotenv)
        assert os.environ["HF_TOKEN"] == "quoted token"
        assert os.environ["HF_HASH_TOKEN"] == "a#b"
        assert os.environ["LITELLM_BASE_URL"] == "https://host/v1"
        assert os.environ["LITELLM_MODEL"] == "single"

    def test_hash_without_space_is_part_of_value(self, tmp_path, monkeypatch):
        """A '#' only starts a comment when spaced out; tokens contain hashes."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        dotenv = tmp_path / ".env"
        dotenv.write_text("HF_TOKEN=abc#def\n")
        load_dotenv(dotenv)
        assert os.environ["HF_TOKEN"] == "abc#def"

    def test_malformed_line_names_the_line(self, tmp_path):
        dotenv = tmp_path / ".env"
        dotenv.write_text("HF_TOKEN=ok\nthis is not an assignment\n")
        with pytest.raises(ConfigError, match=r"\.env:2: expected KEY=value"):
            load_dotenv(dotenv)

    def test_blank_lines_and_whitespace_ok(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        dotenv = tmp_path / ".env"
        dotenv.write_text("\n\n   HF_TOKEN = spaced   \n\n")
        load_dotenv(dotenv)
        assert os.environ["HF_TOKEN"] == "spaced"


class TestCredentialsReachConfigFromDotenv:
    # The autouse conftest fixture disables the implicit project-root read to keep the
    # suite independent of this machine's real credentials. These two tests are *about*
    # that read, so they opt back in.
    @pytest.fixture(autouse=True)
    def _allow_implicit_dotenv(self, monkeypatch):
        monkeypatch.delenv("MULTIMODAL_PIPELINE_NO_DOTENV", raising=False)

    def test_load_config_reads_project_dotenv_without_export(self, tmp_path, monkeypatch):
        """The whole point: no shell wrapper, no `source`, just a .env in the root."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        root = tmp_path / "proj"
        (root / "config").mkdir(parents=True)
        (root / ".env").write_text("HF_TOKEN=from-dotenv\n")
        conf = root / "config" / "c.yaml"
        conf.write_text(
            "input:\n  directory: in\noutput:\n  directory: out\n"
            "diarization:\n  hf_token_env: HF_TOKEN\n"
        )
        cfg = load_config(conf)
        # The stage reads the credential from the variable named in the config, so
        # the .env value has to be in the environment by the time it looks.
        assert cfg.diarization.hf_token_env == "HF_TOKEN"
        assert os.environ[cfg.diarization.hf_token_env] == "from-dotenv"

    def test_explicit_environment_beats_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "from-shell")
        root = tmp_path / "proj"
        (root / "config").mkdir(parents=True)
        (root / ".env").write_text("HF_TOKEN=from-dotenv\n")
        conf = root / "config" / "c.yaml"
        conf.write_text(
            "input:\n  directory: in\noutput:\n  directory: out\n"
            "diarization:\n  hf_token_env: HF_TOKEN\n"
        )
        cfg = load_config(conf)
        assert os.environ[cfg.diarization.hf_token_env] == "from-shell"

    def test_opt_out_variable_suppresses_the_implicit_read(self, tmp_path, monkeypatch):
        """The escape hatch the test suite relies on must be real, not incidental."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("MULTIMODAL_PIPELINE_NO_DOTENV", "1")
        root = tmp_path / "proj"
        (root / "config").mkdir(parents=True)
        (root / ".env").write_text("HF_TOKEN=from-dotenv\n")
        conf = root / "config" / "c.yaml"
        conf.write_text("input:\n  directory: in\noutput:\n  directory: out\n")
        load_config(conf)
        assert "HF_TOKEN" not in os.environ

    def test_explicit_call_still_works_when_opted_out(self, tmp_path, monkeypatch):
        """The opt-out covers the implicit read only; the loader itself stays usable."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("MULTIMODAL_PIPELINE_NO_DOTENV", "1")
        dotenv = tmp_path / ".env"
        dotenv.write_text("HF_TOKEN=explicit\n")
        assert load_dotenv(dotenv) == ["HF_TOKEN"]
        assert os.environ["HF_TOKEN"] == "explicit"
