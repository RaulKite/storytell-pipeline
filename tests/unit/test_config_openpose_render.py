"""Config defaults and YAML round-trip for the rendered-skeleton switches.

``openpose.write_images`` is off by default on purpose, so the default is the
behaviour that protects every existing dataset and has to be pinned: a default that
quietly flipped to ``True`` would change the ``openpose`` stage fingerprint and re-run
the slowest stage of the pipeline on every video already processed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from multimodal_pipeline.config import load_config


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


class TestDefaults:
    def test_rendering_is_off_by_default(self, tmp_path: Path) -> None:
        config = load_config(write_config(tmp_path, base_payload(tmp_path)))
        assert config.openpose.write_images is False

    def test_no_downscale_is_declared_by_default(self, tmp_path: Path) -> None:
        """``None`` means "ask nothing of OpenPose", not "downscale to zero"."""
        config = load_config(write_config(tmp_path, base_payload(tmp_path)))
        assert config.openpose.image_max_side is None

    def test_the_defaults_survive_a_dump(self, tmp_path: Path) -> None:
        config = load_config(write_config(tmp_path, base_payload(tmp_path)))
        dumped = config.model_dump(mode="json")["openpose"]
        assert dumped["write_images"] is False
        assert dumped["image_max_side"] is None


class TestYamlRoundTrip:
    @pytest.mark.parametrize("payload", [
        {"write_images": True},
        {"write_images": True, "image_max_side": 640},
        {"write_images": False, "image_max_side": 320},
    ])
    def test_values_are_read_back(self, tmp_path: Path, payload: dict) -> None:
        data = base_payload(tmp_path)
        data["openpose"] = {"root": "/opt/openpose", **payload}
        config = load_config(write_config(tmp_path, data, "round_trip.yaml"))
        for key, value in payload.items():
            assert getattr(config.openpose, key) == value

    def test_they_coexist_with_the_other_openpose_keys(self, tmp_path: Path) -> None:
        data = base_payload(tmp_path)
        data["openpose"] = {
            "root": "/opt/openpose",
            "gpu": 1,
            "hands": {"enabled": False},
            "write_images": True,
            "image_max_side": 960,
        }
        config = load_config(write_config(tmp_path, data, "coexist.yaml"))
        assert config.openpose.gpu == 1
        assert config.openpose.hands_enabled is False
        assert config.openpose.write_images is True
        assert config.openpose.image_max_side == 960

    def test_a_zero_side_is_rejected(self, tmp_path: Path) -> None:
        """``--output_resolution 0x0`` would be a render that writes nothing useful."""
        data = base_payload(tmp_path)
        data["openpose"] = {"write_images": True, "image_max_side": 0}
        with pytest.raises(ValueError):
            load_config(write_config(tmp_path, data, "zero_side.yaml"))

    def test_a_non_numeric_side_is_rejected(self, tmp_path: Path) -> None:
        data = base_payload(tmp_path)
        data["openpose"] = {"image_max_side": "640 pixels"}
        with pytest.raises(ValueError):
            load_config(write_config(tmp_path, data, "bad_side.yaml"))

    def test_an_unknown_openpose_key_is_still_rejected(self, tmp_path: Path) -> None:
        """The render flags did not loosen ``extra = "forbid"``."""
        data = base_payload(tmp_path)
        data["openpose"] = {"write_images": True, "write_video": True}
        with pytest.raises(ValueError):
            load_config(write_config(tmp_path, data, "unknown.yaml"))


class TestShippedExample:
    def test_the_example_config_documents_both_keys(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        text = (repo / "config" / "config.example.yaml").read_text(encoding="utf-8")
        assert "write_images" in text
        assert "image_max_side" in text
        # A commented-out key with no cost note is how a 4 GB render happens.
        assert "-1x-1" in text
        openpose_block = text.split("openpose:", 1)[1].split("\nactivespeaker:", 1)[0]
        assert "write_images" in openpose_block
        assert "image_max_side" in openpose_block
