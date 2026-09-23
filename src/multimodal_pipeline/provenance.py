"""Reproducibility provenance: exact tools, models and configuration per video.

Three files are written per video dataset:

* ``provenance/config.json``       masked, fully-resolved configuration;
* ``provenance/tools.json``        machine + executable + model inventory;
* ``provenance/processing.json``   what actually ran, when, with which hashes.

Credential-shaped values are masked before anything is written.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION, __version__
from .artifacts import VideoPaths, atomic_write_json
from .config import PipelineConfig, configuration_hash, mask_secrets
from .subprocess_utils import probe_version, which


def _read_text(path: Path, limit: int = 4000) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except OSError:
        return None


def cpu_count() -> int:
    return os.cpu_count() or 1


def total_memory_gib() -> float | None:
    try:
        return round((os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024**3, 2)
    except (ValueError, OSError):  # pragma: no cover - non-posix
        return None


def nvidia_smi_report(timeout: float = 15.0) -> dict[str, Any]:
    """GPU model/driver from ``nvidia-smi`` (no torch import, cheap and honest)."""
    binary = which("nvidia-smi")
    if not binary:
        return {"available": False, "reason": "nvidia-smi not on PATH"}
    result = probe_version(
        [binary, "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"],
        timeout=timeout,
    )
    if not result:
        return {"available": False, "reason": "nvidia-smi produced no output"}
    gpus = []
    driver = None
    for line in result.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            driver = driver or parts[1]
            gpus.append({"name": parts[0], "driver_version": parts[1], "memory": parts[2],
                         "compute_capability": parts[3] if len(parts) > 3 else None})
    return {"available": True, "driver_version": driver, "count": len(gpus), "devices": gpus}


def cuda_toolkit_report() -> dict[str, Any]:
    report: dict[str, Any] = {"nvcc": probe_version(["nvcc", "--version"]) or None}
    for candidate in ("/usr/local/cuda/version.json", "/usr/local/cuda/version.txt"):
        text = _read_text(Path(candidate))
        if text:
            report["toolkit_path"] = candidate
            report["toolkit_info"] = text[:500]
            break
    return report


def system_report() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "os": pretty_os_name() or platform.system(),
        "os_id": os_release_id(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu_model() or platform.processor(),
        "cpu_count": cpu_count(),
        "memory_total_gib": total_memory_gib(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "uv": probe_version(["uv", "--version"]),
    }


def pretty_os_name() -> str | None:
    text = _read_text(Path("/etc/os-release"), 800)
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return None


def os_release_id() -> str | None:
    text = _read_text(Path("/etc/os-release"), 800)
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("ID="):
            return line.split("=", 1)[1].strip('"')
    return None


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def openpose_report(config: PipelineConfig) -> dict[str, Any]:
    """Discover binary + models under ``openpose.root`` without assuming layout."""
    root = Path(config.openpose.root)
    report: dict[str, Any] = {"root": str(root), "available": root.is_dir()}
    if not root.is_dir():
        return report
    binary = config.openpose.executable
    candidates = [binary] if binary != "auto" else [
        root / "build/examples/openpose/openpose.bin",
        root / "bin/openpose.bin",
        root / "openpose.bin",
    ]
    resolved = next((Path(c) for c in candidates if Path(c).is_file()), None)
    report["executable"] = str(resolved) if resolved else None
    model_folder = config.openpose.model_folder
    model_candidates = [Path(model_folder)] if model_folder != "auto" else [root / "models", root / "share/openpose/models"]
    resolved_models = next((Path(m) for m in model_candidates if Path(m).is_dir()), None)
    report["model_folder"] = str(resolved_models) if resolved_models else None
    if resolved_models:
        report["models"] = {
            "body_25_deploy": _first_existing(resolved_models / "pose/body_25/pose_deploy.prototxt"),
            "body_25_caffemodel": _first_existing(resolved_models / "pose/body_25/pose_iter_584000.caffemodel"),
            "hand_deploy": _first_existing(resolved_models / "hand/pose_deploy.prototxt"),
            "hand_caffemodel": _first_existing(resolved_models / "hand/pose_iter_102000.caffemodel"),
            "face_deploy": _first_existing(resolved_models / "face/pose_deploy.prototxt"),
            "face_caffemodel": _first_existing(resolved_models / "face/pose_iter_116000.caffemodel"),
        }
    report["version_probe"] = probe_version([str(resolved), "--version"]) if resolved else None
    return report


def _first_existing(path: Path) -> str | None:
    return str(path) if path.is_file() else None


def tools_report(config: PipelineConfig) -> dict[str, Any]:
    """The machine inventory recorded once per video (``tools.json``)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pipeline_version": __version__,
        "system": system_report(),
        "gpu": nvidia_smi_report(),
        "cuda": cuda_toolkit_report(),
        "ffmpeg": {
            "ffmpeg": probe_version([config.ffmpeg.executable, "-version"]),
            "ffprobe": probe_version([config.ffmpeg.ffprobe, "-version"]),
            "resolved_ffmpeg": which(config.ffmpeg.executable),
            "resolved_ffprobe": which(config.ffmpeg.ffprobe),
        },
        "openpose": openpose_report(config),
        "uv_projects": {
            name: {"project": str(config.resolve(cfg.uv_project)), "exists": config.resolve(cfg.uv_project).is_dir()}
            for name, cfg in config.stage_configs.items()
            if hasattr(cfg, "uv_project")
        },
    }


def config_report(config: PipelineConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config.config_path) if config.config_path else None,
        "config_hash": configuration_hash(config),
        "resolved_configuration": config.masked_dict(),
    }


def processing_report(
    *,
    video_id: str,
    config: PipelineConfig,
    state_summary: dict[str, Any],
    stage_records: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
    started_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "pipeline_version": __version__,
        "pipeline_git_commit": git_commit(config.project_root),
        "run_started_at": started_at,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "status": state_summary.get("status"),
        "stages": stage_records,
        "artifacts": artifacts,
    }


def git_commit(root: Path) -> str | None:
    head = Path(root) / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            ref_path = Path(root) / ".git" / ref[5:]
            return ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None
        return ref
    except OSError:
        return None


def write_provenance(paths: VideoPaths, config: PipelineConfig, tools: dict[str, Any],
                     processing: dict[str, Any]) -> dict[str, Path]:
    written = {
        "config": atomic_write_json(paths.artifact("provenance_config"), mask_secrets(config_report(config))),
        "tools": atomic_write_json(paths.artifact("provenance_tools"), tools),
        "processing": atomic_write_json(paths.artifact("provenance_processing"), processing),
    }
    return written
