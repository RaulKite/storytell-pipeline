"""Optional native TalkNet video: configuration, publication and resume seams."""
from __future__ import annotations

import pytest

from multimodal_pipeline.config import ActiveSpeakerConfig
from multimodal_pipeline.stages.activespeaker import ActiveSpeakerStage


def test_video_is_opt_in_and_bound_to_worker_request(context):
    assert ActiveSpeakerConfig().write_video is False
    stage = ActiveSpeakerStage()
    context.config.activespeaker.talknet_root = context.config.project_root
    audio = context.artifact('audio')
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b'fixture audio')
    before = stage.request_digest(context)
    context.config.activespeaker.write_video = True
    assert stage.request_digest(context) != before
    args = stage.worker_argv(context, context.artifact('activespeaker_raw'), 'digest')
    assert args[args.index('--output-video') + 1] == str(
        context.artifact('activespeaker_video_raw'))
    context.config.activespeaker.write_video = False
    assert '--output-video' not in stage.worker_argv(
        context, context.artifact('activespeaker_raw'), 'digest')


@pytest.fixture
def worker():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "workers/activespeaker_worker.py"
    spec = importlib.util.spec_from_file_location("asd_video_worker", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def avi(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required for real media publication")
    video = tmp_path / "native.avi"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x64:r=25",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
        "-t", "0.4", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(video),
    ], check=True)
    return video


@pytest.fixture
def seeded_video(context, avi):
    import json
    from tests.unit.test_activespeaker import make_frames_document
    from multimodal_pipeline.stages.base import stamp_raw

    context.config.activespeaker.write_video = True
    raw = context.artifact('activespeaker_raw')
    raw.write_text(json.dumps(make_frames_document()))
    context.artifact('metadata').write_text(json.dumps({'duration_seconds': 4.2}))
    stage = ActiveSpeakerStage()
    stamp_raw(raw, request=stage.request(context), digest=stage.request_digest(context), worker=None)
    stage.normalize(context)
    for name in ('activespeaker_tracks_pkl', 'activespeaker_scores_pkl', 'activespeaker_scenes_csv'):
        context.artifact(name).write_bytes(b'raw fixture')
    context.artifact('activespeaker_video_raw').write_bytes(avi.read_bytes())
    return context


def test_optional_video_absence_and_empty_video_invalidate_enabled_output(seeded_video):
    stage = ActiveSpeakerStage()
    video = seeded_video.artifact('activespeaker_video_raw')
    assert stage.outputs_present(seeded_video)
    video.write_bytes(b'')
    assert not stage.outputs_present(seeded_video)
    video.unlink()
    assert not stage.outputs_present(seeded_video)
    seeded_video.config.activespeaker.write_video = False
    assert stage.outputs_present(seeded_video)


@pytest.mark.parametrize('missing', [False, True])
def test_resume_repairs_missing_video_despite_matching_raw_request(seeded_video, avi, missing):
    from multimodal_pipeline.stages.base import stamp_raw

    class ModelBoundary(ActiveSpeakerStage):
        runs = 0

        def run_model(self, ctx, request, raw_path, digest):
            self.runs += 1
            ctx.artifact('activespeaker_video_raw').write_bytes(avi.read_bytes())
            stamp_raw(raw_path, request=request, digest=digest, worker=None)

    stage = ModelBoundary()
    if missing:
        seeded_video.artifact('activespeaker_video_raw').unlink()
    result = stage.execute(seeded_video)
    assert stage.runs == int(missing)
    assert result['extra']['raw_reused'] is (not missing)
    assert stage.outputs_present(seeded_video)
    stage.validate(seeded_video)


def test_validation_requires_enabled_video_but_off_preserves_existing_bytes(seeded_video):
    from multimodal_pipeline.exceptions import ValidationError
    from multimodal_pipeline.stages.base import stamp_raw

    stage = ActiveSpeakerStage()
    video = seeded_video.artifact('activespeaker_video_raw')
    original = video.read_bytes()
    video.write_bytes(b'')
    with pytest.raises(ValidationError, match='annotated video'):
        stage.validate(seeded_video)
    video.write_bytes(original)
    seeded_video.config.activespeaker.write_video = False
    raw = seeded_video.artifact('activespeaker_raw')
    stamp_raw(raw, request=stage.request(seeded_video), digest=stage.request_digest(seeded_video), worker=None)
    stage.execute(seeded_video)
    assert video.read_bytes() == original


def test_native_video_survives_temporary_cleanup_byte_identically(worker, avi, tmp_path):
    import tempfile
    from pathlib import Path

    expected = avi.read_bytes()
    destination = tmp_path / "published with spaces" / "video_out.avi"
    with tempfile.TemporaryDirectory(dir=tmp_path) as workspace:
        source = Path(workspace) / "video_out.avi"
        source.write_bytes(expected)
        worker.retain_video(source, destination, "ffprobe")
    assert not source.exists()
    assert destination.read_bytes() == expected
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("contents", [None, b"", b"not an AVI"])
def test_missing_empty_or_invalid_video_does_not_replace_previous(worker, tmp_path, contents):
    source = tmp_path / "native.avi"
    if contents is not None:
        source.write_bytes(contents)
    destination = tmp_path / "video_out.avi"
    destination.write_bytes(b"previous operator output")
    with pytest.raises(worker.WorkerFailure):
        worker.retain_video(source, destination, "ffprobe")
    assert destination.read_bytes() == b"previous operator output"


def test_talknet_visualisation_uses_safe_staged_audio(worker, avi, tmp_path):
    import json

    root = tmp_path / "checkout with spaces"
    root.mkdir()
    (root / "run_talknet.py").write_text('''
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--videoFolder')
p.add_argument('--audioFilePath')
p.add_argument('--visualisation', action='store_true')
a, _ = p.parse_known_args()
w = Path(a.videoFolder)
(w / 'pywork').mkdir(exist_ok=True)
(w / 'pywork/tracks.pckl').write_bytes(b'tracks')
(w / 'pywork/scores.pckl').write_bytes(b'scores')
(w / 'observed.json').write_text(json.dumps(vars(a)))
''')
    temp = tmp_path / "work"
    temp.mkdir()
    audio = tmp_path / "audio $(touch HACKED).wav"
    audio.write_bytes(b"audio bytes")
    worker.run_talknet(root, temp, temp / 'scenes.csv', audio, 'cpu', write_video=True)
    observed = json.loads((temp / "observed.json").read_text())
    assert observed['visualisation'] is True
    assert observed['audioFilePath'] == str(temp / 'audio.wav')
    assert (temp / 'audio.wav').read_bytes() == audio.read_bytes()


def test_unsafe_temporary_path_is_rejected_before_native_shell(worker, tmp_path):
    root = tmp_path / 'checkout'
    root.mkdir()
    (root / 'run_talknet.py').write_text("raise RuntimeError('must not run')")
    with pytest.raises(worker.WorkerFailure, match='shell-safe'):
        worker.run_talknet(root, tmp_path / 'unsafe;path', tmp_path / 'scenes.csv',
                          tmp_path / 'audio.wav', 'cpu', write_video=True)


def test_silent_video_is_not_reported_as_annotated_with_audio(worker, avi, tmp_path):
    import subprocess

    silent = tmp_path / "silent.avi"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(avi), "-an", "-c:v", "copy",
                    str(silent)], check=True)
    with pytest.raises(worker.WorkerFailure, match="audio"):
        worker.retain_video(silent, tmp_path / "published.avi", "ffprobe")
