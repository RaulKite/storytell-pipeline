"""Native skeleton video is opt-in and independent of still-image retention."""
from pathlib import Path

import pytest

from multimodal_pipeline.config import OpenPoseConfig
from multimodal_pipeline.exceptions import StageError
from multimodal_pipeline.stages.openpose import OpenPoseStage


def test_video_only_enables_rendering(context, tmp_path):
    binary = tmp_path / 'openpose'
    binary.write_text('#!/bin/sh\nexit 0\n')
    binary.chmod(0o755)
    models = tmp_path / 'models'
    models.mkdir()
    context.config.openpose = OpenPoseConfig(
        executable=str(binary), model_folder=str(models),
        write_video=True, write_images=False,
    )
    argv = OpenPoseStage().build_command(context, context.artifact('pose_raw'))
    assert argv[argv.index('--render_pose') + 1] != '0'
    assert '--write_images' not in argv
    assert Path(argv[argv.index('--write_video') + 1]).name == 'skeletons.mp4'
    # Muxing audio is done by argv-based ffmpeg, never upstream's shell command.
    assert '--write_video_with_audio' not in argv


def test_video_only_refuses_no_drawable_module(context):
    context.config.openpose = OpenPoseConfig(
        write_video=True, body={'enabled': False}, hands={'enabled': False},
        face={'enabled': False},
    )
    with pytest.raises(StageError, match='nothing would be drawn'):
        OpenPoseStage().refuse_undrawn_render(context)


def test_enabled_video_must_exist_for_reuse(context):
    stage = OpenPoseStage()
    for name in ('pose_body', 'pose_hands', 'pose_face'):
        context.artifact(name).touch()
    assert stage.outputs_present(context)
    context.config.openpose = OpenPoseConfig(write_video=True)
    assert not stage.outputs_present(context)
    video = context.artifact('pose_skeleton_video')
    video.write_bytes(b'retained')
    assert stage.outputs_present(context)
    video.write_bytes(b'')
    assert not stage.outputs_present(context)


@pytest.fixture
def real_media(tmp_path):
    import shutil
    from multimodal_pipeline.subprocess_utils import run_command
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')
    native = tmp_path / 'native.mp4'
    source = tmp_path / "source audio's name.mp4"
    run_command(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                 'color=c=black:s=64x48:r=25:d=0.4', '-c:v', 'libx264',
                 '-pix_fmt', 'yuv420p', str(native)])
    run_command(['ffmpeg', '-v', 'error', '-i', str(native), '-f', 'lavfi',
                 '-i', 'sine=frequency=440:duration=0.4', '-c:v', 'copy',
                 '-c:a', 'aac', '-shortest', str(source)])
    return native, source


@pytest.mark.parametrize('with_audio', [True, False])
def test_preview_muxes_and_decodes_real_media(real_media, tmp_path, with_audio):
    from multimodal_pipeline.skeleton_video import publish_skeleton_video
    from multimodal_pipeline.stages.metadata import ffprobe_streams
    from multimodal_pipeline.subprocess_utils import run_command
    native, source = real_media
    destination = tmp_path / 'output' / 'skeletons.mp4'
    publish_skeleton_video(native, source if with_audio else native, destination)
    kinds = {s['codec_type'] for s in ffprobe_streams(destination)['streams']}
    assert kinds == ({'video', 'audio'} if with_audio else {'video'})
    assert run_command(['ffmpeg', '-v', 'error', '-i', str(destination),
                        '-f', 'null', '-']).returncode == 0


def test_failed_mux_preserves_previous_video(real_media, tmp_path):
    from multimodal_pipeline.skeleton_video import publish_skeleton_video
    from multimodal_pipeline.subprocess_utils import CommandError
    native, source = real_media
    destination = tmp_path / 'skeletons.mp4'
    destination.write_bytes(b'previous preview')
    with pytest.raises(CommandError):
        publish_skeleton_video(native, source, destination, ffmpeg='/bin/false')
    assert destination.read_bytes() == b'previous preview'


@pytest.fixture
def native_stage_context(context, real_media, tmp_path):
    import json
    import sys
    from dataclasses import replace
    import pyarrow as pa
    from multimodal_pipeline.schemas import FRAME_INDEX_SCHEMA, write_table
    from multimodal_pipeline.stages.metadata import ffprobe_streams
    _, source = real_media
    context.source = replace(context.source, path=source)
    binary = tmp_path / 'native-renderer'
    binary.write_text(f'#!{sys.executable}\n' + '''import json, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
def val(flag): return args[args.index(flag)+1]
raw = Path(val('--write_json'))
for i in range(10):
    (raw / f'clip_{i:012d}_keypoints.json').write_text(json.dumps({'people': []}))
subprocess.run(['ffmpeg', '-v', 'error', '-i', val('--video'), '-an',
                '-c:v', 'copy', val('--write_video')], check=True)
''')
    binary.chmod(0o755)
    models = tmp_path / 'models'
    models.mkdir()
    context.config.openpose = OpenPoseConfig(executable=str(binary),
                                             model_folder=str(models), write_video=True)
    context.artifact('metadata').write_text(json.dumps({'width': 64, 'height': 48,
                                                       'duration_seconds': 0.4,
                                                       'streams_present': ['video', 'audio']}))
    write_table(context.artifact('frame_index'), pa.Table.from_pylist([
        {'schema_version': '1.0', 'video_id': context.video_id, 'frame_number': i,
         'pts_seconds': i * 0.04} for i in range(10)
    ]), FRAME_INDEX_SCHEMA)
    return context


def test_stage_executes_native_video_and_publishes_with_configured_ffmpeg(native_stage_context):
    from multimodal_pipeline.stages.metadata import ffprobe_streams
    context = native_stage_context
    detail = OpenPoseStage().execute(context)
    assert detail['extra']['skeleton_video'] == 'pose/skeletons.mp4'
    assert detail['extra']['render']['face_render'] == '1'
    assert detail['extra']['render']['hand_render'] == '1'
    assert {s['codec_type'] for s in ffprobe_streams(
        context.artifact('pose_skeleton_video'))['streams']} == {'video', 'audio'}
    assert OpenPoseStage().validate(context)['skeleton_video'] == 'pose/skeletons.mp4'


def test_failed_stage_mux_preserves_all_previous_pose_outputs(native_stage_context):
    from multimodal_pipeline.subprocess_utils import CommandError
    context = native_stage_context
    raw = context.artifact('pose_raw') / 'clip_000000000000_keypoints.json'
    raw.write_bytes(b'previous raw')
    for name in ('pose_body', 'pose_hands', 'pose_face', 'pose_skeleton_video'):
        context.artifact(name).write_bytes(b'previous ' + name.encode())
    paths = [raw] + [context.artifact(n) for n in
                    ('pose_body', 'pose_hands', 'pose_face', 'pose_skeleton_video')]
    before = {p: p.read_bytes() for p in paths}
    context.config.ffmpeg.executable = '/bin/false'
    with pytest.raises(CommandError):
        OpenPoseStage().execute(context)
    assert {p: p.read_bytes() for p in paths} == before


def test_lexical_symlink_alias_never_reaches_native_shell(native_stage_context, tmp_path):
    context = native_stage_context
    record = tmp_path / 'native-path.txt'
    binary = Path(context.config.openpose.executable)
    text = binary.read_text().replace("raw = Path(val('--write_json'))",
        f"Path({str(record)!r}).write_text(val('--write_video'))\nraw = Path(val('--write_json'))")
    binary.write_text(text)
    alias = tmp_path / "alias';touch SHELL_EXECUTED;#"
    alias.symlink_to(context.paths.dataset_dir, target_is_directory=True)
    context.paths.dataset_dir = alias
    OpenPoseStage().execute(context)
    sent = Path(record.read_text())
    assert "'" not in str(sent)
    assert sent == sent.resolve()


def test_truncated_native_preview_is_rejected(real_media):
    from multimodal_pipeline.skeleton_video import check_skeleton_video
    native, _ = real_media
    with pytest.raises(StageError, match='Incomplete skeleton video'):
        check_skeleton_video(native, expected_frames=100)


def test_render_cleanup_preserves_handwritten_explanation(context):
    images = context.artifact('pose_images_raw')
    guide = images / 'explanation.md'
    guide.write_text('Operator notes: preserve me')
    (images / 'old.jpg').write_bytes(b'old image')
    assert OpenPoseStage().clear_render_dir(context) == 1
    assert guide.read_text() == 'Operator notes: preserve me'


@pytest.mark.parametrize('override', ['--write_video', '--write_video=other.mp4',
                                     '--write_video_with_audio'])
def test_native_video_flags_cannot_be_overridden_outside_staging(context, override):
    context.config.openpose = OpenPoseConfig(write_video=True, extra_args=[override])
    with pytest.raises(StageError, match='extra_args'):
        OpenPoseStage().execute(context)


def test_unsafe_native_video_path_is_rejected_before_execution(context):
    context.config.openpose = OpenPoseConfig(write_video=True)
    context.paths.dataset_dir = context.paths.dataset_dir / "apostrophe's path"
    with pytest.raises(StageError, match='apostrophes'):
        OpenPoseStage().execute(context)
