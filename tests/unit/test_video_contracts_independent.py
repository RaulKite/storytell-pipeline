"""Independent probes: video request, failure preservation and provenance."""
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pytest
from multimodal_pipeline.artifacts import ArtifactRegistry
from multimodal_pipeline.config import OpenPoseConfig
from multimodal_pipeline.schemas import FRAME_INDEX_SCHEMA, write_table
from multimodal_pipeline.stages.openpose import OpenPoseStage
from multimodal_pipeline.subprocess_utils import CommandError

@pytest.fixture
def native_stage(context, tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable')
    media = tmp_path / "source $(touch NEVER)'s clip.mp4"
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'color=blue:s=64x48:r=25:d=0.4', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(media)], check=True)
    context.source = replace(context.source, path=media)
    binary = tmp_path / 'renderer'
    binary.write_text(f'#!{sys.executable}\n' + """import json, shutil, sys
from pathlib import Path
a = sys.argv[1:]
if '--version' in a: print('test renderer'); raise SystemExit(0)
def val(k): return a[a.index(k)+1]
for i in range(10):
    (Path(val('--write_json'))/f'clip_{i:012d}_keypoints.json').write_text(json.dumps({'people': []}))
shutil.copyfile(val('--video'), val('--write_video'))
""")
    binary.chmod(0o755)
    models = tmp_path / 'models'; models.mkdir()
    context.config.openpose = OpenPoseConfig(executable=str(binary),
        model_folder=str(models), write_video=True, write_images=False)
    context.artifact('metadata').write_text(json.dumps({'width':64,'height':48,
        'duration_seconds':0.4, 'streams_present':['video']}))
    write_table(context.artifact('frame_index'), pa.Table.from_pylist([
        {'schema_version':'1.0','video_id':context.video_id,'frame_number':i,
         'pts_seconds':i*.04} for i in range(10)]), FRAME_INDEX_SCHEMA)
    return context

def hashes(ctx):
    pose = ctx.artifact('pose_raw').parent
    return {str(p.relative_to(pose)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in pose.rglob('*') if p.is_file()}

def test_mux_failure_preserves_previous_pose_dataset(native_stage):
    ctx = native_stage
    stage = OpenPoseStage()
    raw = ctx.artifact('pose_raw')/'clip_000000000000_keypoints.json'
    raw.write_text(json.dumps({'people':[{'pose_keypoints_2d':[10,20,.9]}]}))
    stage.normalize(ctx)
    ctx.artifact('pose_skeleton_video').write_bytes(b'previous preview')
    before = hashes(ctx)
    ctx.config.ffmpeg.executable = '/bin/false'
    with pytest.raises(CommandError): stage.execute(ctx)
    after = hashes(ctx)
    assert after == before

def test_video_only_provenance_matches_actual_render_flags(native_stage):
    stage = OpenPoseStage()
    detail = stage.execute(native_stage)
    argv, render = detail['command'], detail['extra']['render']
    for name in ('face_render','hand_render'):
        assert render[name] == argv[argv.index('--'+name)+1]

def test_toggle_default_registry_and_missing_video(native_stage):
    ctx = native_stage; stage = OpenPoseStage()
    assert OpenPoseConfig().write_video is False
    on = stage.config_fingerprint(ctx)
    stage.execute(ctx)
    assert stage.validate(ctx)['skeleton_video'] == 'pose/skeletons.mp4'
    assert ArtifactRegistry(ctx.paths).refresh().describe()['pose_skeleton_video']['path'] == 'pose/skeletons.mp4'
    saved = ctx.artifact('pose_skeleton_video').read_bytes()
    ctx.config.openpose.write_video = False
    assert stage.config_fingerprint(ctx) != on
    assert ctx.artifact('pose_skeleton_video').read_bytes() == saved
    ctx.config.openpose.write_video = True
    ctx.artifact('pose_skeleton_video').unlink()
    assert not stage.outputs_present(ctx)

def test_grossly_truncated_preview_is_rejected(native_stage):
    ctx = native_stage; stage = OpenPoseStage(); stage.execute(ctx)
    video = ctx.artifact('pose_skeleton_video')
    short = video.with_name('short.mp4')
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-t','0.08','-c','copy',str(short)],check=True)
    short.replace(video)
    with pytest.raises(Exception): stage.validate(ctx)

def test_apostrophe_symlink_is_canonicalized_before_upstream_shell(native_stage, tmp_path):
    ctx = native_stage
    alias = tmp_path / "alias';touch SHELL_EXECUTED;#"
    alias.symlink_to(ctx.paths.dataset_dir, target_is_directory=True)
    ctx.paths.dataset_dir = alias
    # Use the exact single-quote concatenation native videoSaver.cpp uses.
    binary = Path(ctx.config.openpose.executable)
    binary.write_text(binary.read_text()+"\nsubprocess_command = \"true '\" + val('--write_video') + \"'\"\nimport subprocess\nsubprocess.run(subprocess_command, shell=True, check=True)\nif Path('SHELL_EXECUTED').exists(): raise SystemExit('shell injection executed')\n")
    OpenPoseStage().execute(ctx)
    assert not (ctx.paths.dataset_dir/'SHELL_EXECUTED').exists()
