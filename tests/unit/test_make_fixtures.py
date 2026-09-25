"""Tests for scripts/make_fixtures.sh.

The script is the first thing a fresh clone runs, and two of its failure modes were
silent enough to survive review:

* it checked for `ffmpeg` but depended on the `flite` *filter*, which many packaged
  ffmpeg builds omit. The failure was a lavfi parse error naming neither flite nor the
  fix, from inside the first ffmpeg call.
* it located OpenPose through an `OPENPOSE_ROOT` environment variable while the pipeline
  itself uses `openpose.root` from the config. A machine with OpenPose anywhere but
  /opt/openpose got "media not found" from the fixture script while processing videos
  with OpenPose perfectly well.

Both are tested against a shim ffmpeg rather than a mock, because the defects live in
shell control flow and exit codes. The first version of the flite gate had its own bug,
found only by running it: `ffmpeg -filters | grep -q flite` under `set -o pipefail` makes
grep exit on the first match, ffmpeg dies of SIGPIPE with 141, pipefail reports the
pipeline as failed, and a *working* ffmpeg was rejected as having no flite. `TestFliteGate`
keeps both directions pinned: rejects a build without the filter, and accepts one with it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_fixtures.sh"
REPO_ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "/bin/bash"


def ffmpeg_shim(tmp_path: Path, *, flite: bool, breaks: bool = False,
                truncated: bool = False) -> Path:
    """A stand-in ffmpeg whose only behaviours are the ones under test.

    `flite=False` reproduces a build without the filter. `breaks=True` makes the actual
    synthesis call fail, standing in for any ffmpeg error after the gate. `truncated=True`
    produces a file that exists but is not probeable media, which is what a silent failure
    used to print "generated" for.
    """
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    filters_block = " ... flite             |->A       Synthesize voice from text\n" if flite else ""
    script = f"""#!/usr/bin/env bash
# ffmpeg shim for make_fixtures.sh tests.
args="$*"
case "$args" in
  *-filters*)
    printf 'ffmpeg version shim\\n'
    printf '%s' '{filters_block}'
    exit 0 ;;
esac
case "$args" in
  # The voice probe is the only call that throws its output away. The pattern has to be
  # `"null -"`, not `null`: the silent-fixture call contains `anullsrc`, and a looser
  # pattern swallowed it and produced no file at all.
  *"null -"*)
    case "$args" in
      *"voice=slt"*) exit {'0' if flite else '234'} ;;
      *) exit 0 ;;
    esac ;;
esac
case "$args" in
  *breaks*) exit 1 ;;
esac
# real invocation: find the output path (last argument) and write something.
out="${{@: -1}}"
mkdir -p "$(dirname "$out")"
if [ '{'yes' if truncated else 'no'}' = yes ]; then
  printf 'not media at all' > "$out"
else
  # A real encoder is not needed: ffprobe is shimmed too, see ffprobe_shim().
  printf 'stub-media' > "$out"
fi
exit 0
"""
    path = shim / "ffmpeg"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return shim


def ffprobe_shim(tmp_path: Path, *, durations: dict[str, str] | None = None,
                 streams: str = "audio,video,") -> Path:
    """A ffprobe that reports whatever the test needs about the files it is shown.

    The real script now verifies every fixture, so the tests need a probe that can agree
    ("this is a 10s av file") or disagree ("this file is 0.0s long") about the same bytes.
    """
    shim = tmp_path / "shim"
    shim.mkdir(exist_ok=True)
    # Matched by suffix: the script hands ffprobe an absolute path, and a pattern that
    # only matched a bare filename silently never fired (the test then asserted about a
    # duration the shim never actually reported).
    table = ""
    for name, seconds in (durations or {}).items():
        table += f'    *{name}) echo "{seconds}" ;;\n'
    script = f"""#!/usr/bin/env bash
args="$*"
out=""
for a in "$@"; do out="$a"; done
case "$args" in
  *codec_type*) printf '%s\\n' '{streams}' ;;
  *duration*)
    case "$out" in
{table}      *) echo "9.985000" ;;
    esac ;;
esac
exit 0
"""
    path = shim / "ffprobe"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def run(tmp_path: Path, *argv: str, ffmpeg_dir: Path, ffprobe_dir: Path | None = None,
        extra_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    path_parts = [str(ffmpeg_dir)]
    if ffprobe_dir is not None:
        path_parts.append(str(ffprobe_dir))
    if extra_path is not None:
        path_parts.append(str(extra_path))
    path_parts.append(os.environ["PATH"])
    env = {
        "PATH": ":".join(path_parts),
        "HOME": str(tmp_path),
        # No OPENPOSE_ROOT anywhere: the variable is supposed to be gone, and inheriting
        # one from the developer's shell would silently make a test pass for the wrong
        # reason.
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run([BASH, str(SCRIPT), *argv], capture_output=True, text=True,
                          env=env, cwd=str(tmp_path), timeout=180)


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "cwd"
    root.mkdir()
    return root


class TestFliteGate:
    def test_rejects_a_build_without_the_flite_filter_and_names_the_fix(self, sandbox, tmp_path):
        shim = ffmpeg_shim(tmp_path, flite=False)
        result = run(sandbox, str(sandbox / "out"), ffmpeg_dir=shim,
                     ffprobe_dir=ffprobe_shim(tmp_path))
        assert result.returncode == 1
        assert "no `flite` filter" in result.stderr
        # Actionable: it has to say what to install and how to check.
        assert "libflite" in result.stderr
        assert "ffmpeg -filters | grep flite" in result.stderr
        # And it must fail before writing anything, not half-generate a corpus.
        assert not (sandbox / "out" / "pipeline_demo.mp4").exists()

    def test_accepts_a_build_that_has_the_filter(self, sandbox, tmp_path):
        """The false-negative guard.

        The first implementation rejected a *good* ffmpeg because `grep -q` closed the pipe
        early and pipefail turned ffmpeg's SIGPIPE into a failed gate. A gate that blocks
        working machines is worse than the silent failure it replaced.
        """
        shim = ffmpeg_shim(tmp_path, flite=True)
        result = run(sandbox, str(sandbox / "out"), ffmpeg_dir=shim,
                     ffprobe_dir=ffprobe_shim(tmp_path))
        assert "no `flite` filter" not in result.stderr
        assert result.returncode == 0, result.stderr
        assert (sandbox / "out" / "pipeline_demo.mp4").is_file()

    def test_still_checks_for_ffmpeg_itself(self, sandbox, tmp_path):
        """A PATH with no ffmpeg at all must produce the named diagnostic, not a crash.

        Built as an allowlist of the utilities the script reaches before its ffmpeg
        check (`mkdir`, and the shell builtins `command`/`printf`) rather than by trying
        to subtract ffmpeg from the host PATH, which is not a thing one can do.
        """
        minimal = tmp_path / "minimalbin"
        minimal.mkdir()
        for tool in ("mkdir", "dirname", "grep", "sed", "awk", "cat"):
            source = shutil.which(tool)
            assert source, f"host is missing {tool}"
            (minimal / tool).symlink_to(source)
        result = subprocess.run(
            [BASH, str(SCRIPT), str(sandbox / "out")], capture_output=True, text=True,
            env={"PATH": str(minimal), "HOME": str(tmp_path)}, timeout=60)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "ffmpeg is required" in result.stderr


class TestOutputVerification:
    def test_reports_every_fixture_with_its_measured_length(self, sandbox, tmp_path):
        """--openpose-root points at nothing so the count is 3 on every machine.

        Without it this test passes here (the host really has /opt/openpose, so the
        person clip is copied and counted) and fails on a clean machine — a test that
        encodes one machine's filesystem.
        """
        shim = ffmpeg_shim(tmp_path, flite=True)
        result = run(sandbox, "--openpose-root", str(tmp_path / "no-openpose"),
                     str(sandbox / "out"), ffmpeg_dir=shim,
                     ffprobe_dir=ffprobe_shim(tmp_path))
        assert result.returncode == 0, result.stderr
        assert result.stdout.count("verified ") == 3, result.stdout
        assert "9.985000s" in result.stdout

    def test_a_file_that_is_not_media_fails_the_run(self, sandbox, tmp_path):
        """`generated ...` used to be printed unconditionally.

        An ffmpeg that exits 0 while producing something unprobeable is now an error
        rather than a success message, so a broken corpus cannot be handed to a batch run.
        """
        shim = ffmpeg_shim(tmp_path, flite=True, truncated=True)
        result = run(sandbox, "--openpose-root", str(tmp_path / "no-openpose"),
                     str(sandbox / "out"), ffmpeg_dir=shim,
                     ffprobe_dir=ffprobe_shim(tmp_path,
                                              durations={"pipeline_demo.mp4": "0.000000"}))
        assert result.returncode == 1
        assert "expected at least 1s" in result.stderr

    def test_a_video_without_an_audio_stream_is_rejected(self, sandbox, tmp_path):
        """A speech fixture with no audio track is useless and would fail later, obscurely."""
        shim = ffmpeg_shim(tmp_path, flite=True)
        result = run(sandbox, str(sandbox / "out"), ffmpeg_dir=shim,
                     ffprobe_dir=ffprobe_shim(tmp_path, streams="video,"))
        assert result.returncode == 1
        assert "expected a video AND an audio stream" in result.stderr


class TestOpenPoseRootResolution:
    """One source of truth: --openpose-root, else openpose.root, else the schema default."""

    def write_config(self, path: Path, root: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"openpose:\n  root: {root}\n", encoding="utf-8")

    def make_openpose_media(self, root: Path) -> Path:
        media = root / "examples" / "media"
        media.mkdir(parents=True)
        (media / "video.avi").write_bytes(b"stub-avi-bytes")
        return root

    def test_explicit_flag_wins_and_names_its_source(self, sandbox, tmp_path):
        shim = ffmpeg_shim(tmp_path, flite=True)
        root = self.make_openpose_media(tmp_path / "custom-openpose")
        cfg = tmp_path / "other.yaml"
        self.write_config(cfg, "/should/not/be/used")
        result = run(sandbox, "--openpose-root", str(root), str(sandbox / "out"),
                     ffmpeg_dir=shim, ffprobe_dir=ffprobe_shim(tmp_path))
        assert result.returncode == 0, result.stderr
        assert (sandbox / "out" / "person_demo.avi").is_file()
        assert "openpose root from --openpose-root" in result.stdout

    def test_config_root_is_used_when_no_flag_is_given(self, sandbox, tmp_path):
        shim = ffmpeg_shim(tmp_path, flite=True)
        root = self.make_openpose_media(tmp_path / "from-config")
        cfg = tmp_path / "config.yaml"
        self.write_config(cfg, str(root))
        result = run(sandbox, "--config", str(cfg), str(sandbox / "out"),
                     ffmpeg_dir=shim, ffprobe_dir=ffprobe_shim(tmp_path))
        assert result.returncode == 0, result.stderr
        assert (sandbox / "out" / "person_demo.avi").is_file()
        assert str(cfg) in result.stdout

    def test_the_environment_variable_is_gone(self, sandbox, tmp_path):
        """OPENPOSE_ROOT used to be read here, disagreeing with openpose.root.

        A machine with OpenPose under a non-default root got "media not found" from this
        script while the pipeline used that same root happily. Two names for one setting is
        the bug; this test fails if either name comes back.
        """
        shim = ffmpeg_shim(tmp_path, flite=True)
        root = self.make_openpose_media(tmp_path / "env-root")
        env = dict(os.environ)
        env["OPENPOSE_ROOT"] = str(root)
        env["HOME"] = str(tmp_path)
        # The ffprobe shim goes first: this is the shimmed-ffmpeg path, so the fixtures
        # are stub bytes and the host's real ffprobe would (correctly) reject them.
        env["PATH"] = f"{shim}:{ffprobe_shim(tmp_path)}:{os.environ['PATH']}"
        result = subprocess.run(
            [BASH, str(SCRIPT), str(sandbox / "out")], capture_output=True, text=True,
            env=env, cwd=str(tmp_path), timeout=180)
        assert result.returncode == 0, result.stderr
        combined = result.stdout + result.stderr
        assert str(root) not in combined, "OPENPOSE_ROOT was honoured again"
        if (sandbox / "out" / "person_demo.avi").exists():
            # Possible on a host that really has OpenPose at the schema default. The
            # claim is only about provenance: the clip came from the config/default root,
            # never from the variable. Asserting "no person clip" here would encode this
            # machine's filesystem into the suite.
            assert "examples/media/video.avi" in result.stdout
            assert "openpose root from" in result.stdout

    def test_schema_default_is_read_from_the_code_not_hardcoded(self, sandbox, tmp_path):
        """The fallback must come from OpenPoseConfig, not a second literal /opt/openpose."""
        source = SCRIPT.read_text(encoding="utf-8")
        assert "OpenPoseConfig" in source, "fallback no longer asks the real config model"

    def test_missing_media_still_warns_but_does_not_fail(self, sandbox, tmp_path):
        """Colour-bar fixtures are a valid corpus; a person clip is a nice-to-have."""
        shim = ffmpeg_shim(tmp_path, flite=True)
        result = run(sandbox, "--openpose-root", str(tmp_path / "nowhere"),
                     str(sandbox / "out"), ffmpeg_dir=shim,
                     ffprobe_dir=ffprobe_shim(tmp_path))
        assert result.returncode == 0, result.stderr
        assert "not a useful pose smoke test" in result.stderr
        assert (sandbox / "out" / "pipeline_demo.mp4").is_file()

    def test_the_remedy_matches_where_the_root_actually_came_from(self, sandbox, tmp_path):
        """The first version told an operator to pass --openpose-root when they just did.

        The note also has to name the root it used, or "not found" is unactionable on a
        machine where OpenPose is installed somewhere non-default.
        """
        shim = ffmpeg_shim(tmp_path, flite=True)
        missing = tmp_path / "nowhere"
        result = run(sandbox, "--openpose-root", str(missing), str(sandbox / "out"),
                     ffmpeg_dir=shim, ffprobe_dir=ffprobe_shim(tmp_path))
        assert str(missing) in result.stderr
        assert "source: --openpose-root" in result.stderr
        assert "pass --openpose-root" not in result.stderr, "advised a flag already passed"

        cfg = tmp_path / "c.yaml"
        cfg.write_text(f"openpose:\n  root: {tmp_path / 'cfg-missing'}\n", encoding="utf-8")
        second = run(sandbox, "--config", str(cfg), str(sandbox / "out"), ffmpeg_dir=shim,
                     ffprobe_dir=ffprobe_shim(tmp_path))
        assert f"source: {cfg}" in second.stderr
        assert "pass --openpose-root" in second.stderr


class TestArgumentHandling:
    def test_help_lists_the_flags(self, sandbox, tmp_path):
        shim = ffmpeg_shim(tmp_path, flite=True)
        result = run(sandbox, "--help", ffmpeg_dir=shim, ffprobe_dir=ffprobe_shim(tmp_path))
        assert result.returncode == 0
        assert "--openpose-root" in result.stdout
        assert "--config" in result.stdout

    def test_an_unknown_option_is_refused_instead_of_generating_anyway(self, sandbox, tmp_path):
        shim = ffmpeg_shim(tmp_path, flite=True)
        result = run(sandbox, "--openpose", str(tmp_path), str(sandbox / "out"),
                     ffmpeg_dir=shim, ffprobe_dir=ffprobe_shim(tmp_path))
        assert result.returncode == 2
        assert "unknown option" in result.stderr

    def test_a_flag_without_its_value_is_refused(self, sandbox, tmp_path):
        shim = ffmpeg_shim(tmp_path, flite=True)
        result = run(sandbox, "--openpose-root", ffmpeg_dir=shim,
                     ffprobe_dir=ffprobe_shim(tmp_path))
        assert result.returncode == 2
        assert "needs a path" in result.stderr


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
class TestAgainstRealFfmpeg:
    """The shim proves control flow; this proves the fixtures are still real media.

    Kept separate from the shimmed tests because it writes to a temporary directory and
    takes about a second, and because a shim can never catch the case where ffmpeg's own
    argument syntax changed under us.
    """

    def test_produces_probeable_speech_fixtures(self, sandbox, tmp_path):
        real_bin = Path(shutil.which("ffmpeg")).parent
        out = tmp_path / "real-out"
        result = subprocess.run([BASH, str(SCRIPT), "--openpose-root", str(tmp_path / "none"),
                                 str(out)], capture_output=True, text=True,
                                env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
                                cwd=str(REPO_ROOT), timeout=300)
        assert result.returncode == 0, result.stderr
        for name in ("pipeline_demo.mp4", "pipeline_demo_ntsc.mov", "pipeline_silent.mp4"):
            path = out / name
            assert path.is_file(), f"{name} missing"
            duration = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                check=True).stdout.strip())
            assert duration >= 1.0, f"{name} is only {duration}s"
            streams = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                 "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                check=True).stdout
            assert "audio" in streams and "video" in streams, f"{name}: {streams}"

    def test_the_speech_fixture_actually_contains_speech(self, sandbox, tmp_path):
        """A silent 'speech' fixture would pass every structural check above.

        Mean absolute audio level is the cheapest honest signal that flite synthesised
        something. pipeline_silent.mp4 is the control: it must measure near zero.
        """
        out = tmp_path / "real-out2"
        subprocess.run([BASH, str(SCRIPT), "--openpose-root", str(tmp_path / "none"), str(out)],
                       capture_output=True, text=True, check=True,
                       env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
                       cwd=str(REPO_ROOT), timeout=300)

        def volume(path: Path) -> float:
            report = subprocess.run(
                ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                capture_output=True, text=True, check=True).stderr
            # Line looks like: "[Parsed_volumedetect_0 @ 0x55f..] mean_volume: -24.3 dB"
            for line in report.splitlines():
                if "mean_volume" in line:
                    return float(line.split("mean_volume:")[1].strip().replace(" dB", ""))
            raise AssertionError(f"no mean_volume for {path.name}: {report[-400:]}")

        speech = volume(out / "pipeline_demo.mp4")
        silent = volume(out / "pipeline_silent.mp4")
        assert speech > -60, f"speech fixture measured {speech} dB - is flite silent here?"
        assert silent < -60, f"'silent' fixture measured {silent} dB - it is not silent"
