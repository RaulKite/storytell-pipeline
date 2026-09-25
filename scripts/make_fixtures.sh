#!/usr/bin/env bash
# Generate synthetic test media with ffmpeg's built-in flite speech synthesiser.
#
# No recording, no copyright, no network: the fixture is real speech-shaped audio
# that WhisperX/Parselmouth can actually analyse, generated in about a second.
#
# Usage:
#   scripts/make_fixtures.sh [output_directory]
#   scripts/make_fixtures.sh --config config/config.local.yaml
#   scripts/make_fixtures.sh --openpose-root /srv/openpose out/
set -euo pipefail

usage() {
  # Print the script's own header comment block. Line-number ranges are how this drifted
  # before: adding a line to the header silently truncated the help text.
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
}

OUT=""
CONFIG=""
OPENPOSE_ROOT=""

while [ $# -gt 0 ]; do
  case "$1" in
    --config)       CONFIG="${2:-}"; [ -n "$CONFIG" ] || { echo "--config needs a path" >&2; exit 2; }; shift 2 ;;
    --openpose-root) OPENPOSE_ROOT="${2:-}"; [ -n "$OPENPOSE_ROOT" ] || { echo "--openpose-root needs a path" >&2; exit 2; }; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    -*)             echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)              OUT="$1"; shift ;;
  esac
done

if [ -z "$OUT" ]; then
  OUT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/data/input_videos"
fi
mkdir -p "$OUT"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }

# The whole fixture depends on the flite *filter*, which is not part of every ffmpeg
# build: it comes from libflite, and the common distribution packages ship ffmpeg
# without it. Checking for `ffmpeg` alone meant the script died inside its first
# ffmpeg call with a lavfi parse error that names neither flite nor the fix.
#
# Captured into a variable rather than piped into `grep -q`: with `set -o pipefail`,
# `ffmpeg -filters | grep -q flite` makes grep exit on its first match, ffmpeg then dies
# of SIGPIPE with 141, and pipefail reports the *whole* pipeline as failed. The result was
# a script that rejected a perfectly good ffmpeg as "no flite filter" — a false negative
# that is worse than no check at all, because it blocks a working machine.
FFMPEG_FILTERS="$(ffmpeg -hide_banner -filters 2>&1 || true)"
if ! printf '%s\n' "$FFMPEG_FILTERS" | grep -qE '(^|[[:space:]])flite([[:space:]]|$)'; then
  cat >&2 <<'DIAG'
error: this ffmpeg build has no `flite` filter, so no speech fixture can be made.

  ffmpeg -filters | grep flite     ->  (nothing)

flite comes from libflite and many packaged ffmpeg builds omit it. Install a build
that includes it, for example on Debian/Ubuntu the ffmpeg from the default
repositories, or build ffmpeg with --enable-libflite. Verify with:

  ffmpeg -filters | grep flite
DIAG
  exit 1
fi

# A missing voice fails the same way a missing filter does, and the fix is different, so
# it gets its own diagnosis rather than a lavfi stack trace. `slt` is the only voice the
# fixtures assume.
FLITE_VOICE="slt"
if ! ffmpeg -hide_banner -loglevel error -y -f lavfi -i "flite=text='probe':voice=${FLITE_VOICE}" -t 0.1 -f null - >/dev/null 2>&1; then  cat >&2 <<DIAG
error: the flite filter is present but voice '${FLITE_VOICE}' is unusable.

  ffmpeg -f lavfi -i "flite=text='probe':voice=${FLITE_VOICE}" -t 0.1 -f null -

This build's available voices are:
$(ffmpeg -hide_banner -f lavfi -i "flite=text='probe':voice=?" 2>&1 | sed 's/^/  /' || true)

Pick one of them and pass it: the voice is the last argument of make_video() in this
script, or edit FLITE_VOICE near the top.
DIAG
  exit 1
fi

# Every generated file is checked with ffprobe before the script calls itself done.
# `-shortest` makes the output as long as the *shorter* stream, so a TTS clip that came
# out shorter than asked, or an ffmpeg invocation that produced a header-only file, both
# looked like success while printing "generated ...".
require_media() {
  local path="$1" want_audio="$2"
  command -v ffprobe >/dev/null || { echo "ffprobe is required to verify fixtures" >&2; exit 1; }
  if [ ! -s "$path" ]; then
    echo "error: $path was not produced or is empty" >&2
    exit 1
  fi
  local streams duration
  streams="$(ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$path" | sort | tr '\n' ',')"
  duration="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$path" 2>/dev/null || true)"
  case "$duration" in ''|*[!0-9.]*) duration=0 ;; esac
  if [ "$(printf '%s\n' "$duration" | awk '{ printf "%d", ($1 >= 1.0) ? 1 : 0 }')" != "1" ]; then
    echo "error: $path is $(printf '%s' "$duration")s long; expected at least 1s" >&2
    exit 1
  fi
  case "$streams" in
    *video*,*audio*|*audio*,*video*) ;;
    *)
      if [ "$want_audio" = "audio" ]; then
        echo "error: $path has streams [$streams], expected a video AND an audio stream" >&2
        exit 1
      fi
      ;;
  esac
  echo "verified $path (${duration}s, $streams)"
}

# English speech over a colour-bar source: exercises transcription, acoustics and
# pose detection (a synthetic source has no person, so pose yields no detections —
# which is itself a case the pipeline must handle without failing).
EN_TEXT='Hello world. This is a test of the multimodal video processing pipeline. It transcribes speech, detects speakers, and extracts acoustic and pose features.'

make_video() {
  local name="$1" fps="$2" size="$3" seconds="$4" text="$5" voice="${6:-$FLITE_VOICE}"
  local target="$OUT/$name"
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "testsrc=size=${size}:rate=${fps}:duration=${seconds}" \
    -f lavfi -i "flite=text='${text}':voice=${voice}" \
    -map 0:v -map 1:a \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p \
    -c:a aac -b:a 128k -shortest \
    "$target"
  require_media "$target" audio
}

make_video "pipeline_demo.mp4"        25      640x480 14 "$EN_TEXT"
# 29.97 fps (30000/1001): the rational-rate case that breaks naive timestamping.
make_video "pipeline_demo_ntsc.mov"   30000/1001 320x240 10 "$EN_TEXT"
# A silent recording: pose/acoustic stages must cope with no speech at all.
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc=size=320x240:rate=25:duration=4" \
  -f lavfi -i "anullsrc=r=16000:cl=mono" -t 4 \
  -map 0:v -map 1:a -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac -shortest \
  "$OUT/pipeline_silent.mp4"
require_media "$OUT/pipeline_silent.mp4" audio

# A clip containing a real person, so the OpenPose stages have a body to find. The
# colour-bar fixtures above deliberately detect nobody. This clip ships with the
# OpenPose install and is not ours to redistribute, so it is copied on demand from
# the local install and stays out of version control (see .gitignore).
#
# WHERE THE OPENPOSE ROOT COMES FROM. This repository's one configured location is
# `openpose.root` in the pipeline YAML, and `provenance.openpose_report()` discovers the
# binary and models under it. This script used to read an `OPENPOSE_ROOT` environment
# variable instead, so a machine with OpenPose under a non-default root could process
# videos correctly while the fixture script reported the media as missing — two sources of
# truth for one path. Now: --openpose-root wins, otherwise `openpose.root` is read from
# --config, otherwise from the repository's config/config.local.yaml when present,
# otherwise from the schema default. The environment variable is gone.
schema_default_openpose_root() {
  # Ask the real model rather than hard-coding /opt/openpose: the default lives in
  # OpenPoseConfig and a second copy here would silently disagree after a change.
  python3 - <<'PY' 2>/dev/null || true
from pathlib import Path
import sys
sys.path.insert(0, str(Path("src").resolve()))
try:
    from multimodal_pipeline.config import OpenPoseConfig
    print(OpenPoseConfig().root)
except Exception:
    pass
PY
}

config_openpose_root() {
  local cfg="$1"
  [ -n "$cfg" ] && [ -f "$cfg" ] || return 0
  python3 - "$cfg" <<'PY' 2>/dev/null || true
import sys
try:
    import yaml
except ImportError:
    sys.exit(0)
try:
    data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}
except Exception:
    sys.exit(0)
root = (data.get("openpose") or {}).get("root")
if root:
    print(root)
PY
}

OPENPOSE_SOURCE="default"
if [ -n "$OPENPOSE_ROOT" ]; then
  OPENPOSE_SOURCE="--openpose-root"
else
  if [ -z "$CONFIG" ]; then
    # The repository's own local config is what the pipeline actually runs with, so it is
    # the next honest place to look. Only consulted when it exists.
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    if [ -n "$repo_root" ] && [ -f "$repo_root/config/config.local.yaml" ]; then
      CONFIG="$repo_root/config/config.local.yaml"
      OPENPOSE_SOURCE="config/config.local.yaml"
    fi
  else
    OPENPOSE_SOURCE="$CONFIG"
  fi
  OPENPOSE_ROOT="$(config_openpose_root "$CONFIG" || true)"
  if [ -z "$OPENPOSE_ROOT" ]; then
    OPENPOSE_ROOT="$(schema_default_openpose_root || true)"
    OPENPOSE_SOURCE="OpenPoseConfig default"
  fi
  [ -n "$OPENPOSE_ROOT" ] || { OPENPOSE_ROOT="/opt/openpose"; OPENPOSE_SOURCE="built-in fallback"; }
fi

OPENPOSE_MEDIA="${OPENPOSE_ROOT}/examples/media/video.avi"
if [ -f "$OPENPOSE_MEDIA" ]; then
  cp "$OPENPOSE_MEDIA" "$OUT/person_demo.avi"
  require_media "$OUT/person_demo.avi" any
  echo "copied $OUT/person_demo.avi from $OPENPOSE_MEDIA (openpose root from ${OPENPOSE_SOURCE})"
else
  # The remedy has to match where the root actually came from. Telling an operator to pass
  # --openpose-root when they just did sends them chasing the wrong setting.
  if [ "$OPENPOSE_SOURCE" = "--openpose-root" ]; then
    FIX="the path you passed to --openpose-root has no examples/media/video.avi"
  else
    FIX="pass --openpose-root, or set openpose.root in the config passed with --config"
  fi
  MESSAGE="note: $OPENPOSE_MEDIA not found (openpose root from ${OPENPOSE_ROOT}, source: ${OPENPOSE_SOURCE}). ${FIX}. The pose stages will run and detect no person, which is a supported outcome but not a useful pose smoke test."
  echo "$MESSAGE" >&2
fi
