#!/usr/bin/env bash
# Generate synthetic test media with ffmpeg's built-in flite speech synthesiser.
#
# No recording, no copyright, no network: the fixture is real speech-shaped audio
# that WhisperX/Parselmouth can actually analyse, generated in about a second.
#
# Usage:
#   scripts/make_fixtures.sh [output_directory]
set -euo pipefail

OUT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)/data/input_videos}"
mkdir -p "$OUT"

command -v ffmpeg >/dev/null || { echo "ffmpeg is required" >&2; exit 1; }

# English speech over a colour-bar source: exercises transcription, acoustics and
# pose detection (a synthetic source has no person, so pose yields no detections —
# which is itself a case the pipeline must handle without failing).
EN_TEXT='Hello world. This is a test of the multimodal video processing pipeline. It transcribes speech, detects speakers, and extracts acoustic and pose features.'

make_video() {
  local name="$1" fps="$2" size="$3" seconds="$4" text="$5" voice="${6:-slt}"
  local target="$OUT/$name"
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i "testsrc=size=${size}:rate=${fps}:duration=${seconds}" \
    -f lavfi -i "flite=text='${text}':voice=${voice}" \
    -map 0:v -map 1:a \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p \
    -c:a aac -b:a 128k -shortest \
    "$target"
  echo "generated $target"
}

# 29.97 fps (30000/1001): the rational-rate case that breaks naive timestamping.
make_video "pipeline_demo.mp4"        25      640x480 14 "$EN_TEXT"
make_video "pipeline_demo_ntsc.mov"   30000/1001 320x240 10 "$EN_TEXT"
# A silent recording: pose/acoustic stages must cope with no speech at all.
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "testsrc=size=320x240:rate=25:duration=4" \
  -f lavfi -i "anullsrc=r=16000:cl=mono" -t 4 \
  -map 0:v -map 1:a -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac -shortest \
  "$OUT/pipeline_silent.mp4"
echo "generated $OUT/pipeline_silent.mp4"
