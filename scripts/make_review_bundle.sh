#!/usr/bin/env bash
# Build a review bundle the operator can open in ELAN on a laptop (B5d).
#
# Why a script and not a list of commands: the bundle's value is that it is the same thing every
# time. Three facts about the .eaf make a hand-assembled bundle fail quietly, and each one is a
# `cp` that looks right:
#
#   1. the .eaf points at its media with RELATIVE_MEDIA_URL, counted from elan/annotations.eaf,
#      and the export computes that string from the real output tree it was written into. So the
#      bundle cannot choose a tidy directory name for the media: the layout has to be derived from
#      the .eaf, which is what this script does (and what it refuses to do otherwise). A bundle
#      whose layout the .eaf cannot follow opens in ELAN with an empty grid, which reads as "the
#      export is broken" when the bundle was assembled wrong.
#   2. the media URL inside the .eaf is an absolute file:// URI of the machine that built it, so
#      the .eaf is only openable through the relative path; ELAN follows RELATIVE_MEDIA_URL, and
#      that is what makes the bundle portable at all.
#   3. the internal LLM gateway hostname appears in the raw per-stage JSON dumps. A bundle leaves
#      this machine, so it is rewritten to LLM-ENDPOINT.REDACTED *inside the bundle* — the
#      pipeline's own output directory is never touched, because raw output is preserved
#      byte-identical on purpose and redacting it there would destroy the record.
#
# Usage:  scripts/make_review_bundle.sh [output-directory] [name]
# Defaults to /tmp/storytel-bundle and storytel-demo-<date>. The bundle tree and the .tgz are
# both left in the output directory; nothing under data/ is read or written.
#
#   BUNDLE_README=/path/to/README.md   copy an authored cover note into the bundle root
set -euo pipefail

OUT_ROOT="${1:-/tmp/storytel-bundle}"
NAME="${2:-storytel-demo-$(date -u +%Y-%m-%d)}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${OUT_ROOT}/${NAME}"

# The two clips the operator asked to review, when their directories are present. A list here is
# deliberate rather than "every dataset with an .eaf": data/processed also holds the synthetic
# fixtures, which are not what the B5d check is about, and a bundle that grows every time someone
# runs the pipeline is a bundle nobody downloads.
WANTED=("2017-12-30_0735_US_KABC_Jimmy_Kimmel_Live_1120_696_1124_896_hear"
        "2019-06-29_2000_ES_La-1_Telediario_1_542-550")

if [[ ! -f "${REPO}/config/config.local.yaml" ]]; then
  echo "config/config.local.yaml is required (it names the datasets and the output directory)" >&2
  exit 1
fi

DATASETS=()
for dataset in "${WANTED[@]}"; do
  if [[ -f "${REPO}/data/processed/${dataset}/elan/annotations.eaf" ]]; then
    DATASETS+=("${dataset}")
  else
    echo "  ! ${dataset} has no elan/annotations.eaf under data/processed; run the pipeline" >&2
  fi
done
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "no wanted dataset has an .eaf; nothing to bundle" >&2
  exit 1
fi

echo "Bundle ${NAME}: ${#DATASETS[@]} dataset(s)"
rm -rf "${WORK}"
mkdir -p "${WORK}/out"

for dataset in "${DATASETS[@]}"; do
  src="${REPO}/data/processed/${dataset}"
  # The whole dataset tree, not just elan/: the point of B5d is asking whether a tier is right,
  # and a reviewer who doubts a label checks it against the Parquet the label came from. This is
  # also what the first bundle contained, measured rather than assumed.
  cp -r "${src}" "${WORK}/out/${dataset}"

  # Where the .eaf says its media is, relative to itself — parsed out of the file rather than
  # assumed. The export computes RELATIVE_MEDIA_URL from the real output tree, so a bundle that
  # picks its own tidy directory name (an earlier version of this script chose `input/`) ships a
  # layout the .eaf cannot follow: ELAN opens the file, finds no media, and shows an empty grid,
  # which reads as "the export is broken" when the bundle was assembled wrong.
  media_lines="$(python3 - "${WORK}/out/${dataset}/elan/annotations.eaf" <<'PY'
import re, sys, posixpath
text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r'RELATIVE_MEDIA_URL="([^"]+)"', text)
if not match:
    sys.exit("no RELATIVE_MEDIA_URL in the .eaf: the bundle layout cannot be derived")
url = match.group(1)
parts = posixpath.normpath(url).split("/")
ups = sum(1 for part in parts if part == "..")
rest = "/".join(part for part in parts if part != "..")
if ups != 3:
    sys.exit(f"unexpected RELATIVE_MEDIA_URL depth {ups} ({url}): expected 3 "
             "(out/<dataset>/elan/), refusing to guess a layout")
print(url)     # as the .eaf wrote it, for the resolution check below
print(rest)    # where it belongs inside the bundle, for the copy
PY
)"
  media_url="$(sed -n 1p <<< "${media_lines}")"
  media_rel="$(sed -n 2p <<< "${media_lines}")"
  media_name="$(basename "${media_rel}")"
  mkdir -p "${WORK}/$(dirname "${media_rel}")"
  placed=0
  for candidate in "${src}/source/${media_name}" "${REPO}/data/input_videos/${media_name}"; do
    if [[ -f "${candidate}" ]]; then
      cp "${candidate}" "${WORK}/${media_rel}"
      placed=1
      break
    fi
  done
  if [[ "${placed}" -ne 1 ]]; then
    echo "  ! ${dataset}: media ${media_name} not found; ELAN would open this .eaf with no video" >&2
    exit 1
  fi
  # Prove it resolves from the .eaf, the way ELAN will resolve it. Without this line a wrong
  # layout is discovered by the person opening ELAN, hours later, on another machine.
  if [[ ! -f "$(realpath -m "${WORK}/out/${dataset}/elan/${media_url}")" ]]; then
    echo "  ! ${dataset}: ${media_rel} does not resolve from the .eaf; bundle layout is wrong" >&2
    exit 1
  fi
  echo "  + ${dataset} -> ${media_rel} (resolved from the .eaf)"
done

# Redact the gateway hostname inside the bundle only (fact 3 above). The pipeline's own output
# directory is never touched: raw output is preserved byte-identical on purpose.
redacted=0
while IFS= read -r -d '' file; do
  if grep -q "nienna-llm" "${file}" 2>/dev/null; then
    sed -i 's#nienna-llm\.inf\.um\.es#LLM-ENDPOINT.REDACTED#g' "${file}"
    redacted=$((redacted + 1))
  fi
done < <(find "${WORK}" -type f \( -name "*.json" -o -name "*.txt" -o -name "*.log" \) -print0)
echo "  redacted endpoint in ${redacted} file(s)"

# Views are generated, not copied: the .eaf under data/ is the record, and an HTML twin of it
# would be a second artifact to keep in step. Generated with RELATIVE paths on purpose — the page
# title prints the path it was given, and a title naming /tmp/... on the build machine is a path
# that means nothing on the laptop that opens it (the first bundle shipped exactly that).
for eaf in "${WORK}"/out/*/elan/annotations.eaf; do
  relative="${eaf#"${OUT_ROOT}"/}"
  (cd "${OUT_ROOT}" && uv run --project "${REPO}" python "${REPO}/scripts/make_elan_view.py" \
      "${relative}" "$(dirname "${relative}")/view.html" >/dev/null)
done

# The authored cover note. Optional: the bundle is readable without it, but note that `rm -rf
# "${WORK}"` above deletes whatever README.md was last placed there, so a rebuild that forgets this
# produces a bundle whose counts table is gone and whose absence nobody notices. Point it at the
# file you wrote (counts in it are measured, so it is authored per export, not kept in the repo).
if [[ -n "${BUNDLE_README:-}" ]]; then
  if [[ -f "${BUNDLE_README}" ]]; then
    cp "${BUNDLE_README}" "${WORK}/README.md"
    echo "  cover note: ${BUNDLE_README}"
  else
    echo "  ! BUNDLE_README=${BUNDLE_README} does not exist; bundle will have no README.md" >&2
  fi
fi

tar -czf "${OUT_ROOT}/${NAME}.tgz" -C "${OUT_ROOT}" "${NAME}"
echo "  tree: ${WORK}"
echo "  tgz : ${OUT_ROOT}/${NAME}.tgz"
shasum -a 256 "${OUT_ROOT}/${NAME}.tgz" 2>/dev/null || sha256sum "${OUT_ROOT}/${NAME}.tgz"
