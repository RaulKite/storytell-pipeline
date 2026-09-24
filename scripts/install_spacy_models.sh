#!/usr/bin/env bash
# Install spaCy language models into the pipeline's isolated spaCy environment.
#
# Models are deliberately NOT declared in environments/spacy/pyproject.toml: a
# missing model must degrade linguistics quality (blank pipeline + sentencizer),
# never fail a video. Install the ones your corpus needs.
#
# Usage:
#   scripts/install_spacy_models.sh                  # English, CPU-safe default
#   scripts/install_spacy_models.sh en_core_web_trf  # transformer (pulls torch)
#   scripts/install_spacy_models.sh es_dep_news_trf de_dep_news_trf
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/environments/spacy"
PY="$ENV_DIR/.venv/bin/python"

[[ -x "$PY" ]] || { echo "spaCy environment missing. Run: (cd environments/spacy && uv sync --python 3.12)" >&2; exit 1; }

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  # en_core_web_lg is a CNN-averaged-perceptron router model: accurate and no
  # torch dependency, so it coexists with the pipeline's CUDA-pinned environments.
  MODELS=(en_core_web_lg)
fi

VERSION="$("$PY" -c "import spacy, importlib.metadata as m; print(m.version('spacy'))")"
MAJOR="${VERSION%%.*}"

installed() {
  "$PY" - "$1" <<'PY'
import sys
from spacy.util import get_installed_models
sys.exit(0 if sys.argv[1] in set(get_installed_models()) else 1)
PY
}

# A model's release version is NOT the spaCy version: es_core_news_lg ships 3.8.0
# while spaCy itself is 3.8.16, so building the URL from $VERSION 404s. Probe the
# spaCy minor line (and the one below it, since models lag spaCy by a patch or a
# minor) instead of guessing once, and install with an explicit --python: 'spacy
# download' resolves its own target and has been observed to report success while
# leaving the package out of this very venv.
CANDIDATES=("${VERSION%.*}.0" "${VERSION%.*}.1" "${VERSION%.*}.2")

target_installed() {
  "$PY" - "$1" <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

for model in "${MODELS[@]}"; do
  if installed "$model"; then
    echo "already installed: $model"
    continue
  fi
  echo "installing $model"
  for version in "${CANDIDATES[@]}"; do
    url="https://github.com/explosion/spacy-models/releases/download/${model}-${version}/${model}-${version}-py3-none-any.whl"
    if uv pip install --python "$PY" "$url" >/dev/null 2>&1; then
      echo "  installed from ${model}-${version}"
      break
    fi
  done
  # The authoritative check is that THIS interpreter can import it. spaCy's own
  # success banner is not evidence: it has printed one for a package this venv
  # cannot see.
  target_installed "$model" || {
    echo "could not install $model into $ENV_DIR/.venv" >&2
    echo "  tried: ${CANDIDATES[*]}" >&2
    echo "  spaCy publishes model releases at https://github.com/explosion/spacy-models/releases" >&2
    exit 1
  }
  installed "$model" || { echo "$model imports but spaCy does not list it" >&2; exit 1; }
  echo "installed $model (spacy $MAJOR)"
done

echo
echo "Installed models:"
"$PY" - <<'PY'
from spacy.util import get_installed_models
for name in sorted(get_installed_models()):
    print("  -", name)
PY
