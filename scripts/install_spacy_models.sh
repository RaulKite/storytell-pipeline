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

for model in "${MODELS[@]}"; do
  if installed "$model"; then
    echo "already installed: $model"
    continue
  fi
  url="https://github.com/explosion/spacy-models/releases/download/${model}-${VERSION}/${model}-${VERSION}-py3-none-any.whl"
  echo "installing $model ($VERSION)"
  if ! uv pip install --python "$PY" "$url"; then
    # spaCy's own installer knows the right URL for every published model/version.
    echo "wheel URL unavailable, falling back to 'python -m spacy download'"
    "$PY" -m spacy download "$model"
  fi
  installed "$model" || { echo "install reported success but $model is not importable" >&2; exit 1; }
  echo "installed $model (spacy $MAJOR)"
done

echo
echo "Installed models:"
"$PY" - <<'PY'
from spacy.util import get_installed_models
for name in sorted(get_installed_models()):
    print("  -", name)
PY
