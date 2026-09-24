"""Tests for scripts/install_spacy_models.sh.

The script exists because two plausible-looking install routes silently produced a
broken environment: it built wheel URLs from the spaCy version (models ship their own
versions, so the URL 404s), and `spacy download` printed "installation successful" for a
package the target venv could not import. Both failures are invisible: spaCy says
success, the model is missing, and linguistics quietly degrades to a blank pipeline at
run time.

Mocking the install logic would reproduce neither failure, so these run the real script
against a throwaway project root with a fake `uv` and a fake venv python. No network is
touched and the repository's real spaCy environment is never written to.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "install_spacy_models.sh"
BASH = shutil.which("bash") or "/bin/bash"

# What the fake environment reports as the installed spaCy. The probe test exists
# because the script must not derive *model* release versions from this number.
SPACY_VERSION = "3.8.16"
CANDIDATES = ("3.8.0", "3.8.1", "3.8.2")


def fake_spacy_util(other_env: Path) -> str:
    """Source for the fake spaCy.util.

    It scans the target site-packages *and* a second tree, which is what makes the
    historical false-success case testable: spaCy's own environment can list a model this
    interpreter cannot import. An empty directory is not enough to hide a package from
    ``find_spec`` -- a directory without ``__init__.py`` is a namespace package and
    imports fine -- so "spaCy lists it" and "this interpreter imports it" must be
    physically different locations, as they were in the real failure.
    """
    return f'''"""Fake spaCy.util for installer tests."""
import json
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
OTHER = Path(r"{other_env}")


def _packages():
    for root in (SITE, OTHER):
        if root.is_dir():
            yield from root.iterdir()


def get_installed_models():
    names = []
    for package in _packages():
        meta = package / "meta.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("spacy_version"):
            names.append(package.name)
    return names
'''


@pytest.fixture
def sandbox(tmp_path):
    """A project root whose spaCy environment is a controlled lie."""
    root = tmp_path / "project"
    script = root / "scripts" / "install_spacy_models.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    script.chmod(0o755)

    # The tree the *other* environment sees: an install landing here is listed by spaCy
    # and invisible to this venv's import machinery.
    other_env = tmp_path / "other_env"
    other_env.mkdir()

    venv = root / "environments" / "spacy" / ".venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    bin_dir = venv / "bin"
    bin_dir.mkdir()

    # The venv "python" is python3 in disguise, and python3 knows nothing about this
    # fabricated site-packages. Embedding PYTHONPATH in the wrapper is what keeps the
    # script's `import spacy` and `find_spec(model)` checks reading this sandbox instead
    # of the developer's real environment.
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\n"
        f'export PYTHONPATH="{site}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        'if [ "$1" = "-c" ]; then exec python3 -c "$2"; fi\n'
        'exec python3 "$@"\n',
        encoding="utf-8",
    )
    (bin_dir / "python").chmod(0o755)

    fake_spacy = site / "spacy"
    fake_spacy.mkdir()
    (fake_spacy / "__init__.py").write_text("", encoding="utf-8")
    (fake_spacy / "util.py").write_text(fake_spacy_util(other_env), encoding="utf-8")
    # importlib.metadata.version('spacy') has to answer, so claim a distribution.
    dist = site / f"spacy-{SPACY_VERSION}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: spacy\nVersion: {SPACY_VERSION}\n", encoding="utf-8"
    )

    ok = tmp_path / "fake_ok"
    ghost = tmp_path / "fake_ghost"
    ok.mkdir()
    ghost.mkdir()
    calls = tmp_path / "uv_calls.txt"

    shim = tmp_path / "shim"
    shim.mkdir()
    # Markers rather than an f-string: this is bash source, and bash's own ${var}
    # substitutions collide with format placeholders.
    (shim / "uv").write_text(
        "#!/usr/bin/env bash\n"
        'echo "$*" >> "@@CALLS@@"\n'
        'url=""\n'
        'for a in "$@"; do case "$a" in *.whl) url="$a";; esac; done\n'
        '[ -n "$url" ] || exit 0\n'
        'base="${url##*/}"\n'
        'rest="${base%-py3-none-any.whl}"\n'
        'version="${rest##*-}"\n'
        'model="${rest%-*}"\n'
        '[ -f "@@OK@@/${model}@${version}" ] || exit 1\n'
        'short="${model#*_}"\n'
        'printf \'{"name": "%s", "version": "%s", "spacy_version": ">=3.8.0,<3.9.0", '
        '"lang": "xx"}\' "$short" "$version" > "$meta"\n'
        'exit 0\n',
        encoding="utf-8",
    )
    text = (shim / "uv").read_text(encoding="utf-8")
    # A normal install lands in this venv (meta.json + an importable package). A "ghost"
    # install lands in the other environment and leaves nothing importable here: that is
    # what `spacy download`'s success banner actually produced.
    normal = text.replace(
        'printf \'{"name"',
        'pkg="@@SITE@@/${model}"\n'
        'mkdir -p "$pkg"\n'
        'meta="$pkg/meta.json"\n'
        'touch "$pkg/__init__.py"\n'
        'printf \'{"name"',
    )
    ghosted = text.replace(
        'printf \'{"name"',
        'if [ ! -f "@@GHOST@@/${model}" ]; then exit 1; fi\n'
        'pkg="@@OTHER@@/${model}"\n'
        'mkdir -p "$pkg"\n'
        'meta="$pkg/meta.json"\n'
        'printf \'{"name"',
    )
    # One binary handles both outcomes so a test only flips a marker file: the ghost shim
    # runs when a ghost marker exists for the model named in the wheel URL (the first
    # argument is `pip`, so the model has to be read off the URL).
    combined = (
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do\n'
        '  case "$a" in\n'
        '    *.whl) base="${a##*/}"; rest="${base%-py3-none-any.whl}"; '
        'model="${rest%-*}";\n'
        '      if [ -f "@@GHOST@@/${model}" ]; then exec "@@GHOSTSHIM@@" "$@"; fi ;;\n'
        '  esac\n'
        'done\n'
    )
    (shim / "uv").write_text(
        combined.replace("@@GHOST@@", str(ghost)).replace("@@GHOSTSHIM@@", str(shim / "uv-ghost"))
        + normal.replace("@@CALLS@@", str(calls))
                  .replace("@@SITE@@", str(site))
                  .replace("@@OK@@", str(ok))
                  .replace("@@GHOST@@", str(ghost))
                  .replace("@@GHOSTSHIM@@", str(shim / "uv-ghost")),
        encoding="utf-8",
    )
    (shim / "uv-ghost").write_text(
        ghosted.replace("@@CALLS@@", str(calls))
               .replace("@@OTHER@@", str(other_env))
               .replace("@@OK@@", str(ok))
               .replace("@@GHOST@@", str(ghost)),
        encoding="utf-8",
    )
    for name in ("uv", "uv-ghost"):
        (shim / name).chmod(0o755)

    return {
        "root": root,
        "site": site,
        "other_env": other_env,
        # Shim first so `uv` is the fake; the host PATH follows because the script
        # legitimately uses dirname, and a PATH with only the shim would fail on
        # coreutils rather than on anything under test.
        "path": f"{shim}:{bin_dir}:{os.environ['PATH']}",
        "calls": calls,
        "ok": ok,
        "ghost": ghost,
    }


def allow(sandbox, model: str) -> None:
    """Let every candidate version of `model` 'download' successfully."""
    for version in CANDIDATES:
        (sandbox["ok"] / f"{model}@{version}").write_text("", encoding="utf-8")


def make_ghost(sandbox, model: str) -> None:
    """Install `model` where spaCy sees it and this interpreter does not."""
    allow(sandbox, model)
    (sandbox["ghost"] / model).write_text("", encoding="utf-8")


def run_installer(sandbox, *models: str) -> subprocess.CompletedProcess[str]:
    # No PYTHONPATH here on purpose: the wrapper injects the sandbox one, and a host
    # PYTHONPATH would let the real spaCy leak into a fake-environment test.
    env = {
        "PATH": sandbox["path"],
        "HOME": str(sandbox["root"]),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    script = sandbox["root"] / "scripts" / "install_spacy_models.sh"
    return subprocess.run([BASH, str(script), *models], capture_output=True,
                          text=True, env=env, timeout=180)


def attempts(sandbox) -> list[str]:
    """Wheel versions the script asked uv to install, in request order."""
    if not sandbox["calls"].exists():
        return []
    versions = []
    for line in sandbox["calls"].read_text(encoding="utf-8").splitlines():
        wheels = [token for token in line.split() if token.endswith(".whl")]
        if wheels:
            # Strip the wheel tag first: splitting on '-' naively returns "any.whl"
            # and the assertion would be about a tag, not about the version requested.
            stem = wheels[0].removesuffix("-py3-none-any.whl")
            versions.append(stem.rsplit("-", 1)[1])
    return versions


class TestInstaller:
    def test_missing_environment_says_how_to_fix_it(self, sandbox):
        shutil.rmtree(sandbox["root"] / "environments" / "spacy" / ".venv")
        result = run_installer(sandbox, "en_core_web_lg")
        assert result.returncode == 1
        assert "uv sync" in result.stderr

    def test_probes_model_versions_instead_of_trusting_the_spacy_one(self, sandbox):
        """The bug the script was rewritten for.

        spaCy is 3.8.16 here while models ship 3.8.0. A script that derives the release
        from the spaCy version asks for 3.8.16, 404s, and then falls back to a route that
        reports success for a package nothing can import.
        """
        allow(sandbox, "es_core_news_lg")
        result = run_installer(sandbox, "es_core_news_lg")
        assert result.returncode == 0, result.stderr
        requested = attempts(sandbox)
        assert requested[0] == "3.8.0"
        assert SPACY_VERSION not in requested
        assert "3.8.0" in result.stdout

    def test_stops_probing_after_the_first_success(self, sandbox):
        allow(sandbox, "es_core_news_lg")
        run_installer(sandbox, "es_core_news_lg")
        assert attempts(sandbox) == ["3.8.0"]

    def test_falls_through_to_the_next_candidate_when_one_404s(self, sandbox):
        allow(sandbox, "es_core_news_lg")
        (sandbox["ok"] / "es_core_news_lg@3.8.0").unlink()
        result = run_installer(sandbox, "es_core_news_lg")
        assert result.returncode == 0, result.stderr
        assert attempts(sandbox)[:2] == ["3.8.0", "3.8.1"]

    def test_fails_loudly_when_no_candidate_installs(self, sandbox):
        # fake_ok stays empty: every probe 404s, which is what an unknown model does.
        result = run_installer(sandbox, "xx_bogus")
        assert result.returncode == 1
        assert "could not install xx_bogus" in result.stderr
        assert "3.8.0 3.8.1 3.8.2" in result.stderr
        assert "releases" in result.stderr

    def test_reports_an_unimportable_install_as_a_failure(self, sandbox):
        """The failure `spacy download` produced: a banner said success while this venv
        could not import the package. 'This interpreter can import it' is the only proof
        the script may accept, so a listing without an importable package is an error."""
        make_ghost(sandbox, "en_core_web_sm")
        result = run_installer(sandbox, "en_core_web_sm")
        assert result.returncode == 1
        assert "could not install" in result.stderr

    def test_rerun_is_idempotent_and_touches_nothing(self, sandbox):
        allow(sandbox, "en_core_web_lg")
        first = run_installer(sandbox, "en_core_web_lg")
        assert first.returncode == 0, first.stderr
        calls = sandbox["calls"].read_text(encoding="utf-8")
        second = run_installer(sandbox, "en_core_web_lg")
        assert second.returncode == 0, second.stderr
        assert "already installed: en_core_web_lg" in second.stdout
        assert sandbox["calls"].read_text(encoding="utf-8") == calls

    def test_default_model_is_the_cpu_one(self, sandbox):
        """No arguments must not pull a transformer model: that drags a second torch
        build into an environment that coexists with the CUDA-pinned ones."""
        allow(sandbox, "en_core_web_lg")
        result = run_installer(sandbox)
        assert result.returncode == 0, result.stderr
        assert "en_core_web_lg" in result.stdout
        assert "_trf" not in result.stdout

    def test_lists_what_the_environment_actually_has(self, sandbox):
        allow(sandbox, "es_core_news_lg")
        run_installer(sandbox, "es_core_news_lg")
        result = run_installer(sandbox, "es_core_news_lg")
        assert "Installed models:" in result.stdout
        assert "  - es_core_news_lg" in result.stdout

    def test_installs_several_models_in_one_call(self, sandbox):
        allow(sandbox, "en_core_web_lg")
        allow(sandbox, "es_core_news_lg")
        result = run_installer(sandbox, "en_core_web_lg", "es_core_news_lg")
        assert result.returncode == 0, result.stderr
        assert (sandbox["site"] / "en_core_web_lg" / "meta.json").is_file()
        assert (sandbox["site"] / "es_core_news_lg" / "meta.json").is_file()
