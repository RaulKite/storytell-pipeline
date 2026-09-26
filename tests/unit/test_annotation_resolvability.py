"""Every annotation in the pipeline package has to actually resolve.

`from __future__ import annotations` (all 37 modules of the package use it) turns every
annotation into a string that is only evaluated if somebody asks for it. That makes a typo in
an annotation invisible to the test suite: the module imports, the stage runs, the rows are
written, and the broken name is only reachable through `typing.get_type_hints`, which nothing
in this pipeline calls.

The defect this guards is exactly that. `schemas.write_table` annotated `extra_metadata` as
`dict[str, Any] | None` while `schemas.py` imported only `Iterable, Iterator, Sequence` from
`typing`. pyflakes had flagged it as `undefined name 'Any'`, and the annotation survived anyway
— the suite was green over it, because nothing resolved annotations. `write_table` is called
with `extra_metadata=` from 21 places, so the line is load-bearing.

The scan is deliberately generic over a root package, and the second test in this file feeds it
a synthetic package whose only sin is an unresolvable function annotation. That is what keeps
the guard honest: a scan that stopped resolving objects would report no failures on real code
and would still be caught here. Measured during development — disabling the function branch
left 296 module- and class-level annotations resolving, so a bare "did the total stay big"
assertion was not enough to notice.

Importing every module of `multimodal_pipeline` is safe by design: the package is the
orchestrator side and never touches torch, pyannote, spacy or parselmouth. Heavy tools live in
`environments/*` and are reached by subprocess.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
import types
import typing

from multimodal_pipeline import __name__ as PACKAGE


def _resolve_hint(module: types.ModuleType, where: str, hint: str, out: list[str]) -> None:
    """Evaluate one string annotation in the module's own namespace."""
    try:
        eval(hint, vars(module))  # noqa: S307 - resolving the package's own annotations
    except Exception as exc:  # noqa: BLE001 - the exception IS the finding
        out.append(f"{where}: {type(exc).__name__}: {exc}")


def _scan_root(root: types.ModuleType) -> tuple[list[str], dict[str, list[str]]]:
    """Resolve every annotation defined in every module under `root`.

    Returns the failures and, per module, the exact targets that were resolved over — so
    coverage can be asserted against named objects rather than a total that can quietly shrink.
    """
    failures: list[str] = []
    scanned: dict[str, list[str]] = {}

    for module_name in sorted(
        info.name
        for info in pkgutil.walk_packages(root.__path__, prefix=f"{root.__name__}.")
        if not info.ispkg
    ):
        resolved: list[str] = []
        module = importlib.import_module(module_name)

        # Module-level variable annotations: `FOO: Bar = ...`.
        for attr, hint in list(getattr(module, "__annotations__", {}).items()):
            where = f"{module_name}:{attr}"
            resolved.append(where)
            if isinstance(hint, str):
                _resolve_hint(module, where, hint, failures)

        for attr, obj in list(vars(module).items()):
            if attr.startswith("__") or getattr(obj, "__module__", None) != module_name:
                continue

            if inspect.isfunction(obj):
                where = f"{module_name}.{attr}()"
                resolved.append(where)
                try:
                    typing.get_type_hints(obj)
                except Exception as exc:  # noqa: BLE001 - the exception IS the finding
                    failures.append(f"{where}: {type(exc).__name__}: {exc}")

            elif inspect.isclass(obj):
                for cattr, hint in list(getattr(obj, "__annotations__", {}).items()):
                    where = f"{module_name}.{attr}:{cattr}"
                    resolved.append(where)
                    if isinstance(hint, str):
                        _resolve_hint(module, where, hint, failures)

        scanned[module_name] = resolved

    return failures, scanned


#: Objects whose annotations must be resolved by the scan, named rather than counted. If the
#: scan ever stops walking functions or classes, these disappear from the report and the test
#: fails even though `failures` is empty. `write_table` is here because it is the one that was
#: actually broken.
_MUST_BE_SCANNED = (
    "multimodal_pipeline.schemas.write_table()",
    "multimodal_pipeline.state.utc_now()",
    "multimodal_pipeline.state.StageRecord:status",
    "multimodal_pipeline.fusion.TurnTableSpec:engine",
    "multimodal_pipeline.config:PERSON_TRACKER_TYPES",
)


def test_every_annotation_in_the_pipeline_package_resolves() -> None:
    package = importlib.import_module(PACKAGE)
    failures, scanned = _scan_root(package)
    assert not failures, (
        "annotations that do not resolve. With `from __future__ import annotations` these are "
        "strings that nothing evaluates at import time, so the suite stays green over them:\n"
        + "\n".join(failures)
    )
    # Coverage, asserted against named objects rather than a total that can shrink in silence.
    seen = {where for targets in scanned.values() for where in targets}
    missing = [w for w in _MUST_BE_SCANNED if w not in seen]
    assert not missing, (
        f"the scan did not resolve these known annotations: {missing}. An empty `failures` "
        f"list over {len(scanned)} modules means nothing if the walk is not reaching objects."
    )
    assert len(scanned) >= 30, (
        f"the walk found only {len(scanned)} modules under {PACKAGE}; it is no longer reaching "
        "the whole package"
    )


def test_the_scan_finds_a_broken_annotation_in_a_package_it_has_never_seen(tmp_path) -> None:
    """The guard has to be able to die, so it is run against a package that is deliberately bad.

    Two modules are written to a temp dir: one clean, one whose only defect is a function
    annotated with a name that does not exist — the `write_table` bug, reproduced from scratch.
    The scan must report exactly that one and nothing else. This is what stops the package-wide
    test above from passing by going blind.
    """
    pkg = tmp_path / "annotation_probe_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good.py").write_text(
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "COUNT: int = 3\n"
        "def ok(x: dict[str, Any] | None = None) -> int: return 0\n"
    )
    (pkg / "bad.py").write_text(
        "from __future__ import annotations\n"
        "def broken(x: MissingTypeName) -> None: return None\n"
    )

    sys.path.insert(0, str(tmp_path))
    try:
        root = importlib.import_module("annotation_probe_pkg")
        failures, scanned = _scan_root(root)
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in list(sys.modules) if n.startswith("annotation_probe_pkg")]:
            del sys.modules[name]

    assert len(failures) == 1, f"expected exactly one broken annotation, got: {failures}"
    assert "annotation_probe_pkg.bad.broken" in failures[0], failures[0]
    assert "NameError" in failures[0], failures[0]
    # The clean module was visited and produced nothing — proof the scan does not just fire
    # on everything, which would make the package-wide pass a coincidence.
    assert scanned["annotation_probe_pkg.good"] == [
        "annotation_probe_pkg.good:COUNT",
        "annotation_probe_pkg.good.ok()",
    ], scanned
