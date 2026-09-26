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

import functools
import importlib
import inspect
import pkgutil
import sys
import types
import typing

from multimodal_pipeline import __name__ as PACKAGE


def _resolve_hint(module: types.ModuleType, where: str, hint: str, out: list[str]) -> None:
    """Evaluate one string annotation in the module's own namespace.

    Two kinds of failure are reported and they mean different things (advisory R2-001). A
    `NameError`/`AttributeError` is the defect this file exists to find: the annotation names
    something the module cannot see. Anything else means the *harness* broke while evaluating,
    and labelling that as a broken annotation would point a reader at innocent package code. So
    the second kind is labelled as what it is rather than swallowed or presented as a finding
    about the pipeline.
    """
    try:
        eval(hint, vars(module))  # noqa: S307 - resolving the package's own annotations
    except (NameError, AttributeError) as exc:
        out.append(f"{where}: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - reported, but never as an annotation defect
        out.append(
            f"{where}: UNEXPECTED {type(exc).__name__} while resolving: {exc} "
            "(this is the guard failing, not the annotation)"
        )


def _resolve_callable(func: object, where: str, out: list[str]) -> None:
    """`get_type_hints` on a function, with the same two-kind split as `_resolve_hint`."""
    try:
        typing.get_type_hints(func)  # type: ignore[arg-type]
    except (NameError, AttributeError) as exc:
        out.append(f"{where}: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - reported, but never as an annotation defect
        out.append(
            f"{where}: UNEXPECTED {type(exc).__name__} while resolving: {exc} "
            "(this is the guard failing, not the annotation)"
        )


def _callable_targets(obj: object) -> tuple[list[object], str | None]:
    """Unwrap one class-member or module value into the callables whose annotations can resolve.

    Returns `(callables, skip_reason)`. The reason is the whole point of the function. Every
    previous version of the class branch filtered on `inspect.isfunction` and continued, and
    `vars(cls)` does not hand back functions: it hands back `classmethod`, `staticmethod`,
    `property` and `cached_property` descriptors, plus callable objects. Each filter-and-continue
    revision therefore kept a different population invisible — advisories
    R3-class-method-annotations, R3-method-descriptors and R3-property-accessor-coverage are the
    same mistake found three times. So this unwraps what it recognises and *names* what it does
    not, and the package test asserts the naming list is empty: a shape the guard learns to skip
    becomes a failure instead of coverage it never had.

    A non-callable (a field default, a constant, `_abc_impl`) is not a skip — it carries no
    annotation of its own — and reports `([], None)`.
    """
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return [obj], None
    if isinstance(obj, (classmethod, staticmethod)):
        inner = obj.__func__
        if inspect.isfunction(inner) or inspect.ismethod(inner):
            return [inner], None
        return [], f"{type(obj).__name__} wrapping a {type(inner).__name__}"
    if isinstance(obj, property):
        accessors = [acc for acc in (obj.fget, obj.fset, obj.fdel) if acc is not None]
        if not accessors:
            return [], "property with no accessors"
        unresolvable = next(
            (acc for acc in accessors if not (inspect.isfunction(acc) or inspect.ismethod(acc))),
            None,
        )
        if unresolvable is not None:
            return [], f"property accessor of type {type(unresolvable).__name__}"
        return accessors, None
    if isinstance(obj, functools.cached_property):
        inner = obj.func
        if inspect.isfunction(inner):
            return [inner], None
        return [], f"cached_property wrapping a {type(inner).__name__}"
    if callable(obj):
        return [], f"callable of type {type(obj).__name__}"
    # Non-callable: a field default, a constant, `_abc_impl`. Measured on this package, 169
    # objects reach here (53 tuple, 37 _abc_data, 32 str, 22 NoneType, 19 dict, 6 bool/int/float)
    # and *none* of them carries annotations of its own, which is what makes "nothing to resolve"
    # true rather than assumed (advisory R3-unknown-descriptor-silence). So the one case that
    # would hurt is named: an object sitting here while carrying its own annotations.
    if getattr(obj, "__dict__", {}).get("__annotations__"):
        named = sorted(getattr(obj, "__dict__")["__annotations__"])
        return [], f"non-callable of type {type(obj).__name__} carrying annotations {named[:4]}"
    return [], None


def _scan_root(
    root: types.ModuleType,
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Resolve every annotation defined in every module under `root`.

    Returns the failures, the exact targets resolved over per module (so coverage is asserted
    against named objects rather than a total that can quietly shrink), and the callables that
    were *not* resolved because the unwrapper did not recognise them. That third list is what
    makes the first two honest.
    """
    failures: list[str] = []
    scanned: dict[str, list[str]] = {}
    skipped: list[str] = []

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
                _resolve_callable(obj, where, failures)

            elif inspect.isclass(obj):
                for cattr, hint in list(getattr(obj, "__annotations__", {}).items()):
                    where = f"{module_name}.{attr}:{cattr}"
                    resolved.append(where)
                    if isinstance(hint, str):
                        _resolve_hint(module, where, hint, failures)

                # Methods, including the ones `vars(cls)` hands back as descriptors. A class's own
                # `__annotations__` covers only its attributes and `inspect.isfunction` rejects
                # every descriptor, so filtering on either alone left whole populations of methods
                # never resolved over — the same blind spot that let `write_table` ship, one level
                # down. `vars(cls)` is used rather than `inspect.getmembers` to keep inherited
                # pydantic/`BaseModel` methods out: an annotation can only resolve against the
                # module that defined it. Most of what the unwrapper rescues here are pydantic
                # validators and computed flags in `config.py`.
                for mattr, mobj in list(vars(obj).items()):
                    if mattr.startswith("__"):
                        continue
                    where = f"{module_name}.{attr}.{mattr}()"
                    callables, reason = _callable_targets(mobj)
                    if callables:
                        resolved.append(where)
                        for target in callables:
                            _resolve_callable(target, where, failures)
                    elif reason is not None:
                        skipped.append(f"{where}: {reason}")

        # Module-level callables that are not plain functions and not classes (an instance with
        # `__call__`, say). The branch above cannot see them; they are named, never skipped.
        for attr, obj in list(vars(module).items()):
            if attr.startswith("__") or getattr(obj, "__module__", None) != module_name:
                continue
            if inspect.isfunction(obj) or inspect.isclass(obj):
                continue
            _callables, reason = _callable_targets(obj)
            if reason is not None:
                skipped.append(f"{module_name}.{attr}(): {reason}")

        scanned[module_name] = resolved

    return failures, scanned, skipped


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
    # A method on a class, which the attribute-only class branch could not see.
    "multimodal_pipeline.state.StageRecord.to_dict()",
    # A classmethod descriptor, which `inspect.isfunction` rejects (R3-method-descriptors).
    "multimodal_pipeline.config.InputConfig._normalise_extensions()",
    # A property getter (there are 19; the package has no cached_property, measured).
    "multimodal_pipeline.config.OpenPoseConfig.hands_enabled()",
)


def test_every_annotation_in_the_pipeline_package_resolves() -> None:
    package = importlib.import_module(PACKAGE)
    failures, scanned, skipped = _scan_root(package)
    assert not failures, (
        "annotations that do not resolve. With `from __future__ import annotations` these are "
        "strings that nothing evaluates at import time, so the suite stays green over them:\n"
        + "\n".join(failures)
    )
    # Nothing may be passed over in silence. Several advisories were raised about populations this
    # scan could not see; asserting an empty skip list turns the *next* such shape into a failing
    # test naming the object, instead of another advisory about coverage that never existed.
    # Deliberately count-free: the last count in this file argued with its own paragraph (§36).
    assert not skipped, (
        "the scan found callables it does not know how to unwrap, so their annotations went "
        "unresolved while this file reported success:\n" + "\n".join(sorted(skipped))
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

    Three modules are written to a temp dir: one clean; one with a module-level function and a
    plain method whose annotations name things that do not exist (the `write_table` bug, rebuilt
    from scratch); one with four well-behaved method kinds, a classmethod carrying the same
    defect, a property whose accessor is a `partial`, and a non-callable object that carries
    annotations of its own. The scan must report exactly the three findings, in order, must
    *visit* the healthy methods — the part `inspect.isfunction` alone cannot see, since `vars(cls)`
    hands back descriptors — and must *name* the two shapes it cannot unwrap. This is what stops
    the package-wide test above from passing by going blind.
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
        "class AlsoBad:\n"
        "    def method(self, y: AlsoMissing) -> None: return None\n"
    )
    # `cached_property` is the other descriptor `inspect.isfunction` rejects; a property whose
    # accessor is not a function is the shape that must be *named*, not skipped.
    (
        pkg
        / "methods.py"
    ).write_text(
        "from __future__ import annotations\n"
        "from functools import cached_property, partial\n"
        "class AnnotatedDefault:\n"
        "    def __init__(self):\n"
        "        # Annotations on the object itself, not inherited from a class the scan also\n"
        "        # walks: this is the shape the non-callable branch has to name.\n"
        "        self.__annotations__ = {'depth': int}\n"
        "class Good:\n"
        "    @classmethod\n"
        "    def maker(cls, v: list[str]) -> list[str]: return v\n"
        "    @staticmethod\n"
        "    def helper(v: int) -> int: return v\n"
        "    @property\n"
        "    def ready(self) -> bool: return True\n"
        "    @cached_property\n"
        "    def loaded(self) -> str: return 'x'\n"
        "class AlsoBad:\n"
        "    @classmethod\n"
        "    def maker(cls, v: ClassMissing) -> None: return None\n"
        "class Odd:\n"
        "    weird = property(fget=partial(lambda: 1))\n"
        "    annotated_default = AnnotatedDefault()\n"
    )

    sys.path.insert(0, str(tmp_path))
    try:
        root = importlib.import_module("annotation_probe_pkg")
        failures, scanned, skipped = _scan_root(root)
    finally:
        sys.path.remove(str(tmp_path))
        for name in [n for n in list(sys.modules) if n.startswith("annotation_probe_pkg")]:
            del sys.modules[name]

    assert len(failures) == 3, f"expected three broken annotations, got: {failures}"
    assert "annotation_probe_pkg.bad.broken" in failures[0], failures[0]
    assert "NameError" in failures[0], failures[0]
    # The second finding comes from a method on a class — the branch R3-class-method-annotations
    # was raised about, so it needs its own proof of life rather than riding on the module one.
    assert "annotation_probe_pkg.bad.AlsoBad.method()" in failures[1], failures[1]
    assert "NameError" in failures[1], failures[1]
    # The third comes from a classmethod descriptor: R3-method-descriptors.
    assert "annotation_probe_pkg.methods.AlsoBad.maker()" in failures[2], failures[2]
    assert "NameError" in failures[2], failures[2]
    # Descriptors that are fine must still be *visited*, or the count above proves nothing.
    assert scanned["annotation_probe_pkg.methods"] == [
        "annotation_probe_pkg.methods.Good.maker()",
        "annotation_probe_pkg.methods.Good.helper()",
        "annotation_probe_pkg.methods.Good.ready()",
        "annotation_probe_pkg.methods.Good.loaded()",
        "annotation_probe_pkg.methods.AlsoBad.maker()",
    ], scanned["annotation_probe_pkg.methods"]
    # And the two shapes the unwrapper genuinely cannot handle are named rather than dropped: a
    # property whose accessor is not a function, and a non-callable sitting on a class while
    # carrying annotations of its own. These assertions are what make the package-wide empty skip
    # list mean something: without them, an unwrapper that classified everything as "not a
    # function" would look identical and the suite would stay green.
    assert skipped == [
        "annotation_probe_pkg.methods.Odd.weird(): property accessor of type partial",
        "annotation_probe_pkg.methods.Odd.annotated_default(): non-callable of type "
        "AnnotatedDefault carrying annotations ['depth']",
    ], skipped
    # The clean module was visited and produced nothing — proof the scan does not just fire
    # on everything, which would make the package-wide pass a coincidence.
    assert scanned["annotation_probe_pkg.good"] == [
        "annotation_probe_pkg.good:COUNT",
        "annotation_probe_pkg.good.ok()",
    ], scanned


def test_a_guard_failure_is_reported_as_the_guards_own_failure() -> None:
    """R2-001: the resolve sites used to `except Exception` and label anything that raised as a
    broken annotation. A hint that raises something other than a name lookup means the harness
    broke; the report has to say so, or a reader is sent to blame package code that is fine.
    """

    class Boom:
        def __getattr__(self, name):
            raise ZeroDivisionError("the harness broke")

    fake = types.ModuleType("fake_two")
    fake.__dict__["Boom"] = Boom()

    out: list[str] = []
    _resolve_hint(fake, "fake.two:x", "Boom.attr", out)
    assert len(out) == 1, out
    assert "UNEXPECTED ZeroDivisionError" in out[0], out[0]
    assert "the guard failing" in out[0], out[0]

    # And a genuine name failure still reads as an annotation defect, not as harness trouble.
    names: list[str] = []
    _resolve_hint(fake, "fake.two:y", "dict[str, NotInThisModule]", names)
    assert len(names) == 1 and "UNEXPECTED" not in names[0] and "NameError" in names[0], names
