"""``elan``: export the whole dataset as one ELAN ``.eaf`` (T22).

A pure-python stage — no subprocess, no new uv environment, nothing heavy. It reads Parquet
tables that already exist and writes one XML file beside them, so changing a tier's text costs
a re-export (seconds) rather than another GPU pass, and the corpus is untouched.

Why it is the **last** stage. Every tier summarises some producer's table, so an export that ran
mid-pipeline would produce a file missing a tier for a reason that has nothing to do with the
video — and this is the artifact an analyst opens first, where a missing tier reads as "this
clip has no person data". The dependency is on ``finalization`` alone for the complementary
reason recorded in ``STAGE_DEPENDENCIES``: the manifest has to be able to declare this stage, so
this stage cannot be upstream of the stage that writes the manifest.

Why it defaults to **on** when ``persons`` and ``diarization_nemotron`` default to off: those two
cost a torch environment and a checkpoint download a fresh clone never asked for. This one costs
a pure-Python library and a few seconds over files already on disk. See
:class:`~multimodal_pipeline.config.ElanConfig`.

The table-to-tier mapping lives in ``elan.py``, so the tier algebra can be driven against a
synthetic dataset directory with no stage, no config and no output directory. This file is the
reader, the writer, and the reuse guarantees around them:

* per-tier absence is normal and logged, never an error — translation has no endpoint by
  default, ``persons`` ships disabled, either diarizer may be off;
* a dataset with *neither* transcript table is a different case and is refused, because an .eaf
  with no words and no segments tier opens to an empty grid and reads as a broken export rather
  than an unfinished dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..elan import TIERS, build_eaf, tier_counts
from ..exceptions import ValidationError
from .base import Stage, StageContext

#: The artifacts whose joint absence makes the export pointless rather than partial. Every
#: other input is optional; these two are the tiers everyone opens the file for.
TRANSCRIPT_ARTIFACTS = ("speech_words", "speech_segments")


class ElanStage(Stage):
    """Write one ELAN ``.eaf`` per video from the tables every other stage produced."""

    name = "elan"
    # Every table the export reads, in the order `elan.TIERS` declares them — verified to exist
    # by importing the registry, never invented (test_declared_inputs_are_registered_artifacts).
    # Declared for the same reason every stage declares inputs: the state file and
    # ``status --plan`` should show what a stage consumes. They are read *softly*:
    # ``ctx.input()`` raises on a missing artifact, and an optional producer that never ran is a
    # supported state here rather than an error. ``build_eaf`` therefore resolves its own paths
    # from the dataset directory and skips a tier whose file is absent.
    inputs = tuple(spec.artifact for spec in TIERS)
    outputs = ("elan_annotations",)
    config_keys = ("elan",)

    # ------------------------------------------------------------------ fingerprint

    def config_fingerprint(self, ctx: StageContext) -> dict[str, Any]:
        """The config, the bytes of every input table, and the code that reads them.

        The input digests are here because this export is a function of those bytes and of
        nothing else the dependency hash can reach: a table replaced by hand parses, has rows,
        and would otherwise be exported as whatever it now says while the recorded
        ``config_hash`` still matched. An absent input is recorded as ``None`` rather than
        dropped from the payload, so a table that appears later changes the fingerprint — a key
        that vanished would hash the same as some other combination of missing inputs.
        ``speaker_fusion`` and ``pose_normalized`` make the same argument for their inputs; this
        stage simply has more of them.

        ``_python_code_sha256`` is the third thing the output is a function of, and the one no
        dependency hash can see: there is no worker, so without it a change to the
        millisecond rounding or the block collapsing in ``elan.py`` would leave the config,
        every threshold and every input digest identical — the stage would look reusable and the
        fix would silently do nothing until someone deleted the .eaf by hand. Both modules are
        named because half the tier text is assembled here and half in ``elan.py``. See
        :func:`~multimodal_pipeline.stages.base.python_source_digest`.
        """
        from .. import elan as elan_core
        from ..stages.metadata import sha256_of
        from . import elan as elan_stage
        from .base import python_source_digest

        cfg = ctx.config.elan
        payload: dict[str, Any] = {
            "stage": self.name,
            "enabled": cfg.enabled,
            # The tier set is part of the contract, not an implementation detail: renaming or
            # dropping a tier changes every file, and these names are what a reader expects to
            # see when they open ELAN.
            "tiers": [spec.tier for spec in TIERS],
            "_python_code_sha256": python_source_digest(elan_core, elan_stage),
        }
        for spec in TIERS:
            path = ctx.artifact(spec.artifact)
            # Memoised like every other derived stage's digest, because the fingerprint is
            # computed on `status --plan` as well as before a run.
            cache_key = f"elan_digest:{spec.artifact}"
            if cache_key not in ctx.scratch:
                ctx.scratch[cache_key] = sha256_of(path) if path.is_file() else None
            payload[f"{spec.tier}_digest"] = ctx.scratch[cache_key]
        return payload

    # ------------------------------------------------------------------ enablement

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        """Skip with a reason when there is no transcript to align anything against.

        ``words`` and ``segments`` are the two tiers everyone opens the file for; with both
        absent there is nothing to put the visual and acoustic tiers next to, and the honest
        outcome is a skip naming why rather than an .eaf that opens to an empty grid. Every
        other table is optional: a dataset with a transcript and no pose data is a normal
        dataset, and it gets a .eaf with exactly the tiers its producers wrote.

        Decided from the filesystem rather than from ``whisperx.enabled`` alone, for the reason
        ``pose_normalized`` gives for the same choice: a dataset whose transcript was deleted is
        described accurately instead of excused by a flag.
        """
        cfg = ctx.config.elan
        if not cfg.enabled:
            return False, "elan.enabled = false"
        if any(ctx.artifact(name).is_file() for name in TRANSCRIPT_ARTIFACTS):
            return True, ""
        return False, ("no transcript to export: neither speech/words.parquet nor "
                       "speech/segments.parquet exists in this dataset — run the whisperx "
                       "stage first")

    # ------------------------------------------------------------------ execution

    def execute(self, ctx: StageContext) -> dict[str, Any]:
        """Build the .eaf and publish it atomically.

        The two refusals below are the only ways this stage fails a video, and both are about
        the file being *unusable* rather than incomplete: no transcript at all (see
        :meth:`enabled`, re-checked because the two calls are not atomic and a file that
        vanished between them is worth failing loudly for), or a source video that is not on
        disk (the media descriptor is the one part of the export nothing can recover later).
        """
        missing = [name for name in TRANSCRIPT_ARTIFACTS if not ctx.artifact(name).is_file()]
        if len(missing) == len(TRANSCRIPT_ARTIFACTS):
            absent = ", ".join(str(ctx.artifact(name)) for name in TRANSCRIPT_ARTIFACTS)
            raise ValidationError(self.name, [
                f"no transcript to export: {absent} are both absent — run the whisperx stage "
                "first; an .eaf with neither tier opens to an empty grid and reads as a broken "
                "export rather than an unfinished dataset"])

        video_path = Path(ctx.source.path)
        if not video_path.is_file():
            raise ValidationError(self.name,
                                  [f"source video not found for the media descriptor: "
                                   f"{video_path} — the .eaf would link a file that is not there"])

        dataset_dir = ctx.paths.dataset_dir
        eaf = build_eaf(dataset_dir, video_path, log=ctx.log)
        counts = tier_counts(eaf)
        descriptor = eaf.media_descriptors[0] if eaf.media_descriptors else {}
        summary: dict[str, Any] = {
            "tiers": len(counts),
            "annotations": sum(counts.values()),
            "tier_counts": dict(sorted(counts.items())),
            # Named in the record, not only in the log: "this dataset has no person tier" should
            # be answerable from status.json without opening the .eaf.
            "skipped_tiers": [spec.tier for spec in TIERS if spec.tier not in counts],
            "media_url": descriptor.get("MEDIA_URL"),
            "relative_media_url": descriptor.get("RELATIVE_MEDIA_URL"),
            "mimetype": descriptor.get("MIME_TYPE"),
        }
        path = self._write_atomically(ctx.artifact("elan_annotations"), eaf)
        summary["bytes"] = path.stat().st_size
        ctx.scratch["elan"] = summary
        ctx.log(f"elan: {summary['tiers']} tier(s), {summary['annotations']} annotation(s), "
                f"{summary['bytes']} bytes -> "
                f"{path.relative_to(dataset_dir).as_posix()}")
        return {"tool_version": _pympi_version(), "model_version": None, "extra": summary}

    @staticmethod
    def _write_atomically(path: Path, eaf: Any) -> Path:
        """Publish via a sibling temp file plus ``os.replace``.

        ``atomic_write_text`` is unusable here: it takes a string, while ``Eaf`` writes its own
        bytes — so the library writes the temp file and this function performs the replace. The
        reason for the dance is the one recorded in ``artifacts.py``: a crash or a full disk
        mid-write would otherwise leave a truncated .eaf that ``outputs_present`` reports as a
        finished export, because "the file exists" is that check's entire question.
        ``os.replace`` is atomic within a filesystem, which is why the temp file's name comes
        from the target directory rather than the OS temp directory.

        The reserve-then-unlock step is not tidiness: ``pympi.Elan.to_eaf`` renames any file that
        already occupies its destination to ``<name>.bak`` before writing. Handing it a name
        ``NamedTemporaryFile`` has already created therefore litters every dataset's ``elan/``
        directory with a zero-byte ``.bak`` sibling of the artifact — which the registry would
        then count inside the ``elan_annotations`` slot's file list. Reserving the name and
        releasing it means the destination never exists when the library looks, so no backup is
        made and nothing is left behind on the success path or the failure path.
        """
        import os
        import tempfile

        path.parent.mkdir(parents=True, exist_ok=True)
        reserved = tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.",
                                               suffix=".tmp", delete=False)
        temp_path = Path(reserved.name)
        reserved.close()
        temp_path.unlink()
        try:
            eaf.to_file(str(temp_path))
            with temp_path.open("rb") as handle:  # durable before it becomes the artifact
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
        return path

    # ------------------------------------------------------------------ validation

    def validate(self, ctx: StageContext) -> dict[str, Any]:
        """The file exists, parses, and still says what this stage wrote it as saying.

        Deliberately *not* a re-derivation of the tier set from the tables on disk. Comparing
        the file's tiers against what is present now would make a tier whose build failed
        permanently rerunnable — every run would skip the same tier, ``validate`` would complain
        again, and the stage would never settle. Content staleness is covered instead by the
        per-table digests in the fingerprint and by the dependency hash, so an upstream rerun
        does send this stage back through ``execute``.

        What is checked is openability and self-consistency, the two things a reader notices and
        the file cannot report from outside: ELAN refusing to parse it, a linked video it cannot
        find, a tier lost to a truncation that left well-formed XML behind.
        """
        path = ctx.artifact("elan_annotations")
        if not path.is_file():
            raise ValidationError(self.name, [f"ELAN export missing ({path.name})"])
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValidationError(self.name, [f"{path.name} unreadable: {exc}"]) from exc

        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            # Naming a truncated write is the point: it is what a crash leaves, and "not valid
            # XML" sends a reader to a schema where this sends them to the rerun.
            raise ValidationError(self.name,
                                  [f"{path.name} is not well-formed XML ({exc}); rerun elan"])
        if not root.tag.endswith("ANNOTATION_DOCUMENT"):
            raise ValidationError(self.name,
                                  [f"{path.name} root element is {root.tag!r}, expected "
                                   "ANNOTATION_DOCUMENT"])

        issues: list[str] = []
        # pympi writes a `default` TIER into every document it emits (its own implicit tier,
        # which nothing here ever annotates), so it is excluded the same way tier_counts
        # excludes it. Counting it would report six tiers for five, and leaving it in the list
        # would make "declares no tier" unreachable — the check could never fire, because a
        # document with no real tier still contains `default`.
        tiers = [element.attrib.get("TIER_ID") for element in root.iter("TIER")
                 if element.attrib.get("TIER_ID") != "default"]
        if not tiers:
            issues.append(f"{path.name} declares no tier")
        descriptors = list(root.iter("MEDIA_DESCRIPTOR"))
        if not descriptors:
            issues.append(f"{path.name} links no media file")
        else:
            issues.extend(self._check_media(ctx, descriptors[0], eaf_dir=path.parent))
        issues.extend(self._check_census(root, tiers))
        if issues:
            raise ValidationError(self.name, issues)
        return {"tiers": len(tiers), "media_descriptors": len(descriptors),
                "bytes": path.stat().st_size}

    @staticmethod
    def _check_media(ctx: StageContext, descriptor: Any, *, eaf_dir: Path) -> list[str]:
        """The linked media must still be findable *from the directory the .eaf sits in*.

        Both URLs are checked because ELAN uses both and each breaks differently: an absolute URL
        dies when the corpus moves, a relative one when the video is renamed. The relative form is
        resolved against ``eaf_dir`` — the .eaf's own directory — and not the process cwd, because
        that is the base ELAN resolves it against. Resolving it against the dataset directory is
        the mistake this check used to make: it agreed with the writer's off-by-one-directory bug
        and pronounced seven unreachable links reachable.
        """
        issues: list[str] = []
        absolute = descriptor.attrib.get("MEDIA_URL") or ""
        relative = descriptor.attrib.get("RELATIVE_MEDIA_URL") or ""
        if not absolute:
            issues.append("media descriptor has no MEDIA_URL")
        if not relative:
            issues.append("media descriptor has no RELATIVE_MEDIA_URL, so the .eaf cannot "
                          "survive the corpus moving")
        expected = Path(ctx.source.path).name
        if expected and expected not in f"{absolute}/{relative}":
            # A stale export copied in from another dataset directory parses, links a video, and
            # shows the wrong clip.
            issues.append(f"media descriptor does not name this video ({expected!r}); the .eaf "
                          "was written for another source")
        if relative and not (eaf_dir / relative).resolve().exists():
            issues.append(f"linked media is not reachable from {eaf_dir.name}/: {relative}")
        return issues

    @staticmethod
    def _check_census(root: Any, tiers: list[Any]) -> list[str]:
        """The tier census written into the document must match the document.

        The stage writes the census as a document property so the file can be checked against
        itself: a tier removed by a hand edit, or by a truncation that left parseable XML behind,
        would otherwise read as a dataset that produced one fewer signal. A missing property is
        reported by nothing and fails nothing — an export from before it existed, or one whose
        HEADER was edited, has annotations that are still fine.
        """
        recorded = ""
        for element in root.iter("PROPERTY"):
            if element.attrib.get("NAME") == "pipeline-tiers":
                recorded = (element.text or "").strip()
                break
        if recorded in ("", "none"):
            return []
        names = {part.split("=", 1)[0] for part in recorded.split() if "=" in part}
        missing = sorted(names - {tier for tier in tiers if tier})
        if missing:
            return [f"{len(missing)} tier(s) listed in the document's own census are absent from "
                    f"it: {', '.join(missing)} — the file was modified after elan wrote it"]
        return []


def _pympi_version() -> str | None:
    """The ELAN writer's version, for provenance.

    Recorded because the .eaf's ``VERSION`` attribute and its XML dialect come from this library
    rather than from anything in this repository, so "which ELAN schema is this corpus in?" has to
    be answerable from the dataset rather than from whatever happens to be installed.
    """
    from importlib import metadata

    try:
        return metadata.version("pympi-ling")
    except Exception:  # noqa: BLE001 - absent dist-info is no reason to fail an export
        return None
