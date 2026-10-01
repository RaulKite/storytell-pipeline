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
* what the export represents and what it leaves out travels in the file and in this stage's
  record, because "that tier is missing" has two causes an opened ``.eaf`` cannot tell apart —
  see :func:`~multimodal_pipeline.elan.coverage_inventory`;
* a dataset with *neither* transcript table is a different case and is refused, because an .eaf
  with no words and no segments tier opens to an empty grid and reads as a broken export rather
  than an unfinished dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from ..elan import (ALL_INPUTS, COVERAGE_EXPORTED, COVERAGE_PRESENT_NOT_EXPORTED,
                    COVERAGE_PROPERTY, COVERAGE_STATES, COVERAGE_SUMMARISED, COVERAGE_VERSION,
                    OVERLAP_PROJECTION_PROPERTY, SECONDARY_INPUTS, TIERS, build_eaf, coverage_of,
                    drop_counts, overlap_projection, tier_counts)
from ..exceptions import ValidationError
from .base import Stage, StageContext

#: The artifacts whose joint absence makes the export pointless rather than partial. Every
#: other input is optional; these two are the tiers everyone opens the file for.
TRANSCRIPT_ARTIFACTS = ("speech_words", "speech_segments")


class ElanStage(Stage):
    """Write one ELAN ``.eaf`` per video from the tables every other stage produced."""

    name = "elan"
    # Every table the export reads, in the order `elan.TIERS` declares them and then the tiers'
    # secondary inputs — verified to exist by importing the registry, never invented
    # (test_declared_inputs_are_registered_artifacts). Declared for the same reason every stage
    # declares inputs: the state file and ``status --plan`` should show what a stage consumes.
    # They are read *softly*: ``ctx.input()`` raises on a missing artifact, and an optional
    # producer that never ran is a supported state here rather than an error. ``build_eaf``
    # therefore resolves its own paths from the dataset directory and skips a tier whose file is
    # absent.
    #
    # `ALL_INPUTS` rather than the tier list alone: the person tier reads two tables that no
    # tier is named after, and a file whose contents change the .eaf has to appear in the
    # dependency record and in the fingerprint or the reuse check will not see it.
    inputs = ALL_INPUTS
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

        ``inventory_present`` is here because the coverage inventory makes the document a function
        of three files this stage never reads. ``pose/hands``, ``pose/face`` and
        ``pose/normalized`` are not inputs — no tier is built from them — so their appearance on
        disk moved a *state* in the .eaf while leaving the config, every digest and the dependency
        hash identical, and the reuse check would have kept a file claiming ``absent`` about a table
        that was now there. Existence only, never contents: unread bytes cannot change a state, and
        hashing them would rerun the export on every ``openpose`` run for an unchanged file.

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
            # Which extra files each tier is allowed to read. Hashed alongside the tier names for
            # the same reason the tier names are here: adding a secondary dependency changes what
            # the export can claim, so a dataset must not look reusable across that change.
            "secondary_inputs": {tier: list(names)
                                 for tier, names in sorted(SECONDARY_INPUTS.items())},
            # Which of the inventoried tables were on disk, as booleans only. The coverage
            # inventory writes a table's state into the .eaf, and a table it does not read is one
            # of the four states away from being accurately described by an *existence* check and
            # nothing else — `pose/face.parquet` appearing changes what the document says while
            # changing no byte of any file the export reads. Content is deliberately not hashed
            # here: unread-table contents cannot move a state, so hashing them would rerun the
            # export every time `openpose` reran for no change in the file. Cost is 22 stat()s on a
            # path that already runs on `status --plan`; the 19 read inputs need no flag here,
            # because their digest is already `None` versus a hash.
            "inventory_present": {
                name: (ctx.paths.dataset_dir / elan_core.artifact_path(name)).is_file()
                for name in elan_core.normalized_artifact_names()},
            "_python_code_sha256": python_source_digest(elan_core, elan_stage),
        }
        for artifact in ALL_INPUTS:
            path = ctx.artifact(artifact)
            # Memoised like every other derived stage's digest, because the fingerprint is
            # computed on `status --plan` as well as before a run.
            cache_key = f"elan_digest:{artifact}"
            if cache_key not in ctx.scratch:
                ctx.scratch[cache_key] = sha256_of(path) if path.is_file() else None
            payload[f"{artifact}_digest"] = ctx.scratch[cache_key]
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

        Three parts of the record exist so that a question about the clip can be answered without
        opening the XML. Two of them travel through :meth:`validate`, because that return value is
        the only stage output the orchestrator persists: ``coverage_states`` and
        ``coverage_not_exported`` for "is this signal missing because of the video or because of the
        export", and ``projected_tiers`` for "why are there more bars than rows". The third,
        ``skipped_tiers``, is in the run log and in the run's provenance block only — the
        orchestrator keeps ``tool_version``, ``model_version``, ``command``, ``executable``,
        ``exit_code`` and the validation result, and nothing else from this dict (measured: no
        ``status.json`` on this corpus has ever contained ``tier_counts`` or ``coverage``, for any
        stage). The projected counts are read back out of the built document rather than accumulated
        here, so what the record claims is what the file carries.
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
        eaf, report = build_eaf(dataset_dir, video_path, log=ctx.log)
        counts = tier_counts(eaf)
        # The tier counts in this record are what ELAN shows. Where a tier was re-cut, the number
        # of producer rows behind it is different, and a reader comparing `tier_counts` against a
        # table's row count would otherwise conclude rows went missing. So the projected tiers are
        # named with both numbers, read back out of the document rather than accumulated in memory:
        # what the record reports is what the file carries.
        projection = overlap_projection(eaf)
        descriptor = eaf.media_descriptors[0] if eaf.media_descriptors else {}
        summary: dict[str, Any] = {
            "tiers": len(counts),
            "annotations": sum(counts.values()),
            "tier_counts": dict(sorted(counts.items())),
            # Named in the record and in the log: "this dataset has no person tier" is answered by
            # the log line and by the .eaf's own coverage property. It is *not* answered by
            # status.json — see the method docstring for what the orchestrator actually persists.
            "skipped_tiers": [spec.tier for spec in TIERS if spec.tier not in counts],
            "projected_tiers": {tier: {"logical_rows": document["logical_row_count"],
                                       "emitted": document["final_annotation_count"]}
                                for tier, document in sorted(projection.items())},
            # The other direction of the same question. A tier can show fewer bars than its table
            # has rows because two rows shared an instant and were re-cut (above), or because rows
            # carried no usable time and were refused. Only the second is a producer defect. These
            # counts are in the run log and in this returned block; they are *not* in status.json,
            # because they come from the build report and not from the document, and `validate`
            # (which is what gets persisted) reads only the document. Measured on this corpus every
            # tier dropped zero rows, so nothing here is currently answering a real question — the
            # counters exist for the day a producer starts writing null timestamps.
            "dropped_rows": drop_counts(report),
            # Read back out of the document rather than recomputed here, exactly like the tier
            # counts and the projection: the record's coverage *is* the file's coverage, so the two
            # cannot disagree about whether `pose/face.parquet` is represented.
            "coverage": coverage_of(eaf),
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
        find, a tier lost to a truncation that left well-formed XML behind, a declared tier the
        coverage inventory no longer accounts for, and — since the export began projecting
        simultaneous rows — same-tier overlap and unselectable intervals. Every one of those is
        answered out of the XML: the registry and the tables on disk are never consulted, because
        a document's consistency is a property of the document.

        Because ``validate`` is also the reuse gate (:func:`outputs_present` and the rerun check in
        ``stages.base`` both call it), a document written before the projection rule fails and is
        sent back through ``execute``. That is the intended consequence: a file ELAN cannot lay out
        is not a reusable result, and the rerun is a two-second, pure-python re-export of tables
        that have not changed. It does mean the corpus ``.eaf`` files still on disk report overlap
        until they are regenerated — measured, four of the seven on this machine (KABC, CNN,
        La-1 and ``person_demo``, in ``person_tracks``/``turns_nemotron``/``fusion_nemotron``/
        ``face_tracks``; the three ``pipeline_*`` clips are clean).
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
        issues.extend(self._check_coverage(root, tiers, path.name))
        issues.extend(self._check_independent_tiers(root, path.name))
        if issues:
            raise ValidationError(self.name, issues)
        return {"tiers": len(tiers), "media_descriptors": len(descriptors),
                "bytes": path.stat().st_size, **self._reported_state(root)}

    @staticmethod
    def _reported_state(root: Any) -> dict[str, Any]:
        """What the document says about itself, read out of its own properties.

        This exists because of how the orchestrator persists a stage. ``execute`` returns a rich
        provenance block, but only ``tool_version``, ``model_version``, ``command``, ``executable``,
        ``exit_code`` and **this function's return value** reach ``status.json`` — measured on this
        corpus, no ``status.json`` has ever held ``tier_counts``, ``coverage`` or ``projected_tiers``
        for any stage. So either the questions the export promises to answer are answered here, or
        the promise belongs in the run log instead of in the record.

        Both halves are read from the parsed document rather than remembered from the build, for the
        reason :meth:`_check_independent_tiers` gives for the bars: a value carried in from the run
        that wrote the file is a memory, and a hand-edited file would keep being described by it.
        Reading them here means the record describes the artifact that was just validated.

        State *counts* plus the not-exported names, not the whole 22-entry inventory: the detail
        lives in the file, and the record only has to say how to route the question. Absent keys are
        reported as absent rather than as an empty claim, so a document written before a property
        existed does not gain one.
        """
        properties: dict[str, str] = {}
        for element in root.iter("PROPERTY"):
            name = element.attrib.get("NAME")
            if name in (COVERAGE_PROPERTY, OVERLAP_PROJECTION_PROPERTY):
                properties[name] = (element.text or "").strip()

        reported: dict[str, Any] = {}
        raw_coverage = properties.get(COVERAGE_PROPERTY, "")
        if raw_coverage:
            try:
                document = json.loads(raw_coverage)
            except ValueError:
                document = None
            artifacts = document.get("artifacts") if isinstance(document, dict) else None
            if isinstance(artifacts, dict):
                states = {name: entry.get("state") for name, entry in artifacts.items()
                          if isinstance(entry, dict)}
                counts: dict[str, int] = {}
                for state in states.values():
                    counts[str(state)] = counts.get(str(state), 0) + 1
                reported["coverage_states"] = dict(sorted(counts.items()))
                reported["coverage_not_exported"] = sorted(
                    name for name, state in states.items() if state == COVERAGE_PRESENT_NOT_EXPORTED)
        raw_projection = properties.get(OVERLAP_PROJECTION_PROPERTY, "")
        if raw_projection:
            try:
                document = json.loads(raw_projection)
            except ValueError:
                document = None
            tiers = document.get("tiers") if isinstance(document, dict) else None
            if isinstance(tiers, dict):
                reported["projected_tiers"] = {
                    tier: {"logical_rows": entry.get("logical_row_count"),
                           "emitted": entry.get("final_annotation_count")}
                    for tier, entry in sorted(tiers.items()) if isinstance(entry, dict)}
        return reported

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
        if expected:
            # Every non-empty URL must name the video, compared on the decoded last segment.
            # Three shapes this rejects, each with its own test:
            #  - a substring test accepted ``clip_v2.mp4`` as ``clip.mp4`` (killed by
            #    test_validate_rejects_a_document_written_for_another_video);
            #  - an any-segment test accepted ``../clip.mp4/other.mp4``, where the name is a
            #    directory on the way to some other file (advisory R3-001, killed by
            #    test_validate_rejects_a_name_that_only_appears_as_a_directory);
            #  - requiring *one* URL only, which let a descriptor with a correct absolute and a
            #    relative pointing elsewhere through — that was this line, and the decoy test is
            #    what showed it.
            # ``unquote`` because ``as_uri()`` percent-encodes (``clip one.mp4`` becomes
            # ``clip%20one.mp4``) while ``os.path.relpath`` leaves the name raw; without decoding
            # the absolute side, a video with a space in its name fails its own check.
            tails = {unquote(url.rsplit("/", 1)[-1]) for url in (absolute, relative) if url}
            if tails != {expected}:
                # A stale export copied in from another dataset directory parses, links a video,
                # and shows the wrong clip.
                issues.append(f"media descriptor does not name this video ({expected!r}); the .eaf "
                              "was written for another source")
        if relative and not (eaf_dir / relative).resolve().exists():
            issues.append(f"linked media is not reachable from {eaf_dir.name}/: {relative}")
        return issues

    @staticmethod
    def _check_coverage(root: Any, tiers: list[Any], name: str) -> list[str]:
        """Every tier this document declares must be claimed by a coverage entry.

        Two properties make a claim about the same ``TIER`` elements: ``pipeline-tiers`` says which
        tiers carry annotations (checked by :meth:`_check_census`) and ``pipeline-coverage`` says
        which tier represents each table in the registry. Only the second maps an **artifact** to a
        tier, so it is the one that can be contradicted by an edit the census check cannot see.

        The rule is one-directional, and the direction is the only one a legitimate export cannot
        produce: **every tier this document declares has to be claimed by some coverage entry.**
        A built tier always has an entry naming it, so a declared tier that no entry claims means
        the property was edited — a bar is still on screen and the file no longer says which table
        it came from.

        The reverse is deliberately **not** an error, and that is not an oversight: an entry naming
        a tier this document does not declare is what a *skipped* tier looks like. A table that
        exists but cannot be read leaves its tier undeclared while its artifact stays `exported` —
        the documented "lose one tier, keep the other sixteen" state — and a tier that was never
        written is `absent` and names no tier at all. Complaining there would fail a partial export
        the writer produces on purpose, and since `validate` is also the reuse gate the stage would
        never settle: re-export, skip the same tier, fail the same way forever instead of settling.
        The case worth catching in that direction — a tier removed from the document while its count
        stayed in the census — is exactly what :meth:`_check_census` already reports, with a message
        naming the missing tier. (An earlier revision of this docstring opened by stating that
        reverse as the rule; ``TestValidateChecksCoverageAgainstTheDocument`` in
        ``tests/unit/test_elan_stage.py`` is the guard that the reverse is not enforced, since the
        corrupt-table partial export it would refuse has to keep validating.)

        **The registry is deliberately not consulted.** Asking ``ARTIFACT_LAYOUT`` which tables
        exist would ask the tree what this clip contains, and a tree is edited by rerunning stages;
        that would make a finished document unvalidateable because a producer later wrote one more
        file. The state set, each entry's shape and the tier names are all properties of the file,
        which is the same boundary :meth:`_check_independent_tiers` holds.

        A missing property fails nothing, as with the census: a document written before coverage
        existed, or one whose ``HEADER`` was edited, still holds annotations that are fine, and this
        stage's reuse gate must not destroy a usable export over a missing property. A property that
        parses but holds no artifacts is the same shape — nothing to cross-check. A property that is
        there and is not JSON is different: absence says "this file predates the claim", corruption
        says "the claim exists and cannot be read", which is what an operator needs to know before
        trusting an empty tier to mean "the clip has nothing".

        The version marker *is* refused when it is not the one this code reads, because that failure
        mode is silent. :func:`~multimodal_pipeline.elan.coverage_of` ignores the marker and returns
        the artifact map whatever its shape turns out to be, so a document from a later export could
        be reported — or validated — against meanings its keys no longer carry. Only a document whose
        entries match what this code reads is pronounced consistent, so only such a document may be
        reused as one.
        """
        raw = ""
        for element in root.iter("PROPERTY"):
            if element.attrib.get("NAME") == COVERAGE_PROPERTY:
                raw = (element.text or "").strip()
                break
        if not raw:
            return []
        try:
            document = json.loads(raw)
        except ValueError:
            return [f"{name}: {COVERAGE_PROPERTY} is not valid JSON; rerun elan"]
        if not isinstance(document, dict):
            return [f"{name}: {COVERAGE_PROPERTY} is {type(document).__name__}, expected an "
                    "object; rerun elan"]
        version = document.get("version")
        if version != COVERAGE_VERSION:
            return [f"{name}: {COVERAGE_PROPERTY} is version {version!r}, this export reads "
                    f"{COVERAGE_VERSION}; rerun elan"]
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            return []

        present = {tier for tier in tiers if tier}
        issues: list[str] = []
        claimed: set[Any] = set()
        for artifact, entry in sorted(artifacts.items()):
            if not isinstance(entry, dict):
                issues.append(f"{name}: coverage entry for {artifact} is "
                              f"{type(entry).__name__}, expected an object")
                continue
            state = entry.get("state")
            if state not in COVERAGE_STATES:
                issues.append(f"{name}: coverage entry for {artifact} claims state {state!r}, "
                              f"outside {sorted(COVERAGE_STATES)}")
                continue
            if state not in (COVERAGE_EXPORTED, COVERAGE_SUMMARISED):
                continue
            tier = entry.get("tier")
            if not tier:
                # A shape defect rather than a claim about the tiers: an entry that says a table is
                # represented and names nothing is not a claim that can be checked, so it is not a
                # claim. Absent/present-not-exported carry no tier and are silent by design.
                issues.append(f"{name}: coverage says {artifact} is {state} but names no tier")
                continue
            if not isinstance(tier, str):
                # Same reasoning as the `state` branch above: a value this code cannot interpret is
                # reported, not acted on. It cannot go in the set — a list or dict raises
                # `TypeError: unhashable type` there, and `validate` is the reuse gate, so a
                # non-ValidationError escape is a crashed pipeline rather than a rerun.
                issues.append(f"{name}: coverage entry for {artifact} names tier {tier!r}, "
                              "expected a tier name as a string")
                continue
            claimed.add(tier)
        unclaimed = sorted(present - claimed)
        if unclaimed:
            issues.append(f"{name}: {len(unclaimed)} declared tier(s) are named by no coverage "
                          f"entry: {', '.join(unclaimed)} — the file was modified after elan "
                          "wrote it")
        return issues

    @staticmethod
    def _check_independent_tiers(root: Any, name: str) -> list[str]:
        """No two annotations in one tier may overlap, and every interval must be well-formed.

        ELAN tiers are *independent*: two annotations in the same tier sharing an instant is a
        document ELAN refuses to lay out, and pympi neither writes that away nor reads it back as
        an error — it is exactly the state the export now projects away. Checking it here is what
        makes the rule a validated property rather than a hope: a hand-edited file, an export
        written by an older build, or a future builder that bypassed the projection is caught by
        ``validate`` instead of by whoever next opens ELAN.

        Deliberately resolved from the XML rather than through pympi or the document's own
        properties. :meth:`_check_census` reads a property this stage wrote, so it re-asks the file
        a question the writer already answered; this check reads only ``TIME_SLOT`` values and each
        ``ALIGNABLE_ANNOTATION``'s own ``TIME_SLOT_REF1``/``TIME_SLOT_REF2``, and does **not**
        consult ``pipeline-overlap-projection``. A metadata block that says "these were split
        correctly" is not evidence about the bars, and the failure worth catching is an edit that
        broke the intervals and left the property intact. ``REF_ANNOTATION`` carries no slots of its
        own and this export writes none — every tier here is alignable — so skipping it closes
        nothing.

        Three bound rules come free with the walk, each reported as its own message because each is
        its own way for a file to be wrong: a ``TIME_VALUE`` that is not an integer, a slot
        reference that names no slot, and an interval that is negative or empty (``start >= end``
        cannot be selected in ELAN, however sane each slot looks on its own). Touching intervals
        (``[0, 1000)`` then ``[1000, 2000)``) are legal and are not reported.
        """
        slots: dict[str, int] = {}
        bad_slots: list[str] = []
        for element in root.iter("TIME_SLOT"):
            slot_id = element.attrib.get("TIME_SLOT_ID")
            # EAF writes the millisecond as an attribute (`TIME_VALUE="400"`), not as element text;
            # a missing attribute is read as "not an integer" rather than as 0, because a slot with
            # no value has no claim about when and must not silently become second zero.
            raw = (element.attrib.get("TIME_VALUE") or "").strip()
            try:
                slots[slot_id] = int(raw)
            except ValueError:
                bad_slots.append(f"{slot_id}={raw!r}")
        if bad_slots:
            return [f"{name} has {len(bad_slots)} TIME_SLOT value(s) that are not integers: "
                    f"{', '.join(bad_slots[:5])}"]

        issues: list[str] = []
        for tier in root.iter("TIER"):
            tier_id = tier.attrib.get("TIER_ID")
            if tier_id == "default":
                continue
            pairs: list[tuple[int, int]] = []
            dangling = 0
            for annotation in tier.iter("ALIGNABLE_ANNOTATION"):
                start = annotation.attrib.get("TIME_SLOT_REF1")
                end = annotation.attrib.get("TIME_SLOT_REF2")
                if start not in slots or end not in slots:
                    dangling += 1
                    continue
                pairs.append((slots[start], slots[end]))
            if dangling:
                issues.append(f"{name}: tier {tier_id} has {dangling} annotation(s) whose "
                              "time-slot reference names no TIME_SLOT in the document")
            malformed = sorted(pair for pair in pairs if pair[0] < 0 or pair[0] >= pair[1])
            if malformed:
                issues.append(f"{name}: tier {tier_id} has {len(malformed)} annotation(s) "
                              f"outside ELAN's rule start < end (e.g. {malformed[:3]})")
            # Sorted by start with a running furthest end, so the scan is linear and a *nested*
            # pair — an earlier annotation ending after a later one starts — still counts as the
            # overlap it is. Touching is excluded by the strict `<`.
            pairs.sort()
            overlaps: list[tuple[tuple[int, int], tuple[int, int]]] = []
            furthest: tuple[int, int] | None = None
            for pair in pairs:
                if furthest is not None and pair[0] < furthest[1]:
                    overlaps.append((furthest, pair))
                if furthest is None or pair[1] > furthest[1]:
                    furthest = pair
            if overlaps:
                issues.append(
                    f"{name}: tier {tier_id} has {len(overlaps)} overlapping annotation "
                    f"pair(s) — an ELAN tier is independent and cannot hold them "
                    f"(e.g. {overlaps[:3]}); rerun elan")
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
