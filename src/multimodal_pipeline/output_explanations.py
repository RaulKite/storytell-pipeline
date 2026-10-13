"""Generated ``explanation.md`` guides for one clip's output directory.

``manifest.json`` answers *which files exist*. It cannot answer the questions an
analyst asks two weeks after a run: which of two similar tables was measured and
which was derived, whose ``track_id`` this is, why ``raw/`` is not redundant with
the Parquet beside it, or whether an empty directory means a failure. This module
writes a short guide into every output directory that actually exists.

Each guide's inventory is generated from ``ARTIFACT_LAYOUT``, so an artifact
registered later appears in its own directory's guide without a documentation edit.
The prose says only what a filename cannot recover: clocks, id namespaces, the
raw/normalized rule, and what an optional absence means.

Three properties make these files safe beside a person's own notes:

* The first line is :data:`GENERATOR_MARKER`. A file without it was written by a
  person and is never modified, only reported as preserved.
* The text is a function of the directory alone: no configuration values, no
  machine paths, no timestamps, no counts of what was produced. An unchanged
  dataset therefore rewrites nothing, which is what makes the write idempotent.
* Only directories inside the dataset directory are written. A directory that was
  never produced is not created just to hold a guide, and a link that resolves
  outside the dataset is skipped rather than followed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .artifacts import ARTIFACT_LAYOUT, VideoPaths, atomic_write_text

EXPLANATION_FILENAME = "explanation.md"
ROOT = "."

#: The recognizable marker. Its presence on the *first line* is the only thing that
#: makes a file ours: an ``explanation.md`` without it is the operator's, permanently.
MARKER_TOKEN = "multimodal-pipeline:output-explanations"
GENERATOR_MARKER = (
    f"<!-- {MARKER_TOKEN} generated file: safe to delete, rewritten by every run. "
    "A guide without this line was written by a person and is left untouched. -->"
)

#: Directories that hold no registered file of their own but are part of the layout,
#: either a preserved-raw directory shared by a stage or an optional render target.
EXTRA_DIRS = ("translation/raw", "stories/raw", "pose/raw", "pose/raw_images")

#: Folders that belong to a stage but are not registered artifacts, so a reader is not "
#: left guessing what they are. Named only: a cache holds one file per request, and a "
#: generated guide that enumerated them would be thousands of lines and stale by the "
#: next run. Registered artifacts are listed from the registry, never copied here.
UNREGISTERED_ENTRIES: dict[str, tuple[tuple[str, str], ...]] = {
    "translation/raw": (("`cache/`", "responses already paid for, keyed by request "
                                    "identity; one file per distinct request, and not "
                                    "listed here"),),
    "stories/raw": (("`cache/`", "window responses already paid for, keyed by request "
                                "identity; one file per distinct window, and not listed "
                                "here"),),
}

#: One line per registered artifact: what it is, and where its contract lives.
FILE_BLURBS: dict[str, str] = {
    "metadata": "source metadata derived from ffprobe, including the source SHA256 "
                "used for provenance and invalidation",
    "frame_index": "the source clock: presentation-ordered packet timing, "
                   "`frame_number` → `pts_seconds`",
    "audio": "the single extracted audio track every audio measurement descends from",
    "whisperx_raw": "native WhisperX transcript + word timestamps, untouched",
    "diarization_raw": "native pyannote output, untouched",
    "diarization_rttm": "the same diarization in standard RTTM form",
    "exclusive_diarization_raw": "non-overlapping variant of the pyannote result, the "
                                 "one alignment prefers when configured to",
    "speech_segments": "transcript segments with language and the assigned speaker",
    "speech_words": "words with alignment status and confidence",
    "speaker_turns": "pyannote turn intervals",
    "nemotron_diarization_raw": "native second-engine diarization, untouched",
    "speaker_turns_nemotron": "Nemotron turn intervals — a different `speaker_id` "
                              "namespace from `speaker_turns`",
    "translation_segments": "English gloss keyed by `segment_id`; a translation, not a "
                            "second transcript",
    "translation_raw": "preserved provider responses and the request-keyed cache",
    "spacy_source_tokens": "source-language token annotations",
    "spacy_source_sentences": "source-language sentence annotations",
    "spacy_english_tokens": "English token annotations; not one-to-one with source tokens",
    "spacy_english_sentences": "English sentence annotations",
    "spacy_source_raw": "native spaCy output for the source language, untouched",
    "spacy_english_raw": "native spaCy output for the English text, untouched",
    "acoustic_raw": "the analysis tool's per-frame feature records, untouched",
    "acoustic_frames": "pitch, intensity, voicing and formants on the tool's own time step",
    "acoustic_segments": "per-segment aggregates and pause statistics",
    "pose_raw": "OpenPose's own per-frame JSON, one file per processed frame",
    "pose_body": "BODY_25 keypoints in source pixels",
    "pose_hands": "hand keypoints in source pixels",
    "pose_face": "face landmarks in source pixels",
    "pose_normalized": "body-centred re-expression of `body.parquet`, with basis and "
                       "value status columns; derived, never written back",
    "pose_images_raw": "OpenPose's own rendered skeletons — a preview, and optional",
    "pose_skeleton_video": "OpenPose's own skeleton video (`openpose.write_video=true`): "
                           "a preview rendered by the tool, with source audio muxed in; "
                           "not data and not the measured timeline",
    "activespeaker_raw": "the ASD worker's native per-track result document",
    "activespeaker_tracks_pkl": "TalkNet's native track pickles — Python pickles, so "
                                "load only files you trust",
    "activespeaker_scores_pkl": "TalkNet's native score pickles — Python pickles, so "
                                "load only files you trust",
    "activespeaker_scenes_csv": "TalkNet's own shot (scene) segmentation",
    "activespeaker_video_raw": "TalkNet's *native annotated preview*, muxed with audio: "
                               "what the tool drew, not what the pipeline selected",
    "active_speaker_frames": "the pipeline's *selected* face per TalkNet grid frame, "
                             "with `is_active_speaker` after smoothing and hysteresis",
    "active_speaker_tracks": "per face-track summaries over the audio timeline",
    "speaker_fusion_pyannote": "pyannote turn × face evidence, per turn",
    "speaker_fusion_nemotron": "Nemotron turn × face evidence — kept in its own file so "
                               "the two id namespaces cannot be joined by accident",
    "persons_raw": "the detector's native per-frame document, untouched",
    "person_frames": "body boxes per frame with `person_id`",
    "person_tracks": "per-body-track summaries and sighting gaps",
    "stories": "narrative windows proposed by a language model, with the transcript "
               "evidence each one cites — a proposal to read, not a labelling",
    "stories_raw": "preserved per-window responses and the request-keyed cache",
    "elan_annotations": "the ELAN export: a human-facing summary built from the tables, "
                        "which rewrite nothing and lose detail on purpose",
    "manifest": "produced artifacts by logical name, plus `artifacts_not_generated`",
    "status": "per-stage outcome, skip/rerun reason and recorded validation report",
    "pipeline_log": "the run log for the whole video",
    "provenance_config": "masked configuration snapshot (secrets never reach it)",
    "provenance_tools": "tool and model identities as they were resolved for this run",
    "provenance_processing": "per-stage records: hashes, run sequence, row counts",
}


@dataclass(frozen=True)
class Guide:
    """The label and the prose for one directory; the inventory is generated."""

    label: str
    notes: tuple[str, ...] = field(default_factory=tuple)


#: Notes that apply to every preserved-raw directory, reused verbatim so the rule is
#: stated once and cannot drift between directories.
RAW_NOTES = (
    "This directory preserves a tool's own output, byte-identical, so the dataset can be "
    "re-validated later: normalizing away the original is what makes a stage impossible "
    "to audit.",
    "Normalized tables live beside or above the raw output. Where present, a "
    "`.provenance.json` sidecar records worker request/tool information; other stages "
    "record it in `provenance/` and `status.json`. Raw bytes are not edited to add it.",
)

DIR_GUIDES: dict[str, Guide] = {
    ".": Guide("Clip dataset", (
        "## Where to start",
        "- `manifest.json` names every produced artifact and lists "
        "`artifacts_not_generated` for absent outputs (disabled, skipped or failed). "
        "The manifest, not a directory listing, is the contract.",
        "- `status.json` records each stage's outcome and *why* it ran, reused or was "
        "skipped, with its validation report. An empty directory is therefore normal: "
        "a stage's directories exist before anyone asks the stage to run.",
        "- `provenance/` holds the masked configuration, resolved tool identities and "
        "the per-stage processing record.",
        "## Clocks",
        "- Every time in every table is **seconds from video start**. Intervals use "
        "`start_time`/`end_time`, instants use `timestamp`.",
        "- `source/frame_index.parquet` is the authoritative clock: `frame_number` → "
        "`pts_seconds`, from presentation-ordered packet timing (without decoding).",
        "- `pose/*.parquet` `frame_number` is that same source index. In "
        "`speaker/active_speaker_frames.parquet` it is **not**: TalkNet works on a "
        "re-encoded 25 FPS grid, so `timestamp` is grid time and `source_timestamp` is "
        "the nearest source PTS. Join pose and ASD on `source_timestamp`, never on "
        "`frame_number`, and expect several grid rows to map to one source frame.",
        "- The acoustic clock is the analysis tool's configured time step, not video "
        "frames, and the extracted audio can differ from the container duration "
        "slightly.",
        "## Identifiers",
        "- `speaker_id` is scoped to one diarization engine. "
        "`speech/speaker_turns.parquet` and `speaker/fusion_pyannote.parquet` share one "
        "namespace; `speech/speaker_turns_nemotron.parquet` and "
        "`speaker/fusion_nemotron.parquet` share the other. Two engines means two "
        "namespaces: never union them.",
        "- `track_id` in `speaker/` is a TalkNet *face* track; `person_id` in "
        "`persons/` is a YOLO *body* trajectory. Both are small integers starting near "
        "zero in separate namespaces, so joining them may look reasonable but means "
        "nothing.",
        "- `face_track_id` in `speaker/fusion_*.parquet` is the one explicit bridge, and "
        "it names a TalkNet `track_id` of this dataset, never a diarizer speaker.",
        "- `segment_id` ties translation, linguistic and per-segment acoustic rows back "
        "to `speech/segments.parquet`. It is not a word position.",
        "- All of them are per-clip. Two clips need not agree about who is who.",
        "## Raw and normalized",
        "- A `raw/` directory holds the tool's own output, kept byte-identical. The "
        "Parquet beside or above it is a normalized re-expression, rebuilt by a "
        "normalize step. Preserved raw data enables normalization without new "
        "inference where the stage supports that reuse path.",
        "- Null is not a measured zero, and `score_imputed` means carried evidence "
        "rather than a fresh measurement. Zero rows with a full schema can mean an "
        "analysis that completed and found nothing.",
        "## Anything not listed here",
        "- A directory this guide does not name — `figures/`, a notebook's output, a "
        "hand-made `notes/` — was put here by a person or another script. The pipeline "
        "puts no data in it and deletes nothing from it, and never creates one just to "
        "park a guide. What it may write inside is one generated `explanation.md` saying "
        "exactly that, and nothing else: an `explanation.md` without the marker line on "
        "top stays yours and unmodified.",
    )),
    "source": Guide("Source identity and the frame clock", (
        "The SHA256 in `metadata.json` records source identity. Dependency "
        "fingerprints propagate source changes through downstream stages.",
    )),
    "audio": Guide("Extracted audio", (
        "One WAV is extracted once and every audio measurement descends from it "
        "(transcription, diarization, acoustics). Its duration can differ from the "
        "container duration by a few milliseconds; that difference is the extraction, "
        "not a clock error. `audio_info.json` is the probe of that file and is not a "
        "registered artifact.",
    )),
    "speech": Guide("Transcript, alignment and speakers (normalized)", (
        "Speaker assignment rewrites `segments.parquet` and `words.parquet` in place "
        "with diarizer labels; the native transcript under `raw/` keeps the labels the "
        "ASR tool itself emitted. Read the table for who spoke, the raw file for what "
        "the tool said.",
        "A missing word time is explained by the alignment status columns, not by a "
        "silent zero.",
    )),
    "speech/raw": Guide("Native ASR and diarization output", RAW_NOTES),
    "translation": Guide("English gloss (normalized)", (
        "`segments_en.parquet` joins the transcript by `segment_id`. It is a gloss: "
        "English word positions are not source word positions, and a row's timing can "
        "be its segment's context rather than a measured utterance.",
    )),
    "translation/raw": Guide("Preserved translation responses (optional)", RAW_NOTES + (
        "`cache/` is keyed by request identity and enables reuse of matching "
        "responses. Deleting it can require new API calls; preserve the raw responses "
        "for auditing as well.",
    )),
    "linguistic": Guide("Token and sentence annotations", (
        "Two subdirectories, one per language, are created before execution. "
        "English annotations require translated text; an empty folder proves nothing.",
    )),
    "linguistic/source": Guide("Source-language annotations", RAW_NOTES[:1]),
    "linguistic/english": Guide("English annotations", (
        "Derived from the translated text, so its token spans describe English, not the "
        "source recording. Check the timestamp alignment status before joining tokens on "
        "time.",
    )),
    "linguistic/source/raw": Guide("Native spaCy output, source language", RAW_NOTES),
    "linguistic/english/raw": Guide("Native spaCy output, English", RAW_NOTES),
    "acoustic": Guide("Voice measures (normalized)", (
        "Frame rows sit on the analysis tool's own time step, not on video frames. An "
        "unvoiced or unusable frame is null, not 0 Hz: a silent stretch has no pitch to "
        "report.",
    )),
    "acoustic/raw": Guide("Native acoustic analysis output", RAW_NOTES),
    "pose": Guide("Pose keypoints", (
        "`body.parquet`, `hands.parquet` and `face.parquet` are **source pixels**, "
        "measured per frame of the source index. `normalized.parquet` is a derived "
        "re-expression of the body table only — body-centred, with basis and value "
        "status columns telling a computed value from one that could not be computed.",
        "`raw_images/` and `skeletons.mp4` are the tool's own previews, each opt-in and "
        "independent of the other (`openpose.write_images`, `openpose.write_video`). "
        "They are for looking at. The Parquet is the record, and `frame_index.pts_seconds` "
        "is the timeline: a rendered preview runs at the renderer's constant rate, so "
        "read its timing as approximate and allow at most three trailing frames at the "
        "end. A skeleton video's sound is the source's own audio muxed in afterwards, "
        "absent only when the source really has no audio stream.",
    )),
    "pose/raw": Guide("Native OpenPose JSON", RAW_NOTES + (
        "One file per processed frame, named by frame index, with BODY_25 and "
        "optional face/hand keypoints as flat x, y, score triples. The frame index is "
        "the source packet index, which is how the tables inherit the media clock.",
    )),
    "pose/raw_images": Guide("OpenPose skeleton preview (optional)", (
        "OpenPose's own rendered skeletons, one image per processed frame, present only "
        "when `openpose.write_images=true`. A preview, not data: nothing in this dataset "
        "is computed from these pixels, and `openpose.write_video` asks for the skeleton "
        "video in `pose/` instead — the two switches are independent.",
        "Regeneration clears previous render files rather than mixing them. "
        "`explanation.md` is preserved. With video requested, rendering/normalization/"
        "mux/validation are staged before previous outputs are replaced.",
        "The count is validated against `raw/`, one image per processed frame. "
        "`--render_pose` chooses the rendering engine (`-1` inherits, the GPU path on "
        "this build) and `--output_resolution` chooses the frame size: two different "
        "axes, whatever a provenance record of `-1` looks like.",
    )),
    "persons": Guide("People detected and tracked (normalized)", (
        "`person_id` numbers a *body* trajectory: one id can span a frame where the face "
        "is turned away, and one talking head can produce several ids across a camera "
        "cut. It is a different namespace from TalkNet's `track_id` in `speaker/`, and "
        "the two must never be joined.",
    )),
    "persons/raw": Guide("Native detection output", RAW_NOTES),
    "speaker": Guide("Active speaker evidence", (
        "The two tables here are measured on TalkNet's 25 FPS working grid, not on "
        "source frames: `timestamp` is grid time and `source_timestamp` maps to the "
        "nearest source PTS.",
        "`active_speaker_frames.parquet` carries the *selected* face per grid frame "
        "(`is_active_speaker`, after smoothing and hysteresis). "
        "`speaker/fusion_pyannote.parquet` and `speaker/fusion_nemotron.parquet` answer "
        "the same question about two diarizers and are two files precisely so their id "
        "namespaces cannot be merged by accident.",
    )),
    "speaker/raw": Guide("Native TalkNet output", RAW_NOTES + (
        "`video_out.avi` is TalkNet's **native annotated preview**, kept only when "
        "`activespeaker.write_video=true`: the tool drawing its own per-frame scores on "
        "every face it tracked, muxed with audio. `active_speaker_frames.parquet` above "
        "is the pipeline's **selected** face per grid frame after smoothing and "
        "hysteresis, which is why a flickering face can look active in the video and "
        "not be `is_active_speaker` in the table. The table is the record; the video is "
        "what the tool drew.",
        "Turning that switch off does not delete a video already on disk, so the "
        "manifest reports a file, not a current configuration.",
        "`tracks.pckl` and `scores.pckl` are Python pickles: loading them executes code, "
        "so open only files this repository produced.",
    )),
    "stories": Guide("Narrative candidates", (
        "Rows are a language model's proposal about the transcript, each citing the "
        "segments it came from. They are a reading to check against "
        "`speech/segments.parquet`, never a labelling.",
    )),
    "stories/raw": Guide("Preserved story responses (optional)", RAW_NOTES + (
        "One file per window, named by window and request identity, plus a cache and "
        "`stories_summary.json`; the summary's window count is what the table is "
        "validated against, because an empty window has no rows to count.",
    )),
    "elan": Guide("ELAN export", (
        "`annotations.eaf` is built from the tables and rewrites nothing. It is a "
        "summary a person can open and read: tiers it did not have input for are simply "
        "absent, and the tables stay the record.",
    )),
    "logs": Guide("Run and stage logs", (
        "`pipeline.log` plus one `<stage>.log` per stage; a recorded stage failure names "
        "the log file it wrote. Logs carry tool stderr verbatim, so they can contain "
        "machine paths — treat them as private before sharing a dataset.",
    )),
    "provenance": Guide("Provenance", (
        "`config.json` is the masked configuration snapshot: secrets never reach it. "
        "`tools.json` records the tool and model identities as they were resolved here, "
        "and `processing.json` the per-stage hashes, run sequence and row counts that a "
        "later reuse decision is judged against.",
    )),
}


def is_generated(text: str) -> bool:
    """Whether this text is ours. Only the first line decides."""
    first_line = text.lstrip("\ufeff").split("\n", 1)[0]
    return first_line == GENERATOR_MARKER


def documented_dirs() -> tuple[str, ...]:
    """The directories this module has its own text for, shallowest first.

    Derived from the artifact registry rather than listed by hand: an artifact
    registered into a new directory must bring its own guide with it. This is the set of
    *known* directories; :func:`existing_dirs` is what actually gets written, because a
    real corpus holds directories no layout could name.
    """
    dirs: set[str] = {ROOT, *EXTRA_DIRS}
    for relative in ARTIFACT_LAYOUT.values():
        parts = PurePosixPath(relative).parts[:-1]
        for depth in range(1, len(parts) + 1):
            dirs.add(PurePosixPath(*parts[:depth]).as_posix())
    for relative in DIR_GUIDES:
        dirs.add(relative)
    return tuple(sorted(dirs, key=lambda item: (-1 if item == ROOT else item.count("/") + 1, item)))


def existing_dirs(paths: VideoPaths) -> tuple[str, ...]:
    """Every directory that exists inside one dataset directory, shallowest first.

    A corpus holds more than the layout's fixed slots: ``translation/raw`` accumulates a
    directory per request, ``cache`` grows subdirectories, a stage leaves workspaces
    behind. A guide that only exists in the slots a programmer thought of is missing
    exactly where a person looks, so the walk drives the writing and an unknown
    directory inherits the purpose of the nearest known parent.

    Two things are pruned and never descended into: a hidden directory (a stage's
    temporary workspace, on its way out of the dataset, where a guide would be noise the
    next second) and a symlink (whose content belongs to some other tree).
    """
    root = paths.dataset_dir
    if not root.is_dir():
        return ()
    found: list[str] = [ROOT]
    for current, dirnames, _files in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames
                             if not name.startswith(".")
                             and not (Path(current) / name).is_symlink())
        for name in dirnames:
            found.append((Path(current) / name).relative_to(root).as_posix())
    return tuple(sorted(found, key=lambda item: (-1 if item == ROOT else 0,
                                                  item.count("/"), item)))


def _known_ancestor(relative_dir: str) -> str:
    """The closest directory with its own text, or the dataset root.

    This is what keeps a guide in a deep, unregistered directory useful rather than
    generic: ``translation/raw/cache/<request>/`` is about the translation stage's
    preserved output, because its parent is, and the nearest known parent is the only
    evidence for that.
    """
    known = set(documented_dirs())
    parts = PurePosixPath(relative_dir).parts
    for depth in range(len(parts) - 1, 0, -1):
        candidate = PurePosixPath(*parts[:depth]).as_posix()
        if candidate in known:
            return candidate
    return ROOT


def _dir_label(relative_dir: str, clip: str) -> str:
    if relative_dir == ROOT:
        return f"{DIR_GUIDES[ROOT].label} — clip `{clip}`"
    known = DIR_GUIDES.get(relative_dir)
    if known:
        return f"{known.label} — `{relative_dir}/`"
    ancestor = _known_ancestor(relative_dir)
    if ancestor == ROOT:
        return f"Directory of your own — `{relative_dir}/`"
    return f"{DIR_GUIDES[ancestor].label}, within `{ancestor}/` — `{relative_dir}/`"


def _extend_markdown(lines: list[str], notes: tuple[str, ...]) -> None:
    """Append prose with the blank lines markdown needs, without padding a list.

    Prose notes are separate paragraphs, so they need a blank line between them; a
    bulleted list is meant to be tight, and a heading needs a blank line before it.
    """
    for index, note in enumerate(notes):
        previous = notes[index - 1] if index else ""
        if previous and not (previous.startswith("- ") and note.startswith("- ")):
            lines.append("")
        lines.append(note)


def render_guide(paths: VideoPaths, relative_dir: str) -> str:
    """The full text of one directory's guide. Depends on the directory, nothing else."""
    clip = paths.dataset_dir.name
    lines: list[str] = [GENERATOR_MARKER, "", f"# {_dir_label(relative_dir, clip)}", ""]
    if relative_dir != ROOT:
        up = "../" * (relative_dir.count("/") + 1)
        lines.append("Part of one clip's dataset. The clocks, identifier namespaces and "
                     "raw-vs-normalized rule are explained once, in the guide beside "
                     f"`manifest.json`: [{up}explanation.md]({up}explanation.md).")
        lines.append("")
    guide = DIR_GUIDES.get(relative_dir)
    if relative_dir != ROOT and guide is None:
        _extend_markdown(lines, _inherited_notes(relative_dir))
        lines.append("")
    elif guide and guide.notes:
        _extend_markdown(lines, guide.notes)
        lines.append("")
    lines.append("## What belongs here")
    lines.append("")
    rows = _inventory(relative_dir)
    if rows:
        lines.append("| Name | What it is |")
        lines.append("|---|---|")
        lines.extend(rows)
        lines.append("")
        lines.append("Names are registered in `ARTIFACT_LAYOUT`; `manifest.json` says "
                     "which of them this run produced, and `artifacts_not_generated` "
                     "names the ones it did not. A name listed above and absent on disk "
                     "does not by itself establish a successful or broken dataset; "
                     "read `status.json` for disabled, skipped or failed stages.")
    else:
        lines.append("No individual artifact is registered directly in this directory. "
                     "Its contents may still be covered recursively by a registered "
                     "parent directory artifact (for example `translation/raw`). "
                     "Otherwise they are supplementary material; consult the parent's "
                     "guide and `manifest.json`.")
    return "\n".join(lines) + "\n"


def _inherited_notes(relative_dir: str) -> tuple[str, ...]:
    """The notes for a directory no layout names: what its parent's purpose implies.

    Deliberately short and honest. The pipeline does not know what a per-request or
    per-cache subdirectory holds beyond the fact that the stage owning the parent made
    it, and inventing a purpose for it would be a worse document than saying so.
    """
    ancestor = _known_ancestor(relative_dir)
    if ancestor == ROOT:
        place = "the top level of this dataset, beside `manifest.json`"
        named = "the dataset's own guide"
        link = "../" * (relative_dir.count("/") + 1) + "explanation.md"
    else:
        place = f"`{ancestor}/`"
        named = f"`{ancestor}/explanation.md`"
        link = "../" * (relative_dir.count("/") - ancestor.count("/")) + "explanation.md"
    return (
        "The output layout has no fixed name for this directory, so it is documented by "
        f"inheritance: it sits in {place}, so what it holds belongs to the work "
        f"{named} describes, not to a stage of its own. See [{named}]({link}).",
        "Individual files here are not separate registered names. They may still "
        "belong to a recursively counted parent directory artifact; supplementary "
        "files can also be useful without appearing individually in the manifest.",
        "A directory like this was made by whoever made it — the stage that owns the "
        "parent, or a person. Either way the pipeline puts nothing in it but this guide, "
        "and deletes nothing from it.",
    )


def _inventory(relative_dir: str) -> list[str]:
    """Table rows for the artifacts registered in this directory, plus its subdirectories.

    Read from the live registry rather than a copy kept here, so an artifact added to
    ``ARTIFACT_LAYOUT`` appears in its directory's guide with no documentation edit.
    """
    entries: list[str] = []
    for name, relative in ARTIFACT_LAYOUT.items():
        path = PurePosixPath(relative)
        if path.parent.as_posix() != relative_dir:
            continue
        # A registered path with no suffix is a directory artifact (pose/raw, ...
        # translation/raw), which is how the registry itself treats them.
        label = path.name + ("" if path.suffix else "/")
        blurb = FILE_BLURBS.get(name, "registered artifact; see `manifest.json`")
        entries.append(f"| `{label}` | {blurb} |")
    for child in sorted(item for item in documented_dirs()
                        if item != ROOT and PurePosixPath(item).parent.as_posix() == relative_dir):
        guide = DIR_GUIDES.get(child)
        entries.append(f"| `{PurePosixPath(child).name}/` | "
                       f"{(guide.label if guide else 'Subdirectory').lower()} — see "
                       f"`{child}/explanation.md` |")
    entries.extend(f"| `{name}` | {blurb} |" for name, blurb in UNREGISTERED_ENTRIES.get(relative_dir, ()))
    return sorted(entries)


def _directory_for(paths: VideoPaths, relative_dir: str) -> Path | None:
    """The existing directory inside the dataset, or None when it must not be written."""
    candidate = Path(relative_dir)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    root = paths.dataset_dir
    target = root if relative_dir == ROOT else root / candidate
    if target.is_symlink() or not target.is_dir():
        return None
    # A link, or a directory moved out of the dataset, resolves elsewhere: leave it be.
    resolved_root, resolved = root.resolve(), target.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        return None
    return target


def _new_report() -> dict[str, list[str]]:
    return {"written": [], "unchanged": [], "preserved": [], "skipped": []}


def _write_guarded(target: Path, text: str, key: str, report: dict[str, list[str]]) -> None:
    """One file, with the ownership rules applied: ours is rewritten, theirs is not.

    Every failure mode is reported rather than raised: a guide is documentation, and a
    dataset whose guide could not be written is still a finished dataset.
    """
    if target.is_symlink():
        report["skipped"].append(key)  # never write through a link out of the dataset
        return
    try:
        existing = target.read_text(encoding="utf-8") if target.is_file() else None
    except UnicodeError:
        # Someone wrote this guide in another encoding. It cannot be decoded, so it
        # cannot be recognised as ours — and not ours is the answer that keeps their
        # bytes: an unrecognisable file is preserved, never replaced.
        report["preserved"].append(key)
        return
    except OSError:
        report["skipped"].append(key)
        return
    if existing is not None and not is_generated(existing):
        report["preserved"].append(key)
        return
    if existing == text:
        report["unchanged"].append(key)  # idempotent: an unchanged guide is not rewritten
        return
    try:
        atomic_write_text(target, text)
    except OSError:
        report["skipped"].append(key)
        return
    report["written"].append(key)


def write_explanations(paths: VideoPaths) -> dict[str, tuple[str, ...]]:
    """Write or refresh the guide in every existing output directory of one dataset.

    Returns guide paths by outcome so the caller can say what happened: ``written``,
    ``unchanged`` (already current), ``preserved`` (a person's file, left alone) and
    ``skipped`` (unreadable, a link, or outside the dataset directory).
    """
    report = _new_report()
    for relative_dir in existing_dirs(paths):
        directory = _directory_for(paths, relative_dir)
        if directory is None:
            continue
        key = EXPLANATION_FILENAME if relative_dir == ROOT else f"{relative_dir}/{EXPLANATION_FILENAME}"
        _write_guarded(directory / EXPLANATION_FILENAME, render_guide(paths, relative_dir),
                       key, report)
    return {name: tuple(items) for name, items in report.items()}


def render_batch_guide() -> str:
    """The guide for ``output.directory`` itself: the corpus, not one clip.

    Static on purpose. Clip directories are as numerous as the corpus and their
    contents are the manifest's business, so this names the shape and points inwards.
    """
    return "\n".join([
        GENERATOR_MARKER, "",
        "# Batch output directory",
        "",
        "The folder `output.directory` points at: one dataset directory per input video, "
        "plus the report of the batch that last ran.",
        "",
        "| Name | What it is |",
        "|---|---|",
        "| `batch_report.json` | the last batch's per-video outcome — discovered, "
        "completed, partial, failed — and the elapsed time it took |",
        "| `<video_id>/` | one clip dataset, named by its id. Open its "
        "`explanation.md` for its files, clocks and identifiers |",
        "",
        "## Reading it",
        "",
        "- `batch_report.json` describes a **run**, not the corpus: a clip processed last "
        "month keeps its directory through a batch that never looked at it, and it is the "
        "directory's own `manifest.json` and `status.json` that say what a clip contains.",
        "- Each clip directory describes its own measurements; media references can "
        "still point to source files outside it. Nothing here joins "
        "two clips, and identifiers (`speaker_id`, `track_id`, `person_id`, `segment_id`) "
        "are only meaningful inside one of them.",
        "- The clip folders and their contents are deliberately not listed: a generated "
        "document that enumerated a corpus would be thousands of lines and out of date on "
        "the next run.",
        "- Anything in this folder that is neither a clip directory nor named above was put "
        "here by a person or another script, and the pipeline neither writes into it nor "
        "deletes from it.",
    ]) + "\n"


def write_batch_explanation(root: Path) -> dict[str, tuple[str, ...]]:
    """Write the corpus-level guide into ``root``, same ownership rules as a clip.

    Used by a run whose dataset sits directly in the output directory, and by whoever is "
    backfilling a corpus on disk. A symlinked root is reported as skipped rather than "
    followed, and an unmarked ``explanation.md`` is left exactly as it was.
    """
    report = _new_report()
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        return {name: tuple(items) for name, items in report.items()}
    _write_guarded(directory / EXPLANATION_FILENAME, render_batch_guide(),
                   EXPLANATION_FILENAME, report)
    return {name: tuple(items) for name, items in report.items()}
