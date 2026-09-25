#!/usr/bin/env python
"""Draw the dataset's own Parquet tables as PNG figures.

The pipeline emits Parquet and JSON only — `find data/processed -name '*.png'` returns
nothing — so a README that wants to *show* what a stage produces has to generate the
images itself, from the same tables a consumer reads. Rendering them here (rather than
saving a screenshot) means the figures cannot rot: rerun the command and they describe
today's schema.

Run it through uv, which is the only supported way to get matplotlib. Matplotlib is
deliberately NOT a project dependency — the pipeline itself never plots, and pinning a
plotting library into a video-processing runtime would make every `uv sync` heavier to
buy nothing. It is imported through one lazy seam (`_pyplot`), with the Agg backend
pinned before pyplot is touched, so `pytest tests/unit` without matplotlib still collects
the whole test file and only the tests that put pixels on canvas skip. Running this file
without matplotlib is a named error and exit 1, never a traceback.

    uv run --with matplotlib python scripts/make_dataset_figures.py \
        --synthetic --seed 7 --out docs/assets
    uv run --with matplotlib python scripts/make_dataset_figures.py \
        --dataset data/processed/<video> --out /tmp/figs

Two modes, one set of renderers:

* ``--dataset`` reads the real Parquet tables of one dataset directory (paths come from
  ``ARTIFACT_LAYOUT``, never hand-concatenated) and renders from them. For the operator's
  own inspection; write it wherever ``--out`` says.
* ``--synthetic`` builds a small in-memory dataset with the same column shapes as the real
  tables and renders the identical kinds of figure.

Committed figures under ``docs/assets/`` come ONLY from ``--synthetic``. That is a legal
constraint, not an aesthetic one: the pipeline processes copyrighted broadcast video
(Telediario, Kimmel, CNN), so a frame-derived image is not ours to redistribute, and
``README.md`` is public-facing documentation for a repository that is. A synthetic strip
demonstrates the *schema* — which columns exist, how dense the frames table is, what a
turn lane looks like against word timings — and that is precisely what a README needs.
A real clip would additionally show a real person's face and lip movement, which buys the
reader nothing they cannot get from the shapes.

Determinism is a feature, not a nicety: same seed and same synthetic data produce
byte-identical PNG bytes (Agg backend, fixed figsize/dpi, nothing time-dependent drawn).
Measured: two separate `uv run` invocations produce the same sha256 for all four files,
and `tests/unit/test_make_dataset_figures.py` asserts the committed bytes are what the
command produces. A committed binary that changes when nobody changed anything is
indistinguishable from a commit that touched real footage, and it makes review noise.
Failure follows the rule ``scripts/make_fixtures.sh`` taught this repository: a missing or
empty table is a named error and a nonzero exit, never an empty PNG. An image that
silently renders nothing is the worst possible artifact, because it looks like the data
was there.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from multimodal_pipeline import SCHEMA_VERSION  # noqa: E402
from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT, VideoPaths  # noqa: E402
from multimodal_pipeline.orchestrator import STAGE_CLASSES  # noqa: E402
from multimodal_pipeline.schemas import (  # noqa: E402
    ACTIVE_SPEAKER_FRAMES_SCHEMA,
    BODY_25_KEYPOINT_NAMES,
    BODY_SCHEMA,
    SPEAKER_TURNS_SCHEMA,
    WORDS_SCHEMA,
)
from multimodal_pipeline.stages.activespeaker import FRAME_REASONS  # noqa: E402
from multimodal_pipeline.stages.base import STAGE_DEPENDENCIES, STAGE_ORDER, Stage  # noqa: E402

# --- figure contract -------------------------------------------------------

FIGURES = (
    "stage_graph",
    "active_speaker_strip",
    "speaker_turn_strip",
    "pose_skeleton_strip",
)

#: What each figure reads, named the way an operator would name it, in the order the
#: renderers ask. Parquet sources are resolved through the artifact registry, so a moved
#: file is a named error rather than a figure drawn from the last dataset that had it.
#: "stage list" has no Parquet behind it: it is the orchestrator's own stage order.
TABLE_SOURCES: tuple[tuple[str, str], ...] = (
    ("active_speaker_frames", "active_speaker_frames"),
    ("speaker_turns", "speaker_turns"),
    ("speech_words", "speech_words"),
    ("pose_body", "pose_body"),
)

FIGURE_FILENAMES: dict[str, str] = {name: f"{name}.png" for name in FIGURES}

#: Columns each renderer actually reads, per source. Checked before drawing so a
#: renamed or absent field fails with the column and table named, instead of raising
#: KeyError from inside matplotlib halfway through a figure — or worse, drawing a strip
#: that quietly has no ticks because `row.get("frame_reason")` returned None everywhere.
REQUIRED_COLUMNS: dict[str, dict[str, tuple[str, ...]]] = {
    "stage_graph": {},
    "active_speaker_strip": {
        "active_speaker_frames": ("timestamp", "frame_reason", "talknet_score",
                                  "score_imputed", "is_active_speaker"),
    },
    "speaker_turn_strip": {
        "speaker_turns": ("turn_id", "speaker_id", "start_time", "end_time", "diarization_type"),
        "speech_words": ("speaker_id", "start_time", "end_time", "word"),
    },
    "pose_skeleton_strip": {
        "pose_body": ("frame_number", "timestamp", "keypoint_name", "x", "y", "confidence"),
    },
}

DPI = 150
CONFIDENCE_FLOOR = 0.25
POSE_FRAMES = 4
MAX_TICK_WIDTH_S = 0.25  # caps where a word label sits, so a long word's label does not
                         # drift off its own tick

#: The only supported way to run this, repeated in the docstring, the argparse epilog and
#: the error a matplotlib-free interpreter gets, so the three cannot disagree.
RUN_COMMAND = "uv run --with matplotlib python scripts/make_dataset_figures.py"

#: Reason colours keyed by the imported FRAME_REASONS vocabulary so every value the
#: stage can write has a colour. A reason absent from this map still renders (grey) and
#: still appears in the legend — an unstyled new reason must be visible, not invisible.
REASON_COLORS: dict[str, str] = {
    "scored": "#1f77b4",
    "imputed_tail": "#ff7f0e",
    "no_face": "#bbbbbb",
    "score_not_finite": "#d62728",
    "track_has_no_scores": "#9467bd",
    "past_scored_tail": "#8c564b",
    "tail_score_not_finite": "#e377c2",
    "unknown": "#17becf",
}
FALLBACK_REASON_COLOR = "#555555"
SPEAKER_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd",
                  "#ff7f0e", "#8c564b", "#17becf", "#bcbd22")

# OpenPose BODY_25 bone pairs. Every endpoint is asserted against the imported
# BODY_25_KEYPOINT_NAMES by tests/unit/test_make_dataset_figures.py, so a renamed
# landmark breaks the test instead of silently dropping that bone from every figure.
# `Background` (index 25) is a filler channel, not a joint, and appears nowhere here.
BODY_25_BONES: tuple[tuple[str, str], ...] = (
    ("Nose", "Neck"),
    ("Neck", "MidHip"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("Neck", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("MidHip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("MidHip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("Nose", "LEye"),
    ("LEye", "LEar"),
    ("Nose", "REye"),
    ("REye", "REar"),
    ("LBigToe", "LSmallToe"),
    ("LBigToe", "LHeel"),
    ("RBigToe", "RSmallToe"),
    ("RBigToe", "RHeel"),
)


class FigureDataError(RuntimeError):
    """A figure cannot be drawn from its inputs. Message always names the source."""


class MatplotlibMissing(RuntimeError):
    """matplotlib is not installed in the interpreter this script runs under."""


# --- matplotlib seam -------------------------------------------------------

_PLT: Any = None


def _pyplot() -> Any:
    """Import pyplot on first use, with Agg pinned first.

    Lazy so the module imports (and the whole unit suite collects) without matplotlib
    installed. ``matplotlib.use("Agg")`` runs before pyplot is ever imported, so a
    headless machine never touches a display backend.
    """
    global _PLT
    if _PLT is None:
        try:
            import matplotlib
        except ModuleNotFoundError as error:
            raise MatplotlibMissing(str(error)) from error
        matplotlib.use("Agg")
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as error:
            raise MatplotlibMissing(str(error)) from error

        _PLT = plt
    return _PLT


# --- synthetic dataset -----------------------------------------------------

#: Column shapes for the in-memory dataset, taken from the real schema objects so the
#: synthetic tables cannot drift from the real ones without a test failing.
SYNTHETIC_SCHEMAS: dict[str, pa.schema] = {
    "active_speaker_frames": ACTIVE_SPEAKER_FRAMES_SCHEMA,
    "speaker_turns": SPEAKER_TURNS_SCHEMA,
    "speech_words": WORDS_SCHEMA,
    "pose_body": BODY_SCHEMA,
}


def _rows_to_table(source: str, rows: list[dict[str, Any]], schema: pa.schema) -> pa.Table:
    """Build a table whose columns are exactly ``schema``'s, or say which one is wrong.

    Checked against the row keys rather than letting ``from_pylist(..., schema=...)``
    paper over it: that call silently nulls a column nobody filled and drops a column
    nobody named, which is exactly how a renamed field would hide.
    """
    if not rows:
        raise FigureDataError(f"{source}: 0 rows, refusing to render an empty figure")
    declared = list(schema.names)
    for index, row in enumerate(rows):
        extra = sorted(set(row) - set(declared))
        missing = sorted(set(declared) - set(row))
        if extra or missing:
            raise FigureDataError(
                f"{source}: row {index} does not match the schema "
                f"(unexpected columns {extra}, missing columns {missing})"
            )
    return pa.Table.from_pylist(rows, schema=schema).cast(schema, safe=False)


def synthetic_frames(*, seed: int, frames: int = 200, fps: int = 25) -> list[dict[str, Any]]:
    """A dense TalkNet-style frames table: one row per frame, gaps and all.

    Shaped after the real thing (measured on the La 1 clip: 200 dense rows, 148
    `scored` / 48 `no_face` / 4 `imputed_tail`), because the point of the figure is to
    show what `frame_reason` is doing. Scores are two arcs, so the strip has a
    believable "who is speaking" rhythm instead of noise.
    """
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index in range(frames):
        timestamp = round(index / fps, 2)
        # Two speakers alternate: track 0 holds the floor for the first half, track 1
        # for the second, with a gap in the middle where no face is located.
        if 96 <= index < 120:
            reason, track, score = "no_face", None, None
        else:
            track = 0 if index < 96 else 1
            centre = 48 if track == 0 else 160
            score = round(max(0.02, min(0.98, 0.5 + 0.45 * _arc(index - centre))), 2)
            # A couple of frames past the end of a track keep the score but disclose
            # that it was carried, not measured.
            reason = "imputed_tail" if index in (94, 95, 158, 159) else "scored"
            if reason == "imputed_tail":
                score = round(0.5 + 0.3 * rng.random(), 2)
        rows.append(
            {
                "schema_version": "1.2",
                "video_id": "synthetic",
                "frame_number": index,
                "timestamp": timestamp,
                "source_timestamp": round(timestamp + rng.uniform(0.0, 0.04), 6),
                "scene_id": 1 if index < 100 else 2,
                "track_id": track,
                "face_status": "no_face" if reason == "no_face" else "tracked",
                "frame_reason": reason,
                "x1": None if track is None else 200.0 + (index % 20),
                "y1": None if track is None else 120.0,
                "x2": None if track is None else 320.0 + (index % 20),
                "y2": None if track is None else 300.0,
                "talknet_score_raw": score,
                "talknet_score": score,
                "score_imputed": reason == "imputed_tail",
                "is_active_speaker": bool(score is not None and score >= 0.5),
            }
        )
    return rows


def _arc(offset: float) -> float:
    """A deterministic bell in [-1, 1] — no numpy, no trig table, no surprise."""
    return max(-1.0, 1.0 - (offset / 40.0) ** 2)


def synthetic_turns_and_words(*, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Two speakers, five turns, and words placed inside them.

    The shapes are what a reader needs: turn lanes on a shared timeline, word ticks
    assigned to a speaker, and the `diarization_type` column that says pyannote produced
    an exclusive timeline.
    """
    rng = random.Random(seed)
    layout = [
        ("SPEAKER_00", 0.30, 3.10),
        ("SPEAKER_01", 3.40, 6.20),
        ("SPEAKER_00", 6.60, 9.80),
        ("SPEAKER_01", 10.10, 12.40),
        ("SPEAKER_00", 12.90, 15.60),
    ]
    turns: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    words_per_turn = 6
    for turn_index, (speaker, start, end) in enumerate(layout, start=1):
        turns.append(
            {
                "schema_version": SCHEMA_VERSION,
                "video_id": "synthetic",
                "turn_id": f"turn{turn_index:06d}",
                "speaker_id": speaker,
                "start_time": start,
                "end_time": end,
                "duration": round(end - start, 3),
                "diarization_type": "exclusive",
            }
        )
        slots = (end - start) / words_per_turn
        for word_index in range(words_per_turn):
            wstart = round(start + slots * word_index + rng.uniform(0.0, 0.02), 3)
            wend = round(min(end, wstart + slots * 0.6), 3)
            words.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "video_id": "synthetic",
                    "segment_id": f"seg{turn_index:06d}",
                    "word_id": f"seg{turn_index:06d}-w{word_index:05d}",
                    "start_time": wstart,
                    "end_time": wend,
                    "duration": round(wend - wstart, 3),
                    "speaker_id": speaker,
                    "word": f"word{word_index}",
                    "confidence": round(0.4 + 0.6 * rng.random(), 3),
                    "alignment_status": "aligned",
                    "character_start": word_index * 6,
                    "character_end": word_index * 6 + 5,
                    "speaker_overlap_seconds": round(wend - wstart, 3),
                    "speaker_overlap_ratio": 1.0,
                    "speaker_assignment_method": "max_overlap",
                }
            )
    return turns, words


def synthetic_pose(*, seed: int, frames: int = 6) -> list[dict[str, Any]]:
    """A standing BODY_25 figure, one detection per sampled frame.

    One frame is deliberately left without a confident detection: the skeleton figure
    has an honest "nothing confident here" panel to draw, and if that path ever stops
    working the synthetic render shows it.
    """
    rng = random.Random(seed)
    origin_x, origin_y = 320.0, 120.0
    offsets: dict[str, tuple[float, float]] = {
        "Nose": (0, 0), "Neck": (0, 40), "RShoulder": (-35, 50), "LShoulder": (35, 50),
        "RElbow": (-55, 105), "LElbow": (55, 105), "RWrist": (-60, 160), "LWrist": (60, 160),
        "MidHip": (0, 150), "RHip": (-25, 160), "LHip": (25, 160),
        "RKnee": (-28, 240), "LKnee": (28, 240), "RAnkle": (-30, 320), "LAnkle": (30, 320),
        "REye": (-8, -8), "LEye": (8, -8), "REar": (-16, 0), "LEar": (16, 0),
        "LBigToe": (20, 332), "LSmallToe": (34, 330), "LHeel": (32, 320),
        "RBigToe": (-20, 332), "RSmallToe": (-34, 330), "RHeel": (-32, 320),
    }
    rows: list[dict[str, Any]] = []
    for index in range(frames):
        frame_number = index * 25
        confident = index != 2
        # A whole-frame low-confidence row set is what "no person" looks like in the
        # real table, so it is reproduced rather than skipped.
        level = 0.9 if confident else 0.05
        for keypoint_id, name in enumerate(BODY_25_KEYPOINT_NAMES):
            if name == "Background":
                continue
            dx, dy = offsets[name]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "video_id": "synthetic",
                    "frame_number": frame_number,
                    "timestamp": round(frame_number / 25.0, 3),
                    "detection_index": 0,
                    "keypoint_id": keypoint_id,
                    "keypoint_name": name,
                    "x": round(origin_x + dx + rng.uniform(-2.0, 2.0), 3),
                    "y": round(origin_y + dy + rng.uniform(-2.0, 2.0), 3),
                    "confidence": round(min(1.0, level + rng.uniform(-0.03, 0.03)), 6),
                }
            )
    return rows


def synthetic_tables(*, seed: int) -> dict[str, Any]:
    """The whole in-memory dataset, keyed by the source names the renderers ask for.

    The four Parquet-backed sources come back as real ``pa.Table``s built through
    ``_rows_to_table``, so the synthetic shapes are checked against the imported schema
    objects on the way to the renderers. A column that exists in the table but not in
    the synthetic rows (or the other way round) raises here rather than producing a
    figure that quietly renders nothing.
    """
    turns, words = synthetic_turns_and_words(seed=seed)
    return {
        "stage list": list(STAGE_ORDER),
        "active_speaker_frames": _rows_to_table(
            "active_speaker_frames", synthetic_frames(seed=seed), ACTIVE_SPEAKER_FRAMES_SCHEMA),
        "speaker_turns": _rows_to_table("speaker_turns", turns, SPEAKER_TURNS_SCHEMA),
        "speech_words": _rows_to_table("speech_words", words, WORDS_SCHEMA),
        "pose_body": _rows_to_table("pose_body", synthetic_pose(seed=seed), BODY_SCHEMA),
    }


# --- dataset reading -------------------------------------------------------

def read_dataset_tables(dataset_dir: Path) -> dict[str, Any]:
    """Read every table the figures need through the registry's own paths.

    Missing files are collected and reported at once rather than raised on the first
    one: with a partially produced dataset the operator wants to know all four answers,
    not one per run.
    """
    paths = VideoPaths(dataset_dir)
    tables: dict[str, Any] = {"stage list": list(STAGE_ORDER)}
    missing: list[str] = []
    for source, artifact in TABLE_SOURCES:
        path = paths.artifact(artifact)
        if not path.is_file():
            missing.append(f"{source} ({ARTIFACT_LAYOUT[artifact]})")
            continue
        tables[source] = _read_table(path)
    if missing:
        raise FigureDataError("dataset is missing table(s): " + ", ".join(missing))
    return tables


def _read_table(path: Path) -> pa.Table:
    import pyarrow.parquet as pq

    return pq.read_table(path)


# --- renderers -------------------------------------------------------------

def _finish(fig: Any, out_path: Path) -> Path:
    """Save, close, and refuse a zero-byte PNG."""
    plt = _pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(out_path, format="png", dpi=DPI)
    finally:
        plt.close(fig)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise FigureDataError(f"{out_path.name}: renderer wrote an empty file")
    return out_path


def _source_rows(data: dict[str, Any], figure: str, source: str) -> list[dict[str, Any]]:
    """Rows for one source, as dicts, with emptiness and shape checked.

    Accepts a ``pa.Table`` (what the synthetic builder and Parquet readers hand over)
    or a list of dicts, so a test can feed either.
    """
    value = data.get(source)
    if isinstance(value, pa.Table):
        rows = value.to_pylist()
    else:
        rows = list(value or [])
    if not rows:
        raise FigureDataError(f"{figure}: source '{source}' has 0 rows, refusing an empty figure")
    missing = sorted(set(REQUIRED_COLUMNS.get(figure, {}).get(source, ())) - set(rows[0]))
    if missing:
        raise FigureDataError(
            f"{figure}: source '{source}' is missing column(s): {', '.join(missing)}")
    return rows


def render_stage_graph(data: dict[str, Any], out_path: Path, *, label: str) -> Path:
    """The real stage DAG, drawn from the orchestrator's own order and dependencies.

    Stage names are never hand-copied here: they come from ``STAGE_ORDER`` and
    ``STAGE_CLASSES``, so adding a stage moves the figure and a stale README diagram
    stays the only possible lie. Depth is the longest dependency path, so every edge
    points left-to-right by construction.
    """
    stages = [name for name in _source_rows(data, "stage_graph", "stage list")]
    unknown = [dep for name in stages for dep in STAGE_DEPENDENCIES.get(name, ()) if dep not in stages]
    if unknown:
        raise FigureDataError(f"stage_graph: dependencies not in the stage list: {sorted(set(unknown))}")

    depth: dict[str, int] = {}
    for name in stages:  # STAGE_ORDER is a topological order, so deps are already placed
        deps = [d for d in STAGE_DEPENDENCIES.get(name, ()) if d in depth]
        depth[name] = 1 + max((depth[d] for d in deps), default=-1)
    rows_by_depth: dict[int, list[str]] = {}
    for name in stages:
        rows_by_depth.setdefault(depth[name], []).append(name)
    lane = {name: index for names in rows_by_depth.values() for index, name in enumerate(names)}
    height = max(len(names) for names in rows_by_depth.values())

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(11.0, 1.1 + 0.95 * height))
    box = dict(boxstyle="round,pad=0.45", linewidth=1.2)
    positions: dict[str, tuple[float, float]] = {}
    for name in stages:
        x, y = float(depth[name] * 2), float(height - 1 - lane[name])
        positions[name] = (x, y)
        # "Has its own enable check" is read off the class, not a hand-kept list: a
        # stage that overrides Stage.enabled can be skipped, so it is drawn dashed.
        cls = STAGE_CLASSES.get(name)
        conditional = cls is not None and cls.enabled is not Stage.enabled
        ax.annotate(
            name, (x, y), ha="center", va="center", fontsize=9,
            bbox={**box,
                  "facecolor": "#f2f2f2" if conditional else "#dbe9f6",
                  "edgecolor": "#666666" if conditional else "#1f4e79",
                  "linestyle": "--" if conditional else "-"},
        )

    for name in stages:
        for dep in STAGE_DEPENDENCIES.get(name, ()):
            if dep not in positions:
                continue
            x0, y0 = positions[dep]
            x1, y1 = positions[name]
            ax.annotate(
                "", xy=(x1 - 0.62, y1), xytext=(x0 + 0.62, y0),
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.0,
                                shrinkA=2, shrinkB=2,
                                connectionstyle="arc3,rad=0.08"),
            )

    ax.set_xlim(-1.4, max(depth.values()) * 2 + 1.6)
    ax.set_ylim(-1.0, height)
    ax.axis("off")
    handles = [
        plt.Line2D([], [], color="#1f4e79", lw=1.4, label="always runs"),
        plt.Line2D([], [], color="#666666", lw=1.4, linestyle="--",
                   label="skippable: own enable check (config, credential, or tool)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"Pipeline stage graph — {label}  ({len(stages)} stages, dependency edges left to right)",
                 fontsize=11)
    return _finish(fig, out_path)


def render_active_speaker_strip(data: dict[str, Any], out_path: Path, *, label: str) -> Path:
    """Frames table as three stacked views of the same x axis.

    A tick per frame coloured by `frame_reason`, the TalkNet score trace, and the
    `is_active_speaker` decision. The three belong together because the table's whole
    argument is that a null and a carried score are different facts: drawn on one axis
    you can see the gap in the ticks, the flat carried section, and the band that
    follows neither.
    """
    rows = _source_rows(data, "active_speaker_strip", "active_speaker_frames")
    reasons = [row.get("frame_reason") for row in rows]
    unknown_reasons = sorted({r for r in reasons if r not in REASON_COLORS and r is not None})

    plt = _pyplot()
    fig, (ax_ticks, ax_score, ax_band) = plt.subplots(
        3, 1, figsize=(10.0, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.6, 0.8]},
    )
    present: list[Any] = []
    for reason in FRAME_REASONS + tuple(unknown_reasons):
        if reason in reasons and reason not in present:
            present.append(reason)
    for reason in present:
        xs = [row["timestamp"] for row in rows if row.get("frame_reason") == reason]
        ax_ticks.vlines(xs, 0, 1, color=REASON_COLORS.get(reason, FALLBACK_REASON_COLOR),
                        linewidth=2.2, label=str(reason))
    ax_ticks.set_yticks([])
    ax_ticks.set_ylabel("frame_reason", fontsize=9)
    ax_ticks.legend(fontsize=7, ncol=min(len(present), 4), loc="upper center",
                    bbox_to_anchor=(0.5, 1.32), frameon=False)
    ax_ticks.set_ylim(-0.15, 1.5)  # leaves room for the legend above the ticks

    scored = [(r["timestamp"], r.get("talknet_score")) for r in rows
              if r.get("talknet_score") is not None]
    if scored:
        ax_score.plot([t for t, _ in scored], [s for _, s in scored],
                      color="#1f4e79", lw=1.1, label="talknet_score")
    imputed = [(r["timestamp"], r["talknet_score"]) for r in rows
               if r.get("score_imputed") and r.get("talknet_score") is not None]
    if imputed:
        ax_score.plot([t for t, _ in imputed], [s for _, s in imputed], "o",
                      mfc="none", mec="#ff7f0e", ms=5, label="score_imputed (carried, not measured)")
    active = [(r["timestamp"], r["talknet_score"]) for r in rows
              if r.get("is_active_speaker") and r.get("talknet_score") is not None]
    if active:
        ax_score.plot([t for t, _ in active], [s for _, s in active], "s",
                      color="#2ca02c", ms=3, label="is_active_speaker")
    ax_score.axhline(0.5, color="#999999", lw=0.8, linestyle=":")
    ax_score.set_ylim(0, 1.05)
    ax_score.set_ylabel("talknet_score", fontsize=9)
    ax_score.legend(fontsize=7, loc="upper right", frameon=False)

    xs = [row["timestamp"] for row in rows]
    active_step = [1.0 if row.get("is_active_speaker") else 0.0 for row in rows]
    ax_band.fill_between(xs, 0, active_step, step="mid", color="#2ca02c", alpha=0.75)
    ax_band.set_ylim(0, 1.4)
    ax_band.set_yticks([0, 1])
    ax_band.set_yticklabels(["off", "active"], fontsize=8)
    ax_band.set_ylabel("is_active_speaker", fontsize=9)

    counts = ", ".join(f"{reason}={reasons.count(reason)}" for reason in present)
    fig.suptitle(f"Active-speaker frames — {label}  ({len(rows)} dense rows)", fontsize=11)
    ax_band.set_xlabel(f"timestamp (s) — frame_reason counts: {counts}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _finish(fig, out_path)


def render_speaker_turn_strip(data: dict[str, Any], out_path: Path, *, label: str) -> Path:
    """Diarization turns as lanes, with the aligned words underneath.

    Two speakers talking over the same seconds is the thing to notice: pyannote's
    exclusive timeline forces one speaker per instant, and the word row shows WhisperX
    timings assigned to that same speaker id space.
    """
    turns = _source_rows(data, "speaker_turn_strip", "speaker_turns")
    words = _source_rows(data, "speaker_turn_strip", "speech_words")

    speakers = sorted({str(row.get("speaker_id")) for row in turns}
                      | {str(row.get("speaker_id")) for row in words})
    color_of = {speaker: SPEAKER_COLORS[index % len(SPEAKER_COLORS)]
                for index, speaker in enumerate(speakers)}

    plt = _pyplot()
    fig, (ax_turns, ax_words) = plt.subplots(
        2, 1, figsize=(10.0, 4.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0]}
    )
    spans = [(float(row.get("start_time") or 0.0), float(row.get("end_time") or 0.0))
             for row in turns + words]
    for row in turns:
        speaker = str(row.get("speaker_id"))
        start = float(row.get("start_time") or 0.0)
        end = float(row.get("end_time") or start)
        lane = speakers.index(speaker)
        kind = row.get("diarization_type") or ""
        ax_turns.barh(lane, max(end - start, 0.01), left=start, height=0.55,
                      color=color_of[speaker], alpha=0.85, edgecolor="#333333", linewidth=0.5)
        ax_turns.text(start + max(end - start, 0.01) / 2, lane, str(row.get("turn_id") or ""),
                      ha="center", va="center", fontsize=6, color="white")
        if kind:
            ax_turns.text(end + 0.05, lane, str(kind), ha="left", va="center",
                          fontsize=6, color="#555555")
    ax_turns.set_yticks(range(len(speakers)))
    ax_turns.set_yticklabels(speakers, fontsize=8)
    ax_turns.set_ylabel("speaker turns", fontsize=9)
    ax_turns.set_ylim(-0.7, len(speakers) - 0.3)

    for row in words:
        speaker = str(row.get("speaker_id"))
        start = float(row.get("start_time") or 0.0)
        end = float(row.get("end_time") or start)
        width = min(max(end - start, 0.01), MAX_TICK_WIDTH_S)
        ax_words.vlines(start, 0, 1, color=color_of.get(speaker, FALLBACK_REASON_COLOR), lw=1.4)
        if row.get("word"):
            ax_words.text(start + width / 2, 1.15, str(row["word"]), rotation=90,
                          ha="center", va="bottom", fontsize=5.5, color="#333333")
    ax_words.set_yticks([])
    ax_words.set_ylim(0, 1.9)
    ax_words.set_ylabel("words", fontsize=9)
    lo = min(start for start, _ in spans)
    hi = max(end for _, end in spans)
    # Set explicitly: vlines alone would cut the axis at the last word's start, so the
    # final word would be drawn against nothing.
    ax_words.set_xlim(lo - 0.05, hi + 0.05 * max(hi - lo, 1.0) + 0.2)
    ax_words.set_xlabel(f"timestamp (s) — {len(turns)} turns, {len(words)} words, vertical ticks coloured by speaker_id")
    ax_turns.set_title(f"Speaker turns against word timings — {label}", fontsize=11)
    fig.tight_layout()
    return _finish(fig, out_path)


def render_pose_skeleton_strip(data: dict[str, Any], out_path: Path, *, label: str) -> Path:
    """Stick figures for a few sampled frames of pose/body.parquet.

    Frames are chosen deterministically (evenly spaced across the frames present), only
    keypoints above CONFIDENCE_FLOOR are drawn, and a frame with nothing confident is
    announced inside its own panel instead of being dropped — silently skipping those
    frames is how a figure ends up showing three confident-looking skeletons for a clip
    where most frames had no usable detection.
    """
    rows = _source_rows(data, "pose_skeleton_strip", "pose_body")
    by_frame: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(row.get("frame_number"), []).append(row)
    ordered = sorted(by_frame, key=lambda frame: (frame is None, frame))
    picks = [ordered[index] for index in
             {round(i * (len(ordered) - 1) / (POSE_FRAMES - 1)) for i in range(POSE_FRAMES)}]
    picks = sorted(p for p in picks if p is not None)
    if not picks:
        raise FigureDataError("pose_skeleton_strip: pose_body has no frame_number values")

    plt = _pyplot()
    fig, axes = plt.subplots(1, len(picks), figsize=(2.9 * len(picks), 3.8), sharey=True)
    if len(picks) == 1:
        axes = [axes]
    drawn = 0
    for ax, frame in zip(axes, picks):
        frame_rows = by_frame[frame]
        confident = [r for r in frame_rows
                     if (r.get("confidence") or 0.0) >= CONFIDENCE_FLOOR
                     and r.get("x") is not None and r.get("y") is not None]
        timestamp = next((r.get("timestamp") for r in frame_rows if r.get("timestamp") is not None), None)
        stamp = "" if timestamp is None else f"{float(timestamp):.2f}s"
        if not confident:
            ax.text(0.5, 0.5, f"frame {frame}\n{stamp}\nno detection at or above\n"
                              f"confidence {CONFIDENCE_FLOOR}",
                    ha="center", va="center", fontsize=8, color="#a33",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        drawn += 1
        points = {str(r.get("keypoint_name")): (float(r["x"]), float(r["y"]))
                  for r in confident}
        for a, b in BODY_25_BONES:
            if a in points and b in points:
                ax.plot(*zip(points[a], points[b]), color="#1f4e79", lw=1.3)
        xs = [p[0] for p in points.values()]
        ys = [p[1] for p in points.values()]
        ax.plot(xs, ys, "o", ms=3.0, color="#d62728")
        ax.invert_yaxis()  # image coordinates: y grows downwards
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"frame {frame}  {stamp}\n{len(points)}/{len(BODY_25_KEYPOINT_NAMES) - 1} keypoints",
                     fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)
    fig.suptitle(f"pose/body.parquet skeletons — {label}  "
                 f"({drawn}/{len(picks)} frames with a confident detection, floor {CONFIDENCE_FLOOR})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _finish(fig, out_path)


RENDERERS: dict[str, Callable[..., Path]] = {
    "stage_graph": render_stage_graph,
    "active_speaker_strip": render_active_speaker_strip,
    "speaker_turn_strip": render_speaker_turn_strip,
    "pose_skeleton_strip": render_pose_skeleton_strip,
}


def render_synthetic(out_dir: Path, *, seed: int = 7, label: str | None = None) -> list[Path]:
    """Render all four figures from the in-memory dataset. Committed assets come from here."""
    return _render(synthetic_tables(seed=seed), out_dir,
                   label=label or f"synthetic schema demo (seed {seed})")


def render_dataset(dataset_dir: Path, out_dir: Path, *, label: str | None = None) -> list[Path]:
    """Render all four figures from one real dataset directory."""
    return _render(read_dataset_tables(dataset_dir), out_dir,
                   label=label or Path(dataset_dir).name)


def _render(data: dict[str, Any], out_dir: Path, *, label: str) -> list[Path]:
    written: list[Path] = []
    failures: list[str] = []
    for figure in FIGURES:
        out_path = Path(out_dir) / FIGURE_FILENAMES[figure]
        try:
            written.append(RENDERERS[figure](data, out_path, label=label))
        except FigureDataError as error:
            failures.append(str(error))
    for message in failures:
        print(f"error: {message}", file=sys.stderr)
    if failures:
        raise FigureDataError(f"{len(failures)} of {len(FIGURES)} figures could not be rendered")
    return written


# --- CLI -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_dataset_figures.py",
        description="Render dataset figures from real Parquet tables or from a synthetic one.",
        epilog=f"Run it as: {RUN_COMMAND} ...",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", type=Path,
                      help="one dataset directory under data/processed/ to read tables from")
    mode.add_argument("--synthetic", action="store_true",
                      help="build a deterministic in-memory dataset (the only mode whose "
                           "output may be committed: no broadcast pixels)")
    parser.add_argument("--out", type=Path, required=True, help="directory to write the PNGs into")
    parser.add_argument("--seed", type=int, default=7, help="synthetic seed (default: 7)")
    parser.add_argument("--video", dest="video", default=None,
                        help="caption label, overriding the dataset name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.synthetic:
            written = render_synthetic(args.out, seed=args.seed, label=args.video)
        elif not args.dataset.is_dir():
            print(f"error: --dataset is not a directory: {args.dataset}", file=sys.stderr)
            return 1
        else:
            written = render_dataset(args.dataset, args.out, label=args.video)
    except MatplotlibMissing as error:
        print(f"error: {error}", file=sys.stderr)
        print(f"error: run it as: {RUN_COMMAND}", file=sys.stderr)
        return 1
    except FigureDataError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for path in written:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"{path} {path.stat().st_size} bytes sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
