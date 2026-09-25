"""Fuse a diarization turn table with the per-frame active-speaker table.

Two measurements of the same audio-visual event, and they answer different
questions: a diarizer says *when does a voice speak*, TalkNet says *which visible
face is talking* at 25 FPS. They agree often and disagree in exactly the cases that
matter — an off-screen narrator (voice, nothing visible), a cutaway (voice continues
over a different face), two faces on screen while one speaks, a face that moves its
mouth while silent. Those disagreements are the interesting signal, so this module
does not resolve them into a speaker label. It keeps the audio turn and attaches an
explicit **agreement state** per turn, so a consumer can tell "the face on screen was
talking" from "we never measured anything in that window".

The core is engine-agnostic on purpose (`TURN_TABLES`): pyannote and Nemotron are two
calls of the same function, not two fusions. Both turn tables share the same leading
columns, so one reader handles both; the only structural difference is Nemotron's
`overlap_s`, which is carried through and left `null` for pyannote, which has no such
number.

What the fused table deliberately does **not** do is join the two speaker-id
namespaces. `SPEAKER_00` (pyannote) and `speaker_0` (Nemotron, ordered by arrival) are
unrelated labels for unrelated clusters, and the digits matching is a coincidence — so
`engine` names which namespace a row's `speaker_id` came from, each engine is written to
its own file, and nothing here pretends a label from one means the same label in the
other. The same rule is why the per-turn face evidence is a *track* id: TalkNet's
`track_id` belongs to a third id space and is never equated with a speaker id either.

Absence is kept separate at every level, which is the lesson `frame_reason` and
`face_status` already learned in this repository:

* `no_face_visible` — the ASD table covers this window and reports no face anywhere in
  it. That is a measurement of "nothing was on screen" (off-screen narrator, audio bed).
* `no_frames_measured` — the ASD table covers **no time at all** in this window. Nothing
  was measured; that is not evidence about who was speaking, and it must never be
  reported as if it were the case above.
* `face_never_active` — faces were visible and measured, and no track was ever flagged
  active. A silent mouth or a cutaway face, which is the opposite conclusion from
  `no_face_visible` even though both leave the face columns empty.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Sequence

SCHEMA_VERSION = "1.0"

#: The two turn tables this fusion can consume. Adding a third diarizer means adding a
#: row here, not writing another fusion.
ENGINE_PYANNOTE = "pyannote"
ENGINE_NEMOTRON = "nemotron"
FUSION_ENGINES: tuple[str, ...] = (ENGINE_PYANNOTE, ENGINE_NEMOTRON)

AGREEMENT_FACE_MATCHED = "face_matched"
AGREEMENT_FACE_PARTIAL = "face_partial"
AGREEMENT_NO_FACE_VISIBLE = "no_face_visible"
AGREEMENT_FACE_NEVER_ACTIVE = "face_never_active"
AGREEMENT_NO_FRAMES_MEASURED = "no_frames_measured"

#: Closed vocabulary. A consumer switches on these five strings and cannot handle a
#: sixth, so `validate` treats anything else as a defect rather than a variant.
AGREEMENT_STATES: tuple[str, ...] = (
    AGREEMENT_FACE_MATCHED,
    AGREEMENT_FACE_PARTIAL,
    AGREEMENT_NO_FACE_VISIBLE,
    AGREEMENT_FACE_NEVER_ACTIVE,
    AGREEMENT_NO_FRAMES_MEASURED,
)


@dataclass(frozen=True)
class TurnTableSpec:
    """Which turn table to fuse against, and what it carries.

    `overlap_column` is the *only* structural difference the fused table has to know
    about: pyannote's exclusive timeline has no overlap figure, so its rows get a null
    rather than a fabricated 0.0 (a 0 would read as "measured: no overlap").
    """

    engine: str
    artifact: str
    overlap_column: str | None


#: Logical artifact name of every readable turn table, keyed by engine name.
TURN_TABLES: dict[str, TurnTableSpec] = {
    ENGINE_PYANNOTE: TurnTableSpec(ENGINE_PYANNOTE, "speaker_turns", None),
    ENGINE_NEMOTRON: TurnTableSpec(ENGINE_NEMOTRON, "speaker_turns_nemotron", "overlap_s"),
}

#: Frames whose timestamp falls inside ``[start_time, end_time]``, both ends included.
#: The frames table stores instants rather than spans, so a boundary frame shared by two
#: consecutive turns legitimately belongs to both: at that instant both turns claim a
#: voice. Half-open windows would silently hand a measured frame to one engine's turn and
#: withhold it from the other's.
def frame_in_turn(frame_timestamp: float, start_time: float, end_time: float) -> bool:
    return start_time <= frame_timestamp <= end_time


def fuse_turn_table(
    *,
    video_id: str,
    engine: str,
    turns: Sequence[dict[str, Any]],
    frames: Sequence[dict[str, Any]],
    min_active_ratio: float = 0.5,
    min_face_frames: int = 2,
) -> list[dict[str, Any]]:
    """One fused row per turn of one engine's turn table.

    ``turns`` and ``frames`` are plain row dicts using the column names of
    ``SPEAKER_TURNS_SCHEMA`` / ``SPEAKER_TURNS_NEMOTRON_SCHEMA`` and
    ``ACTIVE_SPEAKER_FRAMES_SCHEMA``. Nothing is re-derived from them: the turn's own
    timing and speaker label are copied verbatim, and the face columns describe what the
    ASD table actually recorded inside the turn's window.

    The two thresholds are the caller's, so the same core answers "who was talking?" with
    whatever evidence bar the operator asked for.
    """
    spec = TURN_TABLES.get(engine)
    if spec is None:
        raise ValueError(
            f"unknown speaker-fusion engine {engine!r}; readable turn tables are "
            f"{', '.join(FUSION_ENGINES)}"
        )
    if not 0.0 < min_active_ratio <= 1.0:
        raise ValueError("min_active_ratio must be in (0, 1]")
    if min_face_frames < 1:
        raise ValueError("min_face_frames must be >= 1")

    timestamps, timed = _sorted_frames(frames)
    rows: list[dict[str, Any]] = []
    for turn in turns:
        start = _number(turn.get("start_time"))
        end = _number(turn.get("end_time"))
        window = _window(timed, timestamps, start, end)
        evidence = _track_evidence(window)
        verdict = _classify(
            evidence,
            frames_in_turn=len(window),
            min_active_ratio=min_active_ratio,
            min_face_frames=min_face_frames,
        )
        overlap_s = turn.get(spec.overlap_column) if spec.overlap_column else None
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "video_id": video_id,
            # Which diarizer produced this turn, and therefore which speaker-id
            # namespace `speaker_id` is drawn from. Not a join key across engines.
            "engine": engine,
            "turn_id": turn.get("turn_id"),
            "speaker_id": turn.get("speaker_id"),
            "start_time": start,
            "end_time": end,
            "duration": _number(turn.get("duration")),
            "diarization_type": turn.get("diarization_type"),
            # Nemotron's measured overlap; genuinely absent for a pyannote turn.
            "overlap_s": None if overlap_s is None else float(overlap_s),
            "face_track_id": verdict["face_track_id"],
            "face_active_frames": verdict["face_active_frames"],
            "face_frames_in_turn": verdict["face_frames_in_turn"],
            "frames_in_turn": verdict["frames_in_turn"],
            "face_mean_score": verdict["face_mean_score"],
            "face_score_max": verdict["face_score_max"],
            "agreement": verdict["agreement"],
            "agreement_detail": verdict["agreement_detail"],
        })
    return rows


# --------------------------------------------------------------------------- core


def _sorted_frames(frames: Sequence[dict[str, Any]]) -> tuple[list[float], list[dict[str, Any]]]:
    """Frames with a usable timestamp, ordered by it.

    A row without a timestamp cannot be placed in any window, so it is dropped here
    rather than counted as evidence for the turn that happens to be scanned first. The
    frames stage writes a timestamp on every row, so this is a guard against a hand-edited
    or truncated table, not an expected path.
    """
    timed = [(value, row) for row in frames
             if (value := _number(row.get("timestamp"))) is not None]
    timed.sort(key=lambda item: item[0])
    return [value for value, _ in timed], [row for _, row in timed]


def _window(rows: Sequence[dict[str, Any]], timestamps: Sequence[float],
            start: float | None, end: float | None) -> list[dict[str, Any]]:
    if start is None or end is None:
        return []
    left = bisect_left(timestamps, start)
    right = bisect_right(timestamps, end)
    return list(rows[left:right])


def _track_evidence(window: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-track counts inside one turn window.

    Choosing the winner is *not* done here: `in_range` is the denominator for the active
    ratio — the frames where TalkNet says *this* track is the face on screen — and that
    choice belongs to `_choose_track`, the only place the two thresholds are applied.
    Ratio-ing a track's active frames against the whole window instead would punish a face
    that is only on screen for part of a turn — a cutaway — and report it as a weaker match
    than the numbers support.
    """
    tracks: dict[int, dict[str, Any]] = {}
    face_frames = 0
    for row in window:
        track_id = row.get("track_id")
        if track_id is None:
            continue  # face_status == no_face: a real measurement of an empty frame
        face_frames += 1
        track = tracks.setdefault(int(track_id), {"present": 0, "active": 0, "scores": []})
        track["present"] += 1
        if row.get("is_active_speaker"):
            track["active"] += 1
        # Every measured frame of this track contributes to the score summary, not just
        # the active ones: `face_mean_score` answers "how strongly did TalkNet believe
        # this face was talking in this window", which needs the low frames too. Frames
        # the ASD stage could not score (face_status tracked_unscored) carry no number and
        # are left out rather than averaged in as a zero.
        score = _number(row.get("talknet_score"))
        if score is not None:
            track["scores"].append(score)

    return {"tracks": tracks, "face_frames": face_frames}


def _ratio(track: dict[str, Any]) -> float:
    return track["active"] / track["present"] if track["present"] else 0.0


def _qualifies(track: dict[str, Any], min_active_ratio: float, min_face_frames: int) -> bool:
    return track["active"] >= min_face_frames and _ratio(track) >= min_active_ratio


def _choose_track(tracks: dict[int, dict[str, Any]], *, min_active_ratio: float,
                  min_face_frames: int) -> tuple[int | None, list[int], bool]:
    """Pick the track that owns the turn, and say whether it actually qualifies.

    Qualification comes first, and that ordering is the whole rule: a track that clears
    both thresholds answers *who is talking*, while a track with more active frames but a
    poor ratio only answers *who was on screen longest*. Ranking by absolute active count
    first and testing the ratio afterwards could report a face that was present all
    through and barely spoke, and hide the one that spoke whenever it was visible — the
    cutaway case this table is meant to catch.

    So: among the qualifying tracks, most active frames wins; when none qualifies, the same
    ordering picks the least-bad candidate and the turn is `face_partial`. Ties on the
    primary measure are broken by higher mean score, then lower track id, and reported — a
    per-turn label that depended on dictionary order would be a label nobody could
    reproduce.
    """
    candidates = [track_id for track_id, track in tracks.items() if track["active"] > 0]
    if not candidates:
        return None, [], False
    qualified = [track_id for track_id in candidates
                 if _qualifies(tracks[track_id], min_active_ratio, min_face_frames)]
    pool = qualified or candidates

    def key(track_id: int) -> tuple[int, float, int]:
        track = tracks[track_id]
        return (track["active"], round(_mean(track["scores"]) or 0.0, 6), -track_id)

    winner = max(pool, key=key)
    # Tracks that tied with the winner on the primary measure and were separated only by a
    # tie-break: the winner is a convention there, so the row says so. A lone leader is not
    # "a tie between track 0", and reporting one would send a reader looking for a loser.
    tied = sorted(track_id for track_id in pool
                  if tracks[track_id]["active"] == tracks[winner]["active"])
    return winner, (tied if len(tied) > 1 else []), bool(qualified)


def _classify(evidence: dict[str, Any], *, frames_in_turn: int, min_active_ratio: float,
              min_face_frames: int) -> dict[str, Any]:
    """One agreement state per turn, with the measured numbers in its words.

    The order of these branches is the whole point: `no_frames_measured` is tested before
    `no_face_visible` because they answer different questions, and the second is tested
    before `face_never_active` because "no face was located" and "a face was located and
    never spoke" share two empty columns and nothing else.
    """
    tracks: dict[int, dict[str, Any]] = evidence["tracks"]
    face_frames = evidence["face_frames"]
    winner, tied, qualifies = _choose_track(
        tracks, min_active_ratio=min_active_ratio, min_face_frames=min_face_frames)
    base = {
        "face_track_id": None,
        "face_active_frames": 0,
        "face_frames_in_turn": face_frames,
        "frames_in_turn": frames_in_turn,
        "face_mean_score": None,
        "face_score_max": None,
    }

    if frames_in_turn == 0:
        return {**base, "agreement": AGREEMENT_NO_FRAMES_MEASURED, "agreement_detail":
                "the active-speaker frame table covers no frame in this turn's window, so "
                "nothing was measured here (absence of measurement, not a sighting of an "
                "empty scene)"}
    if face_frames == 0:
        return {**base, "agreement": AGREEMENT_NO_FACE_VISIBLE, "agreement_detail":
                f"no face was located in any of the {frames_in_turn} ASD frames in this "
                f"turn (voice with nothing visible: off-screen narrator or audio bed)"}

    if winner is None:
        return {**base, "agreement": AGREEMENT_FACE_NEVER_ACTIVE, "agreement_detail":
                f"faces visible on {face_frames}/{frames_in_turn} measured frames across "
                f"{len(tracks)} track(s), but no track was ever flagged an active speaker "
                f"(silent mouth or cutaway face)"}

    track = tracks[winner]
    active, in_range = track["active"], track["present"]
    ratio = _ratio(track)
    mean = _mean(track["scores"])
    top = max(track["scores"]) if track["scores"] else None
    scores = {"face_track_id": winner, "face_active_frames": active,
              "face_mean_score": None if mean is None else round(mean, 6),
              "face_score_max": None if top is None else round(float(top), 6)}
    prefix = _tie_note(tied, winner)

    gap = _gap_note(frames_in_turn, face_frames)
    if qualifies:
        return {**base, **scores, "agreement": AGREEMENT_FACE_MATCHED, "agreement_detail":
                f"{prefix}track {winner} active on {active}/{in_range} frames in turn "
                f"(ratio {ratio:.2f} >= min_active_ratio {min_active_ratio:g}, "
                f"{active} >= min_face_frames {min_face_frames}), "
                f"mean score {_fmt(mean)}{gap}"}

    reason = (_shortfall(active, in_range, ratio, min_active_ratio, min_face_frames))
    return {**base, **scores, "agreement": AGREEMENT_FACE_PARTIAL, "agreement_detail":
            f"{prefix}track {winner} active on {active}/{in_range} frames in turn "
            f"(ratio {ratio:.2f}, mean score {_fmt(mean)}) — {reason}{gap}"}


def _gap_note(frames_in_turn: int, face_frames: int) -> str:
    """Name the part of the window where nothing was on screen.

    A turn can match on the frames where a face *was* visible and still contain a stretch
    with nobody in shot — a voice-over dropped into the middle of a report. Without this
    clause the ratio reads as if it described the whole turn, which is the flattening this
    table exists to avoid; the numbers are already columns, but the detail is the one thing
    a human reads, so it has to carry the qualification too.
    """
    hidden = frames_in_turn - face_frames
    if hidden <= 0:
        return ""
    return f"; no face located on {hidden}/{frames_in_turn} measured frames of the window"


def _shortfall(active: int, in_range: int, ratio: float, min_active_ratio: float,
               min_face_frames: int) -> str:
    """Why a best track still did not clear the bar, naming the bar it missed."""
    misses: list[str] = []
    if active < min_face_frames:
        misses.append(f"{active} active frame(s) is below min_face_frames {min_face_frames}")
    if ratio < min_active_ratio:
        misses.append(f"ratio {ratio:.2f} is below min_active_ratio {min_active_ratio:g}")
    return "; ".join(misses) if misses else "thresholds not met"


def _tie_note(tied: Sequence[int], winner: int) -> str:
    if not tied:
        return ""
    listed = ", ".join(str(track_id) for track_id in tied)
    return f"tie between tracks {listed} resolved to {winner} by mean score, then track id; "


def _mean(values: Sequence[float]) -> float | None:
    """Mean of the measured scores, or None when the track has none.

    A track can be flagged active on frames whose score did not survive normalisation.
    Reporting 0.0 there would claim a measured average of zero, which is a verdict about
    the face rather than the absence of one.
    """
    return sum(values) / len(values) if values else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # A NaN timestamp would sort nowhere and make every window comparison False.
    return None if result != result else result
