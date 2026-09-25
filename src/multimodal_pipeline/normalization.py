"""Normalisation helpers: raw model/tool output → canonical Parquet rows.

Kept separate from stages so the same normalisation can be re-run over
preserved raw artifacts after a schema fix, without re-running the models.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .schemas import BODY_25_KEYPOINT_NAMES, FACE_KEYPOINT_COUNT, HAND_KEYPOINT_NAMES

WHISPERX_SCHEMA_VERSION = "1.0"


def whisperx_segment_rows(payload: dict[str, Any], video_id: str) -> list[dict[str, Any]]:
    """WhisperX ``transcribe``/``align`` result → canonical segment rows."""
    language = payload.get("language")
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(payload.get("segments", []) or []):
        start = _float(segment.get("start"))
        end = _float(segment.get("end"))
        text = (segment.get("text") or "").strip()
        rows.append({
            "schema_version": WHISPERX_SCHEMA_VERSION,
            "video_id": video_id,
            "segment_id": _segment_id(index, segment),
            "start_time": start,
            "end_time": end,
            "duration": _duration(start, end),
            "language": language,
            "speaker_id": segment.get("speaker"),
            "text": text,
            "confidence": _segment_confidence(segment),
        })
    return rows


def whisperx_word_rows(payload: dict[str, Any], video_id: str) -> list[dict[str, Any]]:
    """WhisperX words (aligned or not) → canonical word rows.

    When the alignment stage produced no ``words`` for a segment, word rows are
    synthesised from the segment text with segment-level timing and marked
    ``segment_only`` so consumers know the timestamps are not word-accurate.
    """
    rows: list[dict[str, Any]] = []
    for seg_index, segment in enumerate(payload.get("segments", []) or []):
        segment_id = _segment_id(seg_index, segment)
        seg_start = _float(segment.get("start"))
        seg_end = _float(segment.get("end"))
        words = segment.get("words") or []
        if words:
            for word_index, word in enumerate(words):
                start = _float(word.get("start"))
                end = _float(word.get("end"))
                rows.append({
                    "schema_version": WHISPERX_SCHEMA_VERSION,
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "word_id": f"{segment_id}-w{word_index:05d}",
                    "start_time": start,
                    "end_time": end,
                    "duration": _duration(start, end),
                    "speaker_id": word.get("speaker") or segment.get("speaker"),
                    "word": (word.get("word") or "").strip(),
                    "confidence": _float(word.get("confidence")) if word.get("confidence") is not None
                    else _float(word.get("score")),
                    "alignment_status": word_alignment_status(word),
                    "character_start": _int(word.get("start_char")),
                    "character_end": _int(word.get("end_char")),
                })
            continue
        # No aligned words: fall back to whitespace tokens of the segment text.
        offset = 0
        text = segment.get("text") or ""
        for word_index, match in enumerate(re.finditer(r"\S+", text)):
            rows.append({
                "schema_version": WHISPERX_SCHEMA_VERSION,
                "video_id": video_id,
                "segment_id": segment_id,
                "word_id": f"{segment_id}-w{word_index:05d}",
                "start_time": seg_start,
                "end_time": seg_end,
                "duration": _duration(seg_start, seg_end),
                "speaker_id": segment.get("speaker"),
                "word": match.group(0),
                "confidence": None,
                "alignment_status": "segment_only",
                "character_start": match.start() - offset,
                "character_end": match.end() - offset,
            })
    return rows


def word_alignment_status(word: dict[str, Any]) -> str:
    """Explicit, non-null status so consumers never guess why a timestamp is null."""
    if _float(word.get("start")) is None or _float(word.get("end")) is None:
        return "missing_timestamp"
    return "aligned"


def _segment_id(index: int, segment: dict[str, Any]) -> str:
    raw = segment.get("id")
    if isinstance(raw, int):
        return f"seg{raw + 1:06d}"
    if isinstance(raw, str) and raw:
        return raw
    return f"seg{index + 1:06d}"


def _segment_confidence(segment: dict[str, Any]) -> float | None:
    """WhisperX exposes ``avg_logprob``/``no_speech_prob``; a probability is kept raw."""
    value = _float(segment.get("confidence"))
    return value if value is not None else _float(segment.get("avg_logprob"))


def diarization_turn_rows(payload: dict[str, Any], video_id: str,
                          diarization_type: str = "inclusive") -> list[dict[str, Any]]:
    """Pyannote result (list of ``[start, end, speaker]``) → canonical turn rows."""
    rows: list[dict[str, Any]] = []
    turns = payload.get("turns") or payload.get(f"{diarization_type}_turns") or []
    ordered = sorted(turns, key=lambda turn: (_float(turn[0]) or 0.0, _float(turn[1]) or 0.0))
    for index, turn in enumerate(ordered):
        if len(turn) < 3:
            continue
        start, end, speaker = _float(turn[0]), _float(turn[1]), str(turn[2])
        rows.append({
            "schema_version": WHISPERX_SCHEMA_VERSION,
            "video_id": video_id,
            "turn_id": f"turn{index + 1:06d}",
            "speaker_id": speaker,
            "start_time": start,
            "end_time": end,
            "duration": _duration(start, end),
            "diarization_type": diarization_type,
        })
    return rows


def nemotron_turn_rows(payload: dict[str, Any], video_id: str) -> list[dict[str, Any]]:
    """Nemotron worker document → canonical turn rows, with per-segment overlap measured.

    Separate from :func:`diarization_turn_rows` because the shapes differ in two ways that
    matter: Nemotron emits objects with a speaker *index* rather than `[start, end, name]`
    triples, and its segments overlap across channels, so each row needs the seconds of
    cross-speaker overlap it participates in.

    Zero-length segments are dropped rather than turned into a negative duration.
    ``extract_speaker_dict`` can emit a segment whose end equals its start when a speaker
    is active for exactly one 10 ms frame and the next frame is not; a row with
    ``duration = 0`` poisons every per-speaker mean downstream, and "one frame of speech"
    is not recoverable information at 25 FPS anyway. The raw JSON keeps the segment, so the
    drop is reviewable and reversible.
    """
    raw = [row for row in (payload.get("segments") or []) if isinstance(row, dict)]
    kept: list[dict[str, Any]] = []
    for row in raw:
        start, end = _float(row.get("start")), _float(row.get("end"))
        speaker = row.get("speaker_id")
        if start is None or end is None or not speaker:
            continue
        if end - start <= 0:
            continue
        kept.append({
            "schema_version": WHISPERX_SCHEMA_VERSION,
            "video_id": video_id,
            "turn_id": "",
            "speaker_id": str(speaker),
            "start_time": start,
            "end_time": end,
            "duration": _duration(start, end),
            # "overlapping": this is an inclusive timeline by construction. Naming it here
            # rather than reusing "inclusive" keeps the two engines' provenance distinct.
            "diarization_type": "overlapping",
            "overlap_s": 0.0,
        })

    # Overlap is measured against *other* speakers only. Two segments of the same speaker
    # that touch are a bookkeeping artefact, not the overlapping-speech signal the column
    # exists to record.
    for i, row in enumerate(kept):
        overlapped = 0.0
        for j, other in enumerate(kept):
            if i == j or other["speaker_id"] == row["speaker_id"]:
                continue
            hi = min(row["end_time"], other["end_time"])
            lo = max(row["start_time"], other["start_time"])
            if hi > lo:
                overlapped += hi - lo
        row["overlap_s"] = round(overlapped, 6)

    kept.sort(key=lambda row: (row["start_time"], row["end_time"], row["speaker_id"]))
    for index, row in enumerate(kept):
        row["turn_id"] = f"turn{index + 1:06d}"
    return kept


def parse_rttm(text: str) -> list[dict[str, Any]]:
    """Speaker Turn RTTM → rows. Kept so a diarization rerun is reproducible from raw."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        try:
            start, duration = float(parts[3]), float(parts[4])
        except ValueError:
            continue
        rows.append({"speaker_id": parts[7], "start_time": start,
                     "end_time": start + duration, "duration": duration})
    return rows


# ----------------------------------------------------------------- openpose

BODY_25_NAME_LIST = list(BODY_25_KEYPOINT_NAMES)


def openpose_frame_rows(document: dict[str, Any], video_id: str, frame_number: int,
                        timestamp: float, *, include_background: bool = False) -> dict[str, list[dict[str, Any]]]:
    """One OpenPose JSON frame → body/hands/face row dicts.

    OpenPose stores flat ``[x, y, score, ...]`` triples; a zero score means the
    keypoint was not found, which we keep (with ``confidence = 0``) instead of
    dropping, so downstream code can distinguish "not detected" from "absent
    person". ``detection_index`` is the frame-local person index — it is **not**
    a cross-frame identity unless OpenPose tracking is enabled.
    """
    body: list[dict[str, Any]] = []
    hands: list[dict[str, Any]] = []
    face: list[dict[str, Any]] = []
    people = document.get("people", []) or []
    for detection_index, person in enumerate(people):
        pose = person.get("pose_keypoints_2d") or []
        for keypoint_id, (x, y, score) in enumerate(iterate_keypoints(pose)):
            if keypoint_id >= len(BODY_25_NAME_LIST):
                continue
            name = BODY_25_NAME_LIST[keypoint_id]
            if name == "Background" and not include_background:
                continue
            if score <= 0:
                continue
            body.append({
                "schema_version": WHISPERX_SCHEMA_VERSION, "video_id": video_id,
                "frame_number": frame_number, "timestamp": timestamp,
                "detection_index": detection_index, "keypoint_id": keypoint_id,
                "keypoint_name": name, "x": x, "y": y, "confidence": score,
            })
        for side, key in (("left", "hand_left_keypoints_2d"), ("right", "hand_right_keypoints_2d")):
            points = person.get(key) or []
            for keypoint_id, (x, y, score) in enumerate(iterate_keypoints(points)):
                if score <= 0:
                    continue
                name = HAND_KEYPOINT_NAMES[keypoint_id] if keypoint_id < len(HAND_KEYPOINT_NAMES) else f"K{keypoint_id}"
                hands.append({
                    "schema_version": WHISPERX_SCHEMA_VERSION, "video_id": video_id,
                    "frame_number": frame_number, "timestamp": timestamp,
                    "detection_index": detection_index, "hand": side,
                    "keypoint_id": keypoint_id, "keypoint_name": name,
                    "x": x, "y": y, "confidence": score,
                })
        face_points = person.get("face_keypoints_2d") or []
        for landmark_id, (x, y, score) in enumerate(iterate_keypoints(face_points)):
            if score <= 0:
                continue
            if landmark_id >= FACE_KEYPOINT_COUNT:
                break
            face.append({
                "schema_version": WHISPERX_SCHEMA_VERSION, "video_id": video_id,
                "frame_number": frame_number, "timestamp": timestamp,
                "detection_index": detection_index, "landmark_id": landmark_id,
                "x": x, "y": y, "confidence": score,
            })
    return {"body": body, "hands": hands, "face": face}


def iterate_keypoints(flat: Sequence[float]) -> Iterable[tuple[float, float, float]]:
    """Yield ``(x, y, score)`` triples from an OpenPose flat array."""
    for index in range(0, len(flat) - 2, 3):
        yield float(flat[index]), float(flat[index + 1]), float(flat[index + 2])


def openpose_frame_number(filename: str) -> int | None:
    """``video_000000000042_keypoints.json`` → ``42``."""
    match = re.search(r"_(\d{8,})_keypoints\.json$", filename)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d{6,})", filename)
    return int(match.group(1)) if match else None


# ------------------------------------------------------------------- numeric

def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _duration(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return round(max(end - start, 0.0), 6)
