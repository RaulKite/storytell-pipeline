"""Explicit Parquet schemas for every normalised table.

Column order and dtype are pinned here so downstream consumers (PyArrow,
Polars, DuckDB, pandas) always see the same types, and so ``validate`` can
check a table's structure instead of trusting the writer.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

# --- speech ---------------------------------------------------------------

SEGMENTS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("language", pa.string()),
        ("speaker_id", pa.string()),
        ("text", pa.string()),
        ("confidence", pa.float64()),
        ("speaker_overlap_seconds", pa.float64()),
        ("speaker_overlap_ratio", pa.float64()),
        ("speaker_assignment_method", pa.string()),
    ]
)

WORDS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("word_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("speaker_id", pa.string()),
        ("word", pa.string()),
        ("confidence", pa.float64()),
        ("alignment_status", pa.string()),
        ("character_start", pa.int64()),
        ("character_end", pa.int64()),
        ("speaker_overlap_seconds", pa.float64()),
        ("speaker_overlap_ratio", pa.float64()),
        ("speaker_assignment_method", pa.string()),
    ]
)

SPEAKER_TURNS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("turn_id", pa.string()),
        ("speaker_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("diarization_type", pa.string()),
    ]
)

# --- second-engine diarization (NVIDIA Nemotron 3) -------------------------
#
# Deliberately a separate schema rather than SPEAKER_TURNS_SCHEMA with an extra column.
# Two properties make sharing impossible:
#   * `speaker_id` here is Nemotron's own arrival-ordered namespace (``speaker_0``), not
#     pyannote's (``SPEAKER_00``). Two id spaces in one column invites a join that means
#     nothing, so they live in different files and the README says they must not be joined.
#   * segments from different channels overlap. A per-instant exclusive table cannot hold
#     that, and collapsing it would throw away the one thing this model is for.
SPEAKER_TURNS_NEMOTRON_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("turn_id", pa.string()),
        ("speaker_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("diarization_type", pa.string()),
        # Seconds of this segment that overlap a segment of a *different* speaker. Kept as
        # a number so the two engines are comparable with ordinary parquet arithmetic
        # ("how much overlapping speech did each one claim") instead of a list column that
        # every consumer would have to explode.
        ("overlap_s", pa.float64()),
    ]
)

# --- audio/visual agreement (diarization turns x active speaker) -----------
#
# One row per diarization turn of ONE engine, with what the 25 FPS active-speaker table
# measured inside that turn's window. This is a *second* diarization result, kept beside
# speaker_turns / speaker_turns_nemotron rather than replacing either: the existing tables
# are the input to speaker_assignment and to every dataset already produced, and a fused
# label quietly substituted for an audio turn would relabel all of them.
#
# Why a state column instead of one speaker label: the diarizer answers *when does a voice
# speak* and TalkNet answers *which visible face is talking*. They disagree exactly where it
# matters — off-screen narrator, cutaway, two faces one voice, silent moving mouth — and
# flattening that to a single label would destroy the only information the comparison
# produces. `agreement` is a closed five-value vocabulary (see fusion.AGREEMENT_STATES).
#
# Why the three count columns instead of one ratio: `frames_in_turn` counts every dense ASD
# row in the window *including* the no_face rows, so 0 means "nothing was measured here"
# while frames_in_turn > 0 with face_frames_in_turn == 0 means "measured: nobody was on
# screen". Those two read identically in a single-ratio table, which is the mistake
# `frame_reason` exists to fix upstream. The invariant the stage validates is
# face_active_frames <= face_frames_in_turn <= frames_in_turn.
#
# Why `engine` is not a join key: pyannote's `SPEAKER_00` and Nemotron's arrival-ordered
# `speaker_0` (see the note on SPEAKER_TURNS_NEMOTRON_SCHEMA) are unrelated clusters over
# unrelated channels, so identical digits name different people. `engine` names the
# namespace a row's speaker_id came from, each engine is written to its own file, and a
# consumer that joined pyannote rows to Nemotron rows on speaker_id would get nonsense.
# TalkNet's track_id is a third id space again and is never equated with a speaker id.
SPEAKER_FUSION_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        # Which diarizer produced the turn — and therefore which speaker-id namespace
        # `speaker_id` belongs to. Not a cross-engine join key.
        ("engine", pa.string()),
        ("turn_id", pa.string()),
        ("speaker_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("diarization_type", pa.string()),
        # Carried from the turn table. Nullable because a pyannote turn has no such
        # measurement: a 0.0 there would read as "measured, no overlap".
        ("overlap_s", pa.float64()),
        # TalkNet track with the strongest claim on this turn, or null when no track in
        # the window was ever flagged active. An ASD track id, not a speaker id.
        ("face_track_id", pa.int64()),
        # Frames of the winning track flagged active inside this window.
        ("face_active_frames", pa.int64()),
        # Frames in the window where any face was located at all.
        ("face_frames_in_turn", pa.int64()),
        # Dense ASD rows in the window, no_face rows included: the difference between
        # "nobody visible" and "not measured".
        ("frames_in_turn", pa.int64()),
        ("face_mean_score", pa.float64()),
        ("face_score_max", pa.float64()),
        # The verdict, from the closed vocabulary in fusion.AGREEMENT_STATES.
        ("agreement", pa.string()),
        # The measured numbers in words — the column a human reads first, so it names the
        # threshold the track cleared or missed and any tie that was broken.
        ("agreement_detail", pa.string()),
    ]
)

# --- translation ----------------------------------------------------------

TRANSLATION_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("speaker_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("source_language", pa.string()),
        ("source_text", pa.string()),
        ("english_text", pa.string()),
        ("translation_model", pa.string()),
        ("translation_prompt_version", pa.string()),
    ]
)

# --- linguistic -----------------------------------------------------------

TOKENS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("variant", pa.string()),  # source | english
        ("segment_id", pa.string()),
        ("sentence_id", pa.string()),
        ("token_id", pa.string()),
        ("token_index", pa.int64()),
        ("speaker_id", pa.string()),
        ("text", pa.string()),
        ("lower", pa.string()),
        ("lemma", pa.string()),
        ("pos", pa.string()),
        ("tag", pa.string()),
        ("morph", pa.string()),
        ("dep", pa.string()),
        ("head_token_id", pa.string()),
        ("head_text", pa.string()),
        ("head_pos", pa.string()),
        ("ent_type", pa.string()),
        ("is_alpha", pa.bool_()),
        ("is_stop", pa.bool_()),
        ("is_digit", pa.bool_()),
        ("like_num", pa.bool_()),
        ("shape", pa.string()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("segment_start_time", pa.float64()),
        ("segment_end_time", pa.float64()),
        ("token_start_time", pa.float64()),
        ("token_end_time", pa.float64()),
        ("timestamp_alignment_status", pa.string()),
        ("timestamp_alignment_confidence", pa.float64()),
    ]
)

SENTENCES_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("variant", pa.string()),
        ("segment_id", pa.string()),
        ("sentence_id", pa.string()),
        ("sentence_index", pa.int64()),
        ("speaker_id", pa.string()),
        ("text", pa.string()),
        ("token_count", pa.int64()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("segment_start_time", pa.float64()),
        ("segment_end_time", pa.float64()),
    ]
)

# --- acoustic -------------------------------------------------------------

ACOUSTIC_FRAMES_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("timestamp", pa.float64()),
        ("f0_hz", pa.float64()),
        ("intensity_db", pa.float64()),
        ("voiced", pa.bool_()),
        ("f1_hz", pa.float64()),
        ("f2_hz", pa.float64()),
        ("f3_hz", pa.float64()),
    ]
)

ACOUSTIC_SEGMENTS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("segment_id", pa.string()),
        ("speaker_id", pa.string()),
        ("start_time", pa.float64()),
        ("end_time", pa.float64()),
        ("duration", pa.float64()),
        ("voiced_ratio", pa.float64()),
        ("f0_mean", pa.float64()),
        ("f0_median", pa.float64()),
        ("f0_min", pa.float64()),
        ("f0_max", pa.float64()),
        ("f0_std", pa.float64()),
        ("intensity_mean", pa.float64()),
        ("intensity_median", pa.float64()),
        ("intensity_min", pa.float64()),
        ("intensity_max", pa.float64()),
        ("intensity_std", pa.float64()),
        ("f1_mean", pa.float64()),
        ("f2_mean", pa.float64()),
        ("f3_mean", pa.float64()),
        ("pause_count", pa.int64()),
        ("pause_duration", pa.float64()),
        ("pause_ratio", pa.float64()),
    ]
)

# --- active speaker -------------------------------------------------------

# One row per 25 FPS frame of the TalkNet working timeline, including frames with
# no detected face: the stage's whole purpose is a dense per-frame speaker track,
# so a consumer must be able to trust that frame N exists exactly once.
# One row per 25 FPS frame. `face_status` is the difference between an honest empty and
# a silent one: a frame where S3FD tracked a face but TalkNet produced no usable score
# (past the two imputable tail frames, or a non-finite score) used to look exactly like
# a frame with no face -- track_id null, no bbox, no score. It is now its own state, so
# a consumer reading "no face" is not reading "we lost the score". A malformed bbox is
# still dropped rather than invented: a garbage box is not a location.
# `frame_reason` answers a different question to `face_status`: face_status says whether a
# face was located at all, frame_reason says why a row does or does not carry a TalkNet
# score. Four distinct causes produce an unscored row (a non-finite score, a track with no
# scores, a frame past the imputable tail, a non-finite carried tail score) and the table
# cannot tell them apart -- neither the frame's position inside its track nor the track's
# score count survives the worker -- so only the worker can name the cause.
ACTIVE_SPEAKER_FRAMES_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp", pa.float64()),
        ("source_timestamp", pa.float64()),
        ("scene_id", pa.int64()),
        ("track_id", pa.int64()),
        ("face_status", pa.string()),
        ("frame_reason", pa.string()),
        ("x1", pa.float64()),
        ("y1", pa.float64()),
        ("x2", pa.float64()),
        ("y2", pa.float64()),
        ("talknet_score_raw", pa.float64()),
        ("talknet_score", pa.float64()),
        ("score_imputed", pa.bool_()),
        ("is_active_speaker", pa.bool_()),
    ]
)

# One row per TalkNet face track, summarising where it sits on the audio timeline
# so it can be related to diarization turns without re-reading the frames table.
ACTIVE_SPEAKER_TRACKS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("track_id", pa.int64()),
        ("first_timestamp", pa.float64()),
        ("last_timestamp", pa.float64()),
        ("frame_count", pa.int64()),
        ("active_frame_count", pa.int64()),
        ("active_ratio", pa.float64()),
        ("mean_score", pa.float64()),
        ("max_score", pa.float64()),
        ("scenes", pa.list_(pa.int64())),
        ("mean_bbox_area", pa.float64()),
    ]
)

# --- persons (Ultralytics YOLO detection + tracking) -----------------------
#
# Two tables, both answering questions no existing stage answers: how many distinct people
# appear in a video, and when each one is on screen.
#
# WHY `person_id` AND NOT `track_id` — THIS IS THE ONE THING NOT TO "SIMPLIFY".
# ACTIVE_SPEAKER_TRACKS_SCHEMA and ACTIVE_SPEAKER_FRAMES_SCHEMA carry a `track_id`: TalkNet's
# S3FD face-tracker id. A YOLO ByteTracker id and a TalkNet track id are *unrelated*
# integers over unrelated detectors, and §20.2 names that as a constraint, not a nuance:
# both are small integers starting near zero, so a consumer that joins `person_id` to
# `track_id` gets a result that looks perfectly reasonable and means nothing at all. The
# column is therefore named differently rather than reused, the two tables live in different
# directories (`persons/` vs `speaker/`), and the prohibition is stated in the file metadata
# and in the README as well as here.
#
# WHY A PERSON TRACK IS NOT A FACE TRACK.
# `person_id` 3 is a *body* trajectory: one id can cover a frame where the face is turned
# away, occluded, or out of shot entirely, and a single talking head can produce several
# person ids when the camera cuts or the detector blinks. TalkNet's `track_id` 3 is the
# opposite: a face trajectory that says nothing about the body. A person and a face are
# usually the same human being and the tables cannot know that — matching them needs an
# IoU-style spatial join over the two bbox sets, which is an analysis decision, not a key.
#
# WHY THE FRAMES TABLE IS DENSE OVER *DETECTED* FRAMES AND NOT OVER ALL FRAMES.
# A frame with no person produces no row here, unlike the ASD frames table. The difference
# is that ASD must answer "who is speaking in this frame" for every frame, while this table
# answers "where was this person"; the person count and each person's span come from the
# summary table, and `frames_measured` in the raw document says how many frames were looked
# at. That is what makes "zero people detected" distinguishable from "nothing was run": the
# stage skips without a raw document rather than writing an empty table, and a completed run
# always reports the frames it examined.
PERSON_FRAMES_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        # Index into the frames the worker actually read (0-based), after any
        # `frame_stride` subsampling. NOT a source presentation timestamp and NOT
        # comparable to the ASD stage's 25 FPS `frame_number` -- use `timestamp`.
        ("frame_number", pa.int64()),
        # This frame's pts_seconds from source/frame_index.parquet, i.e. the pipeline's
        # one timeline. The column to join other stages on.
        ("timestamp", pa.float64()),
        # The person's id in THIS stage's own namespace. Never equated with TalkNet's
        # `track_id` or with any speaker id -- see the note above the schema.
        ("person_id", pa.int64()),
        ("x1", pa.float64()),
        ("y1", pa.float64()),
        ("x2", pa.float64()),
        ("y2", pa.float64()),
        # Detection confidence for THIS frame's box, in [0, 1]. Per-frame, not per-track:
        # the same person's confidence moves when they turn, so the track table reports a
        # mean and this row keeps the measurement.
        ("confidence", pa.float64()),
        # Tracker's own confidence that this box continues this id. Nullable because not
        # every tracker publishes one and a 0.0 there would read as "measured, no
        # confidence"; `confidence_reason` says which of the two cases a null is.
        ("track_confidence", pa.float64()),
        # Why track_confidence is or is not there. Closed vocabulary (see
        # stages.persons.CONFIDENCE_REASONS) so a reader can branch on it: `tracked` means
        # the tracker reported one, `no_track_confidence` means this tracker does not.
        ("confidence_reason", pa.string()),
        # Box area in source pixels. Kept as a column rather than derived by every reader:
        # "how big in frame" is the number that separates a foreground presenter from a
        # bystander in the back row, and it is the input to any such cut.
        ("bbox_area", pa.float64()),
        # How many distinct person ids the detector reported in this frame, this row's
        # included. Redundant with a group-by over the same table and kept anyway, for the
        # reason `active_ratio` is: "how many people are on screen at once" is the question
        # this stage exists to answer and it should cost one MAX(), not a re-grouping.
        # `validate` recomputes it from the rows and fails if the two ever disagree.
        ("persons_in_frame", pa.int64()),
    ]
)

# One row per person id in one video: the answer to "how many people, and when".
#
# WHY A SPAN AND NOT A SET OF INTERVALS. A tracker id is not guaranteed contiguous --
# ByteTracker keeps a track through short occlusion and can revive an id after a blink -- so
# `frame_count` and the timestamps are counts over the rows that exist, and
# `longest_gap_seconds` reports the biggest hole between consecutive sightings. A reader who
# needs contiguity has that number; a reader who assumes it has a test telling them not to.
PERSON_TRACKS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("person_id", pa.int64()),
        # First and last sighting on the pipeline timeline (seconds).
        ("first_timestamp", pa.float64()),
        ("last_timestamp", pa.float64()),
        # Wall-clock span between the first and last sighting. Longer than `frame_count`
        # implies frames in between where this person was not detected.
        ("duration_seconds", pa.float64()),
        # Rows in the frames table carrying this id.
        ("frame_count", pa.int64()),
        # Share of the video's measured frames this person was detected in, in [0, 1].
        # Denominator is frames *measured* (after any stride), so it is comparable across
        # videos and honest about a subsampled run.
        ("frame_coverage", pa.float64()),
        # Largest gap between two consecutive sightings of this id, in seconds. 0.0 means
        # the id was seen in every measured frame between its endpoints.
        ("longest_gap_seconds", pa.float64()),
        ("mean_confidence", pa.float64()),
        ("max_confidence", pa.float64()),
        ("mean_bbox_area", pa.float64()),
        ("max_bbox_area", pa.float64()),
        # Where this id sits in the ordering by first sighting (0 = earliest). A rank, not
        # an id: it lets a reader ask "the third person to appear" without assuming
        # tracker ids are assigned in first-seen order, which ByteTracker does not promise.
        ("appearance_order", pa.int64()),
    ]
)

# --- pose -----------------------------------------------------------------

BODY_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp", pa.float64()),
        ("detection_index", pa.int64()),
        ("keypoint_id", pa.int64()),
        ("keypoint_name", pa.string()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("confidence", pa.float64()),
    ]
)

# Derived from BODY_SCHEMA, kept as a new table rather than extra columns on it (§20.4).
#
# Why a new table: the pixel coordinates are the measured quantity and every dataset
# already produced joins on them, so writing normalised values into `pose/body.parquet`
# would silently redefine an existing corpus. Normalised coordinates are a *change of
# basis*, and a change of basis means nothing without the triple that produced it, which
# is why the three columns naming the frame travel with every row instead of living only
# in file metadata a reader may never open.
#
# Why two state columns instead of nulls alone: `x_norm`/`y_norm` are null in two
# different situations that a consumer must not be able to collapse — the keypoint was
# never measured, versus the keypoint was measured but the frame it would be expressed in
# could not be built. `basis_state` answers that for the person-frame (see
# pose_normalize.BASIS_STATES, the `face_status` lesson from §17) and `value_status`
# answers it for the joint. Neither is ever encoded as a zero: a zero is a measurement,
# and on this table's axes a zero means "exactly at the hip".
POSE_NORMALIZED_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp", pa.float64()),
        # Frame-local person index, copied from the body table. Not a cross-frame
        # identity unless OpenPose tracking was on — same reading as BODY_SCHEMA.
        ("detection_index", pa.int64()),
        ("keypoint_id", pa.int64()),
        ("keypoint_name", pa.string()),
        # The frame these numbers are expressed in, repeated per row on purpose: one
        # file can hold several configurations' worth of meaning only if the frame
        # travels with the number.
        ("origin_keypoint_name", pa.string()),
        ("basis_keypoint_name", pa.string()),
        # "perpendicular" = the second axis is vi rotated, i.e. dfMaker's i == j branch.
        ("second_axis", pa.string()),
        # Closed vocabulary over the person-frame: basis_ok | basis_missing_joint |
        # basis_degenerate (the two joints coincide) | basis_non_finite (they hold numbers
        # that overflow the basis, so no coordinate could be built from them).
        ("basis_state", pa.string()),
        # The measured numbers behind that state — which joint was missing, or how long
        # the basis vector was. The column a human reads first.
        ("basis_detail", pa.string()),
        # Coordinates in the body-centred frame. Null only when `value_status` says why.
        ("x_norm", pa.float64()),
        ("y_norm", pa.float64()),
        # Closed vocabulary over this keypoint: normalized | no_coordinate | basis_unusable.
        ("value_status", pa.string()),
    ]
)

HANDS_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp", pa.float64()),
        ("detection_index", pa.int64()),
        ("hand", pa.string()),
        ("keypoint_id", pa.int64()),
        ("keypoint_name", pa.string()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("confidence", pa.float64()),
    ]
)

FACE_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("timestamp", pa.float64()),
        ("detection_index", pa.int64()),
        ("landmark_id", pa.int64()),
        ("x", pa.float64()),
        ("y", pa.float64()),
        ("confidence", pa.float64()),
    ]
)

# --- media ----------------------------------------------------------------

FRAME_INDEX_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("video_id", pa.string()),
        ("frame_number", pa.int64()),
        ("pts_seconds", pa.float64()),
    ]
)

TABLE_SCHEMAS: dict[str, pa.schema] = {
    "speech_segments": SEGMENTS_SCHEMA,
    "speech_words": WORDS_SCHEMA,
    "speaker_turns": SPEAKER_TURNS_SCHEMA,
    "speaker_turns_nemotron": SPEAKER_TURNS_NEMOTRON_SCHEMA,
    # Both engines share this schema; the `engine` column says which one a row came from,
    # and each engine is written to its own file so the namespaces stay apart on disk too.
    "speaker_fusion_pyannote": SPEAKER_FUSION_SCHEMA,
    "speaker_fusion_nemotron": SPEAKER_FUSION_SCHEMA,
    "translation_segments": TRANSLATION_SCHEMA,
    "linguistic_source_tokens": TOKENS_SCHEMA,
    "linguistic_source_sentences": SENTENCES_SCHEMA,
    "linguistic_english_tokens": TOKENS_SCHEMA,
    "linguistic_english_sentences": SENTENCES_SCHEMA,
    "acoustic_frames": ACOUSTIC_FRAMES_SCHEMA,
    "acoustic_segments": ACOUSTIC_SEGMENTS_SCHEMA,
    "pose_body": BODY_SCHEMA,
    # Derived from pose_body, in the same directory, so the pixel table and the frame
    # it can be re-expressed in are found together.
    "pose_normalized": POSE_NORMALIZED_SCHEMA,
    "pose_hands": HANDS_SCHEMA,
    "pose_face": FACE_SCHEMA,
    # Persons: own directory, own schemas, own id namespace (see the note above
    # PERSON_FRAMES_SCHEMA). Distinct from active_speaker_tracks on purpose: two tables
    # with a column called `track_id` invite the join §20.2 forbids.
    "person_frames": PERSON_FRAMES_SCHEMA,
    "person_tracks": PERSON_TRACKS_SCHEMA,
    "frame_index": FRAME_INDEX_SCHEMA,
}

# Standard OpenPose BODY_25 landmark names (verified against
# /opt/openpose/src/openpose/pose/poseParameters.cpp POSE_BODY_25_BODY_PARTS).
BODY_25_KEYPOINT_NAMES: tuple[str, ...] = (
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel", "Background",
)

# OpenPose outputs 21 hand keypoints; names follow the official hand model.
HAND_KEYPOINT_NAMES: tuple[str, ...] = (
    "Wrist", "TH1", "TH2", "TH3", "TH4", "Index1", "Index2", "Index3", "Index4",
    "Middle1", "Middle2", "Middle3", "Middle4", "Ring1", "Ring2", "Ring3",
    "Ring4", "Pinky1", "Pinky2", "Pinky3", "Pinky4",
)

FACE_KEYPOINT_COUNT = 70  # OpenPose face model: 70 landmarks


def write_table(path, table: pa.Table, schema: pa.schema, *, extra_metadata: dict[str, Any] | None = None) -> None:
    """Validate against the declared schema, then write with compression.

    Metadata is normalised to strings because Parquet key/value metadata is
    string-valued and pyarrow rejects anything else from inside Cython, with
    ``expected bytes, NoneType found`` and no mention of which key. Optional
    provenance values are genuinely absent sometimes -- ``diarization_type`` is None
    for a video with no speaker turns -- and losing one annotation must not fail a
    stage that has already done its work.
    """
    if extra_metadata:
        merged = {**(schema.metadata or {}), **extra_metadata}
        schema = schema.with_metadata(
            {key: (value if isinstance(value, str) else str(value))
             for key, value in merged.items() if value is not None}
        )
    table = _coerce(table, schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def _coerce(table: pa.Table, schema: pa.schema) -> pa.Table:
    missing = [field.name for field in schema if field.name not in table.column_names]
    for name in missing:
        # pyarrow has append_column (singular); there is no append_columns, so the
        # plural call raised AttributeError and every "optional column absent"
        # write died instead of writing nulls.
        table = table.append_column(schema.field(name), pa.nulls(len(table), type=schema.field(name).type))
    table = table.select([field.name for field in schema])
    return table.cast(schema, safe=False)


class ChunkedParquetWriter:
    """Bounded-memory writer: buffers rows, flushes row groups to one file."""

    def __init__(self, path, schema: pa.schema, *, rows_per_group: int = 200_000,
                 extra_metadata: dict[str, str] | None = None) -> None:
        self.path = path
        self.schema = schema
        self.rows_per_group = rows_per_group
        self.rows: list[dict[str, object]] = []
        self.rows_written = 0
        self._writer: pq.ParquetWriter | None = None
        self._extra_metadata = extra_metadata or {}

    def add(self, row: dict[str, object]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.rows_per_group:
            self.flush()

    def extend(self, rows: Iterable[dict[str, object]]) -> None:
        for row in rows:
            self.add(row)

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        table = _coerce(table, self.schema)
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            metadata = {**(self.schema.metadata or {}), **self._extra_metadata}
            self._writer = pq.ParquetWriter(
                self.path, self.schema.with_metadata(metadata), compression="zstd"
            )
        self._writer.write_table(table)
        self.rows_written += len(self.rows)
        self.rows.clear()

    def close(self) -> int:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif not self.path.exists():
            # Never leave a missing file: an empty table is a valid, honest result.
            write_table(self.path, pa.table({f.name: [] for f in self.schema}, schema=self.schema),
                        self.schema, extra_metadata=self._extra_metadata)
        return self.rows_written


def read_table(path, columns: Sequence[str] | None = None) -> pa.Table:
    """Read a table, optionally projecting columns (cheap on wide tables)."""
    return pq.read_table(path, columns=list(columns) if columns else None)


def table_rows(path) -> int:
    return pq.read_metadata(path).num_rows


def table_columns(path) -> Sequence[str]:
    return [field.name for field in pq.read_schema(path)]


def iter_rows(path, columns: Sequence[str] | None = None) -> Iterator[dict[str, object]]:
    """Stream rows in row-group batches without materialising the whole table."""
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=20_000, columns=list(columns) if columns else None):
        yield from batch.to_pylist()
