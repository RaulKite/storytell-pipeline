"""Re-express BODY_25 keypoints in a body-centred frame instead of pixels (§20.4, T14).

Pure maths, no Stage class, no pyarrow: the same split ``fusion.py`` / ``stages/
speaker_fusion.py`` use, so the algebra is testable without the pipeline and the
stage stays a thin reader/writer.

Why this is a *new table* and not a column on ``pose/body.parquet``: the pixel
coordinates are the measured quantity and every dataset already produced joins on
them. Normalised coordinates are a derived interpretation — a change of basis whose
numbers mean nothing without the triple that produced them — so they live beside the
pixels in ``pose/normalized.parquet`` and the pixel table is never rewritten. That is
also why the raw JSON in ``pose/raw/`` stays untouched: the pipeline's rule is that a
later decision must be recomputable without re-running OpenPose.

The transform is the linear-transformation branch of ``dfMaker()`` from CRAN
``multimolang`` 0.1.1, read out of the package and validated against it. The committed
fixtures are the reference's own output for two processed clips; ``tests/unit/
test_pose_normalized.py`` runs this stage over the real ``pose/body.parquet`` and compares,
agreeing on all 1720 shared keypoints with a worst error of 8.4e-15. The algebra, and the
two places a naive reimplementation goes wrong:

    pts[k]  = (None if x == 0 else x, None if y == 0 else y)   # PER COORDINATE
    origin  = pts[O],  p_i = pts[I]
    vi      = p_i - origin
    vj      = (vi.y, -vi.x)          # dfMaker's i == j branch, NOT (-vi.y, vi.x)
    den     = vi.x * vj.y - vj.x * vi.y
    nx      = (r.x * vj.y - vj.x * r.y) / den      for r = p - origin
    ny      = (vi.x * r.y - r.x * vi.y) / den

* The sign of the perpendicular is load-bearing. ``dfMaker``'s ``fast_scaling`` path
  uses the *opposite* perpendicular ``(-vi.y, vi.x)``; its linear-transformation path
  — the one that rotates — uses ``(vi.y, -vi.x)``. Copy the wrong one and the table
  still looks plausible, because the error is a reflection about the basis axis: it
  scales with distance from the origin, so the neck agrees and the feet are badly
  wrong. ``test_a_reflected_perpendicular_is_not_the_reference`` kills that mutation.
* Absence is masked **per coordinate**, not per point: a keypoint at ``x = 0, y = 40``
  keeps ``y`` and loses ``x``. Point-wise masking agrees with the reference wherever
  OpenPose never emits a half-zero keypoint, so the difference is invisible on this
  corpus and would bite on the next one. ``mask_point`` is the only place that choice
  is made.

The frame itself is a decision, not a parameter filled in later (§20.4 asks "sternum?
pelvis? neck?"). ``MidHip -> Neck`` is the default because it is the longest two-point
torso segment BODY_25 offers, both endpoints sit in the top availability band on this
corpus, and the divisor is what decides stability: dfMaker's own default
(``Neck -> LShoulder``) has a 17.7 px median basis length here, which turns a
half-pixel OpenPose jitter into a ~0.03 swing and puts a wrist at p95 = 556 — a
"normalised" coordinate *less* stable than the pixel it came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

SCHEMA_VERSION = "1.0"

# ------------------------------------------------------------------ vocabularies

#: The basis of one person-frame exists or it does not, and "it does not" has two
#: causes that a consumer must not be able to collapse. This is the ``face_status``
#: lesson from §17 applied to pose: absence gets a *name*, never a zero and never a
#: silently dropped row, because a zero is a measurement and a dropped row is a
#: person who stopped existing.
BASIS_OK = "basis_ok"
#: One of the two joints that defines the frame has no row in ``pose/body.parquet``
#: for this person-frame, or has a row whose x or y was masked away as a zero
#: coordinate. Nothing about the *body* is being claimed — the joints needed to build
#: a frame were never measured.
BASIS_MISSING_JOINT = "basis_missing_joint"
#: Both joints were measured and the basis is still unusable: ``vi`` is the zero
#: vector, so the determinant is 0 and dividing by it would invent coordinates. With
#: the perpendicular as the second axis this means MidHip and Neck landed on the same
#: pixel.
BASIS_DEGENERATE = "basis_degenerate"
#: Both joints hold a number and those numbers are not pixel positions: the basis
#: vector or its determinant overflowed to infinity. ``mask_coordinate`` already
#: refuses a coordinate that is itself NaN or infinite, but two *finite* doubles can
#: still overflow the arithmetic built from them, and a basis whose denominator is
#: ``-inf`` produces ``nan`` coordinates that are neither null (which means absence on
#: this table) nor a number a reader can use. Named, not collapsed into
#: ``basis_degenerate``: that state says the two joints *coincide*, which is a claim
#: about a body, and this one is a claim about the bytes.
BASIS_NON_FINITE = "basis_non_finite"

#: Closed vocabulary — ``validate`` treats a fifth value as a defect, not a variant.
BASIS_STATES: tuple[str, ...] = (BASIS_OK, BASIS_MISSING_JOINT, BASIS_DEGENERATE,
                                BASIS_NON_FINITE)

#: Per-keypoint state, which is a different question from the person-frame's: a joint
#: can be perfectly visible in a frame whose basis is unusable, and a basis can be
#: fine while this particular joint has no usable coordinate.
VALUE_NORMALIZED = "normalized"
#: This keypoint's own x or y was a zero coordinate, so there is no point to move.
VALUE_NO_COORDINATE = "no_coordinate"
#: The joint was measured, but no frame of reference was available for this
#: person-frame. Saying so is the difference between "we could not locate the wrist"
#: and "we could not say where the wrist is relative to the body".
VALUE_BASIS_UNUSABLE = "basis_unusable"

VALUE_STATES: tuple[str, ...] = (VALUE_NORMALIZED, VALUE_NO_COORDINATE,
                                 VALUE_BASIS_UNUSABLE)

# ------------------------------------------------------------------- the default

#: The name of the second axis when it is derived from the first instead of read from
#: a third joint — dfMaker's ``i_point_index == j_point_index`` branch.
SECOND_AXIS_PERPENDICULAR = "perpendicular"

#: ``transformation_coords = c(type, origin, i, j)`` in dfMaker's own order, with
#: ``j == i`` so the second axis is the perpendicular of ``MidHip -> Neck``. That
#: branch is the one measured against the reference, so it is the only one offered.
DEFAULT_TRANSFORMATION: tuple[str, str, str] = ("MidHip", "Neck", SECOND_AXIS_PERPENDICULAR)


# ---------------------------------------------------------------------- masking

def mask_coordinate(value: Any) -> float | None:
    """One coordinate → a float, or None when it is not a position.

    ``dfMaker`` runs ``m[,1:2][m[,1:2] == 0] <- NA`` over the coordinate columns before it
    emits anything, so a 0 is not a position on this table's axes: it is the tool's marker
    for "no coordinate here". Keeping it as a number would place a joint at the origin of
    the image, which is a confident lie about a body part.

    Non-finite values are refused by the same door rather than by a second rule: ``NaN``
    compares false against 0 so a bare ``== 0`` test would let it through, and one NaN or
    infinity in a basis joint turns every coordinate of that person-frame into NaN — a
    value that is neither null (which means absence here) nor a number a reader can use.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number == 0.0:
        return None
    return number


def mask_point(x: Any, y: Any) -> tuple[float | None, float | None]:
    """``(x, y)`` masked PER COORDINATE, which is what the reference does.

    Deliberately not "both or neither": a keypoint at ``x = 0, y = 40`` keeps its
    ``y``. Masking per point would agree with the reference on every frame of this
    corpus (measured: no half-zero keypoint in either fixture video) and disagree
    silently on any future one, which is the worst kind of agreement to rely on.
    """
    return mask_coordinate(x), mask_coordinate(y)


# ------------------------------------------------------------------- the basis

@dataclass(frozen=True)
class Basis:
    """The frame one person-frame can (or cannot) be expressed in.

    ``vi``/``vj``/``denominator`` are kept so the coordinates are reproducible from
    the row alone, and so ``basis_detail`` can quote the numbers it decided on rather
    than assert a cause.
    """

    state: str
    detail: str
    origin: tuple[float, float] | None = None
    vi: tuple[float, float] | None = None
    vj: tuple[float, float] | None = None
    denominator: float | None = None

    @property
    def usable(self) -> bool:
        return self.state == BASIS_OK


def missing_joint_basis(origin_name: str, basis_name: str, missing: Sequence[str],
                        why: str) -> Basis:
    """A frame that cannot be built because a defining joint was not measured."""
    return Basis(
        state=BASIS_MISSING_JOINT,
        detail=f"{', '.join(missing)} {why}, so the {origin_name}->{basis_name} frame "
               f"cannot be built for this person-frame",
    )


def person_frame_basis(masked: Mapping[int, tuple[float | None, float | None]], *,
                       origin_id: int, basis_id: int, origin_name: str,
                       basis_name: str) -> Basis:
    """Decide one person-frame's basis from its already-masked keypoints.

    ``masked`` maps ``keypoint_id -> (x, y)`` with the absent coordinates already
    replaced by None, i.e. the output of :func:`mask_point`. Nothing here reads
    Parquet, so the three states can be tested against hand-built dictionaries.
    """
    origin = masked.get(origin_id)
    basis_point = masked.get(basis_id)
    if origin is None or basis_point is None:
        absent = [name for name, value in ((origin_name, origin), (basis_name, basis_point))
                  if value is None]
        return missing_joint_basis(origin_name, basis_name, absent,
                                  "has no usable coordinate in this person-frame")
    if origin[0] is None or origin[1] is None or basis_point[0] is None or basis_point[1] is None:
        # mask_point can leave one half of a point standing; a frame needs both.
        half = [name for name, value in ((origin_name, origin), (basis_name, basis_point))
                if None in value]
        return missing_joint_basis(origin_name, basis_name, half,
                                  "is only half measured (one of its two coordinates is a "
                                  "zero, which the reference masks to NA)")

    vi = (basis_point[0] - origin[0], basis_point[1] - origin[1])
    # The i == j branch of dfMaker's linear transformation. The other perpendicular,
    # (-vi.y, vi.x), is what its fast_scaling path uses and it reflects the table
    # about the basis axis; see this module's docstring.
    vj = (vi[1], -vi[0])
    denominator = vi[0] * vj[1] - vj[0] * vi[1]
    if denominator == 0:
        return Basis(
            state=BASIS_DEGENERATE,
            detail=f"{origin_name} and {basis_name} are both measured but coincide at the "
                   f"same pixel, so the basis vector ({vi[0]:g}, {vi[1]:g}) has zero "
                   f"determinant: dividing by it would invent coordinates",
        )
    # ``math.hypot`` rather than ``(vi[0] ** 2 + vi[1] ** 2) ** 0.5``, and it is not a style
    # choice: the power form raises OverflowError on ``1e200 ** 2``, which took the whole
    # stage down on one absurd row (reproduced before this fix). hypot is scale-safe, and on
    # the corpus's real numbers it is the same string — all 2661 MidHip->Neck bases of the
    # seven processed videos compared under ``%g``, zero disagreements — so changing the
    # length in ``basis_detail`` does not move the number a reader checks against R.
    length = math.hypot(vi[0], vi[1])
    if not math.isfinite(length) or not math.isfinite(denominator):
        return Basis(
            state=BASIS_NON_FINITE,
            detail=f"{origin_name} and {basis_name} both hold a number but they are not "
                   f"pixel positions: the basis vector ({vi[0]:g}, {vi[1]:g}) has length "
                   f"{length:g} and determinant {denominator:g}, so every coordinate built "
                   f"from them would be infinite or NaN",
        )
    return Basis(
        state=BASIS_OK,
        detail=f"origin {origin_name} at ({origin[0]:g}, {origin[1]:g}), basis vector "
               f"{origin_name}->{basis_name} = ({vi[0]:g}, {vi[1]:g}) of length "
               f"{length:g} px, second axis = its perpendicular",
        origin=(origin[0], origin[1]), vi=vi, vj=vj, denominator=denominator,
    )


def normalize_point(basis: Basis, point: tuple[float | None, float | None]
                    ) -> tuple[float, float] | None:
    """One masked keypoint into the basis' frame, or None when it cannot be placed.

    The formulas are the reference's, transcribed rather than simplified: the
    simplification is algebraically obvious (the denominator is ``-|vi|^2``) but the
    version under test is the version that was compared against R, and rewriting it
    here would move the thing the fixtures validate.
    """
    if not basis.usable or basis.origin is None:
        return None
    x, y = point
    if x is None or y is None:
        return None
    radius = (x - basis.origin[0], y - basis.origin[1])
    vi, vj, denominator = basis.vi, basis.vj, basis.denominator
    return ((radius[0] * vj[1] - vj[0] * radius[1]) / denominator,
            (vi[0] * radius[1] - radius[0] * vi[1]) / denominator)


def value_state(basis: Basis, point: tuple[float | None, float | None],
                coordinates: tuple[float, float] | None) -> str:
    """Which of the three per-keypoint states a row is in, most specific first.

    Own-coordinate absence wins over an unusable basis: a joint with no coordinate
    could not have been placed even in a frame that was fine, and "we never saw the
    wrist" is the more useful thing to know than "we had no frame".
    """
    if coordinates is not None:
        return VALUE_NORMALIZED
    if None in point:
        return VALUE_NO_COORDINATE
    return VALUE_BASIS_UNUSABLE


# ---------------------------------------------------------------- row assembly

PersonKey = tuple[int, int]


def frame_key(row: Mapping[str, Any]) -> PersonKey:
    """The person-frame a body row belongs to.

    ``detection_index`` is frame-local (OpenPose only gives a stable id with tracking
    on), so this is a per-frame person, not a person across the video — the same
    reading the pixel table already takes.
    """
    return int(row["frame_number"]), int(row["detection_index"])


def person_frame_bases(
    rows: Iterable[Mapping[str, Any]], *, origin_id: int, basis_id: int,
    origin_name: str, basis_name: str
) -> dict[PersonKey, Basis]:
    """One :class:`Basis` per person-frame, in one streaming pass.

    Keyed rather than positional because a basis needs *both* defining joints, which
    arrive as two rows of the same group; and keyed by (frame, detection) rather than
    by an open run of rows so a table that is not perfectly grouped still gets one
    answer per person-frame.
    """
    keypoints: dict[PersonKey, dict[int, tuple[float | None, float | None]]] = {}
    for row in rows:
        keypoints.setdefault(frame_key(row), {})[int(row["keypoint_id"])] = mask_point(
            row.get("x"), row.get("y"))
    return {key: person_frame_basis(masked, origin_id=origin_id, basis_id=basis_id,
                                    origin_name=origin_name, basis_name=basis_name)
            for key, masked in keypoints.items()}


def normalized_rows(
    rows: Iterable[Mapping[str, Any]], bases: Mapping[PersonKey, Basis], *,
    video_id: str, origin_name: str, basis_name: str, second_axis: str
) -> Iterator[dict[str, Any]]:
    """One normalized row per body row, in the order the body table holds them.

    Row-for-row with ``pose/body.parquet`` on purpose: a keypoint that OpenPose never
    found has no row *there*, so it gets none here either, and the person it belongs
    to is still represented by that person-frame's other rows plus its ``basis_state``.
    Dropping a whole person-frame instead would make a partially visible body vanish.
    """
    for row in rows:
        basis = bases.get(frame_key(row))
        if basis is None:  # unreachable for a table read twice; never a guess
            basis = missing_joint_basis(origin_name, basis_name, ["This person-frame"],
                                        "was not seen while the bases were computed")
        point = mask_point(row.get("x"), row.get("y"))
        coordinates = normalize_point(basis, point)
        yield {
            "schema_version": SCHEMA_VERSION,
            "video_id": video_id,
            "frame_number": int(row["frame_number"]),
            "timestamp": row.get("timestamp"),
            "detection_index": int(row["detection_index"]),
            "keypoint_id": int(row["keypoint_id"]),
            "keypoint_name": row.get("keypoint_name"),
            "origin_keypoint_name": origin_name,
            "basis_keypoint_name": basis_name,
            "second_axis": second_axis,
            "basis_state": basis.state,
            "basis_detail": basis.detail,
            # Null means "no coordinate in this frame", and `value_status` says which
            # of the two reasons it was: never measured, or no frame to measure it in.
            "x_norm": None if coordinates is None else coordinates[0],
            "y_norm": None if coordinates is None else coordinates[1],
            "value_status": value_state(basis, point, coordinates),
        }
