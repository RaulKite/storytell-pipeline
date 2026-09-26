"""The pure coordinate transform, checked against `dfMaker`'s own masking and basis rules.

This is the maths half of §20.4 with no pipeline around it: no stage, no Parquet, no
fixture join. It exists as its own file because the two halves fail for different reasons
-- a wrong perpendicular sign or a per-point mask is a maths defect, a stale output or a
missing basis column is a stage defect -- and because the maths is reviewable on its own
(~120 lines) where the stage plus its corpus join is not.

The reference is CRAN multimolang 0.1.1, and the comparison against its output is a
separate test file (it needs the stage and the corpus). What lives here are the rules that
comparison cannot see, because this corpus contains no frame that would tell them apart.
The half-coordinate test below is the example: no keypoint in the corpus is half-missing,
so the reference agrees with BOTH masking rules here, and only the rule written down can
keep the two apart on the next corpus.
"""

from __future__ import annotations

import pytest

from multimodal_pipeline.pose_normalize import (
    BASIS_DEGENERATE,
    BASIS_MISSING_JOINT,
    BASIS_NON_FINITE,
    BASIS_OK,
    VALUE_BASIS_UNUSABLE,
    mask_coordinate,
    mask_point,
    normalize_point,
    person_frame_basis,
    value_state,
)

# ---------------------------------------------------------------- pure transform


def basis_of(masked: dict[int, tuple[float | None, float | None]]):
    """A basis from hand-built masked keypoints, keyed by BODY_25 index."""
    return person_frame_basis(masked, origin_id=8, basis_id=1,
                              origin_name="MidHip", basis_name="Neck")


class TestMaskingIsPerCoordinate:
    """dfMaker masks ``== 0`` coordinate-wise, and that is the only rule it has."""

    @pytest.mark.parametrize("value,expected", [
        (0.0, None), (0, None), (None, None), (float("nan"), None),
        (float("inf"), None), (float("-inf"), None),
        (40.0, 40.0), (-3.5, -3.5), (1e-9, 1e-9),
    ])
    def test_zero_is_absent_and_everything_else_is_a_position(self, value, expected):
        got = mask_coordinate(value)
        assert got == expected or (got is None and expected is None)

    def test_a_half_measured_point_keeps_the_coordinate_it_has(self):
        # The rule a point-wise mask would get wrong: (0, 40) is "the y is known, the x is
        # not", and the reference keeps 40. Point-wise masking agrees on this corpus only
        # because OpenPose never emitted a half-zero keypoint here.
        assert mask_point(0.0, 40.0) == (None, 40.0)
        assert mask_point(12.0, 0.0) == (12.0, None)
        assert mask_point(0.0, 0.0) == (None, None)
        assert mask_point(12.0, 40.0) == (12.0, 40.0)

    def test_a_non_finite_coordinate_is_not_a_coordinate(self):
        # NaN compares false against 0, so a `value == 0` test alone would let it through,
        # and one NaN in a basis joint turns *every* coordinate of that person-frame into
        # NaN — which is neither null (absence, on this table) nor a number anyone can read.
        assert mask_coordinate(float("nan")) is None
        assert mask_coordinate(float("inf")) is None


class TestBasisStates:
    def test_two_usable_joints_build_a_frame_and_say_what_it_is(self):
        basis = basis_of({8: (100.0, 300.0), 1: (100.0, 100.0)})
        assert basis.state == BASIS_OK
        assert basis.usable is True
        assert basis.origin == (100.0, 300.0)
        assert basis.vi == (0.0, -200.0)
        assert basis.vj == (-200.0, -0.0)
        # vi x vj with vj = (vi.y, -vi.x) is -|vi|^2, so the determinant is negative: the
        # sign is part of what makes the second axis point where the reference's points.
        # The detail carries the numbers, because it is the column a human reads first.
        assert "MidHip" in basis.detail and "200" in basis.detail

    def test_a_missing_origin_joint_is_named_not_guessed(self):
        basis = basis_of({1: (100.0, 100.0)})
        assert basis.state == BASIS_MISSING_JOINT
        assert "MidHip" in basis.detail
        assert "has no usable coordinate" in basis.detail

    def test_a_missing_basis_joint_is_named_too(self):
        basis = basis_of({8: (100.0, 300.0)})
        assert basis.state == BASIS_MISSING_JOINT
        assert "Neck" in basis.detail

    def test_a_half_measured_joint_counts_as_missing(self):
        # The frame needs both components of both joints; a masked x is not "x = 0".
        basis = basis_of({8: (None, 300.0), 1: (100.0, 100.0)})
        assert basis.state == BASIS_MISSING_JOINT
        assert "half measured" in basis.detail

    def test_two_joints_at_the_same_pixel_are_degenerate_not_missing(self):
        # A different cause, and a different fix: the joints were measured and coincide, so
        # the determinant is zero. Reporting it as "missing" would send a reader to look for
        # a dropped row.
        basis = basis_of({8: (64.0, 128.0), 1: (64.0, 128.0)})
        assert basis.state == BASIS_DEGENERATE
        assert "determinant" in basis.detail
        assert basis.usable is False

    def test_a_nan_basis_joint_is_refused_before_anything_is_divided(self):
        # The stage masks before it builds a basis, so this is the real path: a NaN Neck
        # becomes a joint with no usable coordinate, and the person-frame gets a named
        # reason instead of a table full of NaN.
        basis = basis_of({8: mask_point(100.0, 300.0), 1: mask_point(100.0, float("nan"))})
        assert basis.state == BASIS_MISSING_JOINT
        assert "Neck" in basis.detail

    def test_finite_but_absurd_coordinates_are_named_not_divided_by(self):
        """Native review R3-numeric-conversion-overflow / R3-derived-nonfinite-basis.

        `mask_coordinate` refuses a coordinate that is itself NaN or infinite, but two
        *finite* doubles still overflow the arithmetic built from them. Before the fix this
        returned basis_ok with denominator=-inf, and normalize_point then produced
        (nan, nan) labelled `normalized` — a coordinate that is neither null (which means
        absence on this table) nor a number, under the status that means "measured".
        """
        basis = basis_of({8: mask_point(1e308, -1e308), 1: mask_point(1e308, 1e308)})
        assert basis.state == BASIS_NON_FINITE
        assert basis.usable is False
        assert "not" in basis.detail and "pixel positions" in basis.detail
        assert normalize_point(basis, mask_point(1e308, 5.0)) is None
        # The point of naming it: a reader can tell this apart from a body that was seen.
        assert value_state(basis, mask_point(1e308, 5.0), None) == VALUE_BASIS_UNUSABLE

    def test_an_overflowing_basis_vector_length_does_not_kill_the_stage(self):
        """The second symptom, which was worse: the detail string itself raised.

        ``(vi[0] ** 2 + vi[1] ** 2) ** 0.5`` overflows on ``1e200 ** 2`` and Python raises
        OverflowError where IEEE division would have produced inf. One absurd row took the
        whole stage down mid-run instead of producing one labelled row.
        """
        basis = basis_of({8: mask_point(1.0, 1.0), 1: mask_point(1e200, 2.0)})
        assert basis.state == BASIS_NON_FINITE

    def test_a_real_pixel_basis_still_reports_the_length_the_reference_used(self):
        # Guard for the hypot swap: at pixel scale hypot and the power form are the same
        # string (all 2661 corpus bases compared under %g, zero disagreements), so the
        # number a reader checks against R must not have moved.
        basis = basis_of({8: mask_point(1.0, 1.0), 1: mask_point(144.7, -57.2)})
        assert basis.state == BASIS_OK
        assert "of length 155.038 px" in basis.detail


class TestNormalizePoint:
    @staticmethod
    def frame():
        # Origin (100, 300), Neck (100, 100): vi = (0, -200), vj = (-200, 0),
        # den = 0*0 - (-200)(-200) = -40000.
        return basis_of({8: (100.0, 300.0), 1: (100.0, 100.0)})

    def test_the_basis_point_is_one_along_the_first_axis_and_the_origin_is_zero(self):
        basis = self.frame()
        assert normalize_point(basis, (100.0, 300.0)) == pytest.approx((0.0, 0.0))
        assert normalize_point(basis, (100.0, 100.0)) == pytest.approx((1.0, 0.0))

    def test_a_joint_beside_the_origin_scales_by_the_basis_length(self):
        basis = self.frame()
        # A wrist 60 px to the right of the hip line, one basis length (200 px) away in
        # y: the basis axis is unchanged and the perpendicular axis carries 60/200 = 0.3.
        # The sign is negative because vj = (vi.y, -vi.x) = (-200, 0) points *left*, so the
        # coordinate of a point to the right is -0.3 — which is exactly what the reference
        # emits for kabc frame 0's LShoulder (x = 513, to the right of the hip, ny =
        # -0.405284064507879).
        assert normalize_point(basis, (160.0, 300.0)) == pytest.approx((0.0, -0.3))
        assert normalize_point(basis, (160.0, 100.0)) == pytest.approx((1.0, -0.3))
        assert normalize_point(basis, (40.0, 300.0)) == pytest.approx((0.0, 0.3))

    def test_the_sign_of_the_second_axis_is_the_reference_sign(self):
        """The sign check, expressed without any fixture.

        ``vj = (vi.y, -vi.x)`` points left when ``vi`` points up the image, so a joint to
        the *right* of the hip->neck axis is negative on the second axis — and that is what
        the reference emits (kabc frame 0, LShoulder at x = 513, ny = -0.405284064507879).
        Taking the other perpendicular flips every one of those signs while leaving the
        origin and the basis point untouched, which is why this is checked on an off-axis
        joint and not on the neck.
        """
        basis = self.frame()
        assert normalize_point(basis, (160.0, 200.0))[1] < 0
        assert normalize_point(basis, (40.0, 200.0))[1] > 0
        # The first axis is unaffected by the reflection, which is the trap: a spot check
        # on "is the neck at 1.0?" passes either way.
        assert normalize_point(basis, (40.0, 200.0))[0] == pytest.approx(0.5)
        assert normalize_point(basis, (160.0, 200.0))[0] == pytest.approx(0.5)

    def test_a_point_with_no_coordinate_cannot_be_placed(self):
        basis = self.frame()
        assert normalize_point(basis, (None, 200.0)) is None
        assert normalize_point(basis, (100.0, None)) is None

    def test_an_unusable_frame_places_nothing(self):
        missing = basis_of({8: (None, None), 1: (100.0, 100.0)})
        assert normalize_point(missing, (150.0, 250.0)) is None
        degenerate = basis_of({8: (1.0, 1.0), 1: (1.0, 1.0)})
        assert normalize_point(degenerate, (150.0, 250.0)) is None

    def test_the_coordinates_are_not_rounded(self):
        # Rounding here would make the reference comparison a claim about a quantisation
        # step rather than about the algebra, and would lose the 1e-14 agreement. This frame
        # has determinant -32500, so 9500/32500 does not terminate: a rounded value would be
        # a different number, not a shorter spelling of the same one.
        basis = basis_of({8: (100.0, 300.0), 1: (110.0, 120.0)})
        x, y = normalize_point(basis, (150.0, 250.0))
        assert x == pytest.approx(0.2923076923076923, abs=1e-15)
        assert x != round(x, 6)
        assert y != round(y, 6)


# ---------------------------------------------------------------- stage behaviour
