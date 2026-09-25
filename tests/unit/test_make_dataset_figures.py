"""Tests for ``scripts/make_dataset_figures.py``.

The script exists so documentation can show what a stage emits, which makes its real
failure mode *a figure that lies*: a renamed column that silently renders an empty strip,
a bone whose endpoint no longer exists so the arm disappears, a synthetic table that
drifted from the schema while the figure still looks fine. Every test here attacks one of
those.

Matplotlib is not a project dependency, so the suite runs without it. The non-drawing
tests — the whole failure-honesty and schema-contract surface — run everywhere, and prove
they run everywhere by replacing the script's plotting seam with one that raises: a
failure path that reached matplotlib would error instead of producing its named message.
The tests that actually put pixels in a file are marked ``requires_matplotlib`` and skip
cleanly when it is absent, the same way the e2e suite skips on a machine without OpenPose:

    uv run --with pytest --with matplotlib pytest tests/unit/test_make_dataset_figures.py -q -p no:randomly

The shapes are asserted against the imported schema objects rather than pasted column
lists — a list copied into a test is a second opinion nobody checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import runpy
import sys
import tomllib
from pathlib import Path

import pyarrow as pa
import pytest

from multimodal_pipeline.artifacts import ARTIFACT_LAYOUT
from multimodal_pipeline.schemas import (
    ACTIVE_SPEAKER_FRAMES_SCHEMA,
    BODY_25_KEYPOINT_NAMES,
    BODY_SCHEMA,
    SPEAKER_TURNS_SCHEMA,
    WORDS_SCHEMA,
)
from multimodal_pipeline.stages.activespeaker import FRAME_REASONS
from multimodal_pipeline.stages.base import STAGE_ORDER

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "make_dataset_figures.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

# find_spec answers without executing matplotlib, so asking keeps the import discipline
# this file is partly there to enforce.
HAVE_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
requires_matplotlib = pytest.mark.skipif(
    not HAVE_MATPLOTLIB,
    reason="needs matplotlib: uv run --with pytest --with matplotlib pytest ...",
)

#: Figure -> the source name its renderer blames when that source has nothing to draw.
FIGURE_SOURCE = {
    "stage_graph": "stage list",
    "active_speaker_strip": "active_speaker_frames",
    "speaker_turn_strip": "speaker_turns",
    "pose_skeleton_strip": "pose_body",
}

PARQUET_SOURCES = ("active_speaker_frames", "speaker_turns", "speech_words", "pose_body")


@pytest.fixture(scope="module")
def figures() -> object:
    """The script as a module. Loaded from path because scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("make_dataset_figures_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def no_plotting(monkeypatch: pytest.MonkeyPatch):
    """Install a plotting seam that raises, so a rejected input proves it rejected early.

    Without it, a guard that silently let bad data through would still be able to pass a
    test by producing an empty-looking PNG.
    """

    def explode() -> None:
        raise AssertionError("drew after the input should have been rejected")

    def install(figures: object) -> None:
        monkeypatch.setattr(figures, "_pyplot", explode)

    return install


class TestSyntheticShapesMatchTheRealSchemas:
    """The synthetic dataset is checked against the schema objects it claims to mirror.

    Without this, `--synthetic` could keep rendering a pretty figure while the real table
    gained or renamed a column, and the committed docs would demonstrate a dataset that no
    longer exists.
    """

    @pytest.mark.parametrize(
        ("source", "schema"),
        [
            ("active_speaker_frames", ACTIVE_SPEAKER_FRAMES_SCHEMA),
            ("speaker_turns", SPEAKER_TURNS_SCHEMA),
            ("speech_words", WORDS_SCHEMA),
            ("pose_body", BODY_SCHEMA),
        ],
    )
    def test_column_names_are_the_declared_ones(self, figures: object, source: str,
                                                schema: pa.schema) -> None:
        table = figures.synthetic_tables(seed=7)[source]
        assert isinstance(table, pa.Table)
        assert table.column_names == list(schema.names), source
        assert table.num_rows > 0

    def test_every_column_the_renderers_read_is_present(self, figures: object) -> None:
        # REQUIRED_COLUMNS is the renderers' own contract; this asserts it is a subset of
        # the declared schema, so a typo in it is caught here as well as at render time.
        for figure, by_source in figures.REQUIRED_COLUMNS.items():
            for source, columns in by_source.items():
                declared = figures.SYNTHETIC_SCHEMAS[source].names
                assert set(columns) <= set(declared), f"{figure}/{source}: {set(columns) - set(declared)}"
                assert set(columns) <= set(figures.synthetic_tables(seed=7)[source].column_names)

    def test_frames_table_is_dense_and_uses_the_real_reason_vocabulary(
        self, figures: object
    ) -> None:
        rows = figures.synthetic_tables(seed=7)["active_speaker_frames"].to_pylist()
        assert [row["frame_number"] for row in rows] == list(range(len(rows))), (
            "the frames table is meant to be dense; a gap here means the figure shows a "
            "sparsity the stage never produces"
        )
        reasons = {row["frame_reason"] for row in rows}
        assert reasons <= set(FRAME_REASONS), f"invented frame_reason: {reasons - set(FRAME_REASONS)}"
        # The three reasons the strip exists to distinguish.
        assert {"scored", "no_face", "imputed_tail"} <= reasons
        assert any(row["score_imputed"] for row in rows)
        assert any(row["is_active_speaker"] for row in rows)
        assert any(row["track_id"] is None for row in rows), "no_face rows carry no track"
        assert all(row["schema_version"] == "1.2" for row in rows)

    def test_pose_rows_only_use_real_body_25_names_and_ids(self, figures: object) -> None:
        rows = figures.synthetic_tables(seed=7)["pose_body"].to_pylist()
        used = {row["keypoint_name"] for row in rows}
        assert used <= set(BODY_25_KEYPOINT_NAMES), f"unknown keypoints: {used - set(BODY_25_KEYPOINT_NAMES)}"
        assert "Background" not in used, "Background is a filler channel, not a joint"
        ids = {row["keypoint_name"]: row["keypoint_id"] for row in rows}
        for name, key_id in ids.items():
            assert BODY_25_KEYPOINT_NAMES[key_id] == name, f"{name} carries the wrong keypoint_id"

    def test_turns_and_words_share_one_speaker_namespace(self, figures: object) -> None:
        tables = figures.synthetic_tables(seed=7)
        turn_speakers = set(tables["speaker_turns"].column("speaker_id").to_pylist())
        word_speakers = set(tables["speech_words"].column("speaker_id").to_pylist())
        assert word_speakers <= turn_speakers, "word ticks would have a lane of their own"
        assert len(turn_speakers) >= 2, "one speaker cannot show an overlap"

    def test_stage_list_is_the_orchestrators_own(self, figures: object) -> None:
        # The figure must move when a stage is added. A hand-copied list would let the
        # committed diagram quietly disagree with the DAG.
        assert figures.synthetic_tables(seed=7)["stage list"] == list(STAGE_ORDER)

    def test_the_script_imports_the_stage_and_reason_vocabularies(self, figures: object) -> None:
        # Stage names and frame reasons come from the pipeline, so a rename upstream moves
        # the figure instead of leaving a confidently wrong diagram behind.
        body = SOURCE.partition("class FigureDataError")[2]
        for stage in STAGE_ORDER:
            assert f'"{stage}"' not in body, f"{stage} was hand-copied into the renderer"
        assert "FRAME_REASONS" in SOURCE and "STAGE_ORDER" in SOURCE
        assert not re.search(r"^FRAME_REASONS\s*=", SOURCE, re.MULTILINE), (
            "the reason vocabulary was redefined instead of imported")
        # Every colour key is a reason the stage can actually write.
        assert set(figures.REASON_COLORS) <= set(FRAME_REASONS), (
            set(figures.REASON_COLORS) - set(FRAME_REASONS))


class TestBoneList:
    """BODY_25 bone endpoints are validated against the imported keypoint names."""

    def test_every_bone_endpoint_is_a_real_keypoint(self, figures: object) -> None:
        known = set(BODY_25_KEYPOINT_NAMES)
        offenders = [f"{a}-{b}" for a, b in figures.BODY_25_BONES
                     if a not in known or b not in known]
        assert not offenders, (
            f"bone endpoints not in BODY_25_KEYPOINT_NAMES: {offenders}. Rename the bone "
            "list with the schema; do not let the renderer drop the bone in silence."
        )

    def test_no_bone_is_a_self_loop_or_a_duplicate(self, figures: object) -> None:
        pairs = [(min(a, b), max(a, b)) for a, b in figures.BODY_25_BONES]
        assert len(set(pairs)) == len(pairs), "duplicate bone pair"
        assert all(a != b for a, b in figures.BODY_25_BONES), "self-loop bone"

    def test_the_skeleton_has_no_floating_part(self, figures: object) -> None:
        """Exactly three components: the body, and one foot each side.

        This is what catches a *dropped* bone — the endpoint test passes when the leg
        bones go missing and the figure just loses a leg. Three, not one, because that is
        how OpenPose itself defines BODY_25 pairs: it links toe-to-toe and toe-to-heel but
        never ankle-to-toe, so its own renders show the feet detached. Reproducing that is
        deliberate (the figure is a view of BODY_25, not of an idealised stick figure); an
        isolated knee or a fourth component is a bug.
        """
        adjacency: dict[str, set[str]] = {}
        for a, b in figures.BODY_25_BONES:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        assert "Background" not in adjacency, "Background is a filler channel, not a joint"
        assert set(adjacency) == set(BODY_25_KEYPOINT_NAMES) - {"Background"}, (
            "a landmark has no bone at all")

        nodes = set(adjacency)
        components: list[set[str]] = []
        while nodes:
            seed = nodes.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                newly = adjacency[frontier.pop()] - component
                component |= newly
                nodes -= newly
                frontier.extend(newly)
            components.append(component)
        assert len(components) == 3, (
            f"expected body + one component per foot, got {sorted(map(sorted, components))}")
        sizes = sorted(map(len, components), reverse=True)
        assert sizes == [19, 3, 3], f"the skeleton changed shape: {sizes}"
        assert any({"Nose", "Neck", "MidHip", "LAnkle", "RAnkle"} <= c for c in components), (
            "head, torso and ankles are no longer one body")
        assert {"LBigToe", "LSmallToe", "LHeel"} in components, "left foot is not a foot"
        assert {"RBigToe", "RSmallToe", "RHeel"} in components, "right foot is not a foot"

    def test_the_renderer_reads_landmarks_from_the_table_not_from_literals(
        self, figures: object
    ) -> None:
        # The synthetic builder is allowed to name landmarks (it has to place them); the
        # renderer is not, because a renderer that only draws hard-coded joints would look
        # correct against a table whose names changed.
        body = SOURCE.partition("def render_pose_skeleton_strip(")[2].partition(
            "RENDERERS: dict")[0]
        offenders = [name for name in BODY_25_KEYPOINT_NAMES if f'"{name}"' in body]
        assert not offenders, f"landmarks hard-coded in the renderer: {offenders}"
        assert "BODY_25_BONES" in body, "the renderer must draw the validated bone list"


class TestFailureHonesty:
    """Empty or missing input is a named error and a nonzero exit, never an empty PNG."""

    @pytest.mark.parametrize("figure", sorted(FIGURE_SOURCE))
    def test_zero_rows_names_the_source_and_writes_nothing(
        self, figures: object, tmp_path: Path, no_plotting, figure: str
    ) -> None:
        source = FIGURE_SOURCE[figure]
        data = figures.synthetic_tables(seed=7)
        data[source] = [] if source == "stage list" else data[source].slice(0, 0)
        out = tmp_path / f"{figure}.png"
        with pytest.raises(figures.FigureDataError) as excinfo:
            figures.RENDERERS[figure](data, out, label="x")
        assert source in str(excinfo.value), f"the message must name '{source}': {excinfo.value}"
        assert not out.exists(), "a failed figure must not leave a PNG behind"

    def test_render_reports_every_figure_that_could_not_be_drawn(
        self, figures: object, tmp_path: Path, no_plotting
    ) -> None:
        data = figures.synthetic_tables(seed=7)
        data["stage list"] = []
        for source in PARQUET_SOURCES:
            data[source] = data[source].slice(0, 0)
        with pytest.raises(figures.FigureDataError) as excinfo:
            figures._render(data, tmp_path, label="x")
        assert "4 of 4" in str(excinfo.value)
        assert not list(tmp_path.glob("*.png"))

    def test_a_renamed_column_fails_loudly_instead_of_rendering_an_empty_strip(
        self, figures: object, tmp_path: Path, no_plotting
    ) -> None:
        """The defect this script was most likely to ship.

        ``row.get("frame_reason")`` returns None everywhere after a rename and draws a
        clean, completely empty strip. The column contract turns that into a named error.
        """
        data = figures.synthetic_tables(seed=7)
        for figure, source in (("active_speaker_strip", "active_speaker_frames"),
                               ("pose_skeleton_strip", "pose_body")):
            table = data[source]
            renamed = ["frame_reason_renamed" if c == "frame_reason" else
                       ("keypoint_name_renamed" if c == "keypoint_name" else c)
                       for c in table.column_names]
            data[source] = table.rename_columns(renamed)
            out = tmp_path / f"{figure}.png"
            with pytest.raises(figures.FigureDataError) as excinfo:
                figures.RENDERERS[figure](data, out, label="x")
            assert "missing column" in str(excinfo.value), excinfo.value
            assert not out.exists()

    @requires_matplotlib
    def test_a_renamed_keypoint_in_the_data_drops_bones_silently(
        self, figures: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The silence that makes the schema-level bone test necessary, measured.

        When the table's `keypoint_name` stops matching the bone list, the renderer draws
        fewer bones and reports nothing: it cannot distinguish "this dataset has no Neck"
        from "we renamed Neck". Nothing downstream can catch that from the output, so
        ``test_every_bone_endpoint_is_a_real_keypoint`` is the check that has to exist —
        this test is the evidence for that claim rather than an assertion about pixels.
        """
        import matplotlib.pyplot as plt

        def skeleton_counts(data: object) -> tuple[int, int, int]:
            """(panels drawn, bone segments drawn, panels that drew a skeleton).

            A panel that found nothing confident gets no title and no lines, so the third
            count is read off the titles the renderer itself wrote.
            """
            captured: list = []
            original = plt.subplots

            def spy(*call_args: object, **call_kwargs: object):
                result = original(*call_args, **call_kwargs)
                captured.append(result)
                return result

            monkeypatch.setattr(plt, "subplots", spy)
            figures.render_pose_skeleton_strip(data, tmp_path / "pose.png", label="x")
            _, axes = captured[-1]
            panels = [axes] if hasattr(axes, "lines") else list(axes)
            monkeypatch.undo()
            plt.close("all")
            skeletons = sum(1 for ax in panels if "keypoint" in ax.get_title())
            return len(panels), sum(len(ax.lines) for ax in panels), skeletons

        intact_panels, intact, drawn = skeleton_counts(figures.synthetic_tables(seed=7))
        data = figures.synthetic_tables(seed=7)
        rows = data["pose_body"].to_pylist()
        neck_degree = sum(1 for a, b in figures.BODY_25_BONES if "Neck" in (a, b))
        assert 0 < drawn < intact_panels, (
            "the synthetic set should mix drawn skeletons with an empty panel")
        rows = data["pose_body"].to_pylist()
        for row in rows:  # the rename a future schema bump would perform
            if row["keypoint_name"] == "Neck":
                row["keypoint_name"] = "NeckRenamed"
        data["pose_body"] = pa.Table.from_pylist(rows, schema=BODY_SCHEMA)
        renamed_panels, renamed, _ = skeleton_counts(data)
        assert renamed_panels == intact_panels, "panels unchanged; only bones went missing"
        assert renamed < intact, (
            "expected the renderer to lose bones quietly; if it now refuses loudly, "
            "delete this test and the schema-level bone test's rationale with it")
        # Every bone touching Neck disappears, once per panel that drew a skeleton.
        assert intact - renamed == neck_degree * drawn, (intact, renamed, drawn, neck_degree)

    def test_cli_exits_nonzero_and_names_every_missing_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An empty dataset directory: exit 1, all four tables named, nothing written.

        Run through ``__main__`` so the exit code is the real one an operator sees. The
        tables are missing, so no figure reaches matplotlib and this runs in a suite
        without it.
        """
        dataset = tmp_path / "empty-dataset"
        dataset.mkdir()
        out = tmp_path / "out"
        monkeypatch.setattr(sys, "argv",
                            [str(SCRIPT), "--dataset", str(dataset), "--out", str(out)])
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        assert excinfo.value.code == 1
        err = capsys.readouterr().err
        for source in PARQUET_SOURCES:
            assert source in err, f"{source} not named in: {err}"
        assert not list(out.glob("*.png")), "nothing renderable, so nothing to leave behind"

    def test_a_zero_row_parquet_file_fails_with_the_table_name(
        self, figures: object, tmp_path: Path, no_plotting
    ) -> None:
        """The file exists and is valid Parquet, and it is still an error."""
        import pyarrow.parquet as pq

        dataset = tmp_path / "ds"
        out = tmp_path / "out"
        tables = figures.synthetic_tables(seed=7)
        for source in PARQUET_SOURCES:
            # pose_body arrives empty; the other three are real.
            table = tables[source].slice(0, 0) if source == "pose_body" else tables[source]
            path = dataset / ARTIFACT_LAYOUT[source]
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)
        data = figures.read_dataset_tables(dataset)
        with pytest.raises(figures.FigureDataError) as excinfo:
            figures.RENDERERS["pose_skeleton_strip"](data, out / "p.png", label="x")
        assert "pose_body" in str(excinfo.value)
        assert not (out / "p.png").exists()


class TestCli:
    def test_a_mode_is_required(self, figures: object, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            figures.build_parser().parse_args(["--out", str(tmp_path)])
        assert excinfo.value.code == 2

    def test_synthetic_and_dataset_cannot_both_be_given(self, figures: object,
                                                        tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as excinfo:
            figures.build_parser().parse_args(
                ["--synthetic", "--dataset", "x", "--out", str(tmp_path)])
        assert excinfo.value.code == 2

    def test_out_is_required(self, figures: object) -> None:
        with pytest.raises(SystemExit) as excinfo:
            figures.build_parser().parse_args(["--synthetic"])
        assert excinfo.value.code == 2

    def test_seed_defaults_to_seven(self, figures: object, tmp_path: Path) -> None:
        assert figures.build_parser().parse_args(
            ["--synthetic", "--out", str(tmp_path)]).seed == 7

    def test_dataset_must_be_a_directory(self, figures: object, tmp_path: Path,
                                         capsys: pytest.CaptureFixture[str]) -> None:
        assert figures.main(["--dataset", str(tmp_path / "nope"), "--out", str(tmp_path)]) == 1
        assert "not a directory" in capsys.readouterr().err

    def test_video_label_override_is_parsed(self, figures: object, tmp_path: Path) -> None:
        args = figures.build_parser().parse_args(
            ["--synthetic", "--seed", "3", "--out", str(tmp_path), "--video", "my-clip"])
        assert (args.seed, args.video) == (3, "my-clip")


class TestModuleImportDiscipline:
    """No pandas, no matplotlib at import time, and the uv command is documented."""

    def test_pandas_is_never_imported(self, figures: object) -> None:
        # pandas is not a project dependency; the pipeline reads Parquet with pyarrow.
        assert "pandas" not in sys.modules
        assert not re.search(r"^\s*(import|from)\s+pandas", SOURCE, re.MULTILINE)
        assert "pd.read_parquet" not in SOURCE

    def test_pyproject_still_lacks_matplotlib(self) -> None:
        def names(deps: list[str]) -> set[str]:
            return {re.split(r"[<>=!\[]", dep)[0].strip().lower() for dep in deps}

        assert "matplotlib" not in names(PYPROJECT["project"]["dependencies"])
        assert "matplotlib" not in names(PYPROJECT.get("dependency-groups", {}).get("dev", []))

    def test_matplotlib_is_imported_lazily_with_agg(self) -> None:
        head, _, rest = SOURCE.partition("def _pyplot(")
        seam = rest.partition("return _PLT")[0]
        assert "import matplotlib" in seam, "matplotlib must be imported inside the seam"
        assert 'matplotlib.use("Agg")' in seam, "Agg must be pinned before pyplot is imported"
        assert "import matplotlib.pyplot" in seam
        assert seam.index('matplotlib.use("Agg")') < seam.index("import matplotlib.pyplot")
        assert "import matplotlib" not in head, "module-scope matplotlib breaks tests/unit"

    @pytest.mark.skipif(HAVE_MATPLOTLIB,
                        reason="only meaningful where matplotlib is genuinely absent")
    def test_a_matplotlib_free_interpreter_gets_a_named_error_not_a_traceback(
        self, figures: object, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The failure the suite itself hits: `pytest tests/unit` without --with matplotlib."""
        with pytest.raises(figures.MatplotlibMissing):
            figures._pyplot()
        assert figures.main(["--synthetic", "--out", str(tmp_path)]) == 1
        err = capsys.readouterr().err
        assert "matplotlib" in err.lower()
        assert "uv run --with matplotlib" in err, "the fix must be in the message"
        assert not list(tmp_path.glob("*.png"))

    def test_the_documented_command_is_the_one_that_works(self) -> None:
        assert "uv run --with matplotlib python scripts/make_dataset_figures.py" in SOURCE
        # One constant, reused by the epilog and the missing-matplotlib message.
        assert SOURCE.count('RUN_COMMAND = "') == 1

    def test_the_script_explains_why_committed_figures_are_synthetic(self) -> None:
        # The legal reason, in the file, so nobody "improves" the committed assets with a
        # real clip.
        doc = SOURCE.partition('"""')[2].partition('"""')[0]
        assert "copyright" in doc.lower()
        assert "--synthetic" in doc and "docs/assets" in doc


@requires_matplotlib
class TestRenderSynthetic:
    def test_writes_four_non_empty_pngs(self, figures: object, tmp_path: Path) -> None:
        out = tmp_path / "assets"
        written = figures.render_synthetic(out, seed=7)
        assert [path.name for path in written] == [
            "stage_graph.png", "active_speaker_strip.png",
            "speaker_turn_strip.png", "pose_skeleton_strip.png",
        ]
        for path in written:
            assert path.is_file(), path
            assert path.stat().st_size > 4000, f"{path.name} is suspiciously small"
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"

    def test_same_seed_is_byte_identical_across_runs(self, figures: object,
                                                     tmp_path: Path) -> None:
        first, second = tmp_path / "a", tmp_path / "b"
        figures.render_synthetic(first, seed=7)
        figures.render_synthetic(second, seed=7)
        assert {p.name: digest(p) for p in first.iterdir()} == {
            p.name: digest(p) for p in second.iterdir()
        }, "a committed PNG that changes when nothing changed is review noise"

    def test_a_different_seed_changes_the_data_driven_figures(self, figures: object,
                                                              tmp_path: Path) -> None:
        first, other = tmp_path / "s7", tmp_path / "s8"
        figures.render_synthetic(first, seed=7)
        figures.render_synthetic(other, seed=8)
        changed = {name for name in (p.name for p in first.iterdir())
                   if digest(first / name) != digest(other / name)}
        assert {"active_speaker_strip.png", "pose_skeleton_strip.png"} <= changed, (
            "the seed reaches nothing, so the dataset is not seeded, just constant")

    def test_pose_strip_keeps_the_panel_for_a_frame_with_no_confident_detection(
        self, figures: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frames with nothing confident are announced inside the figure, not skipped.

        Observed on the axes matplotlib actually built: one panel per sampled frame, and
        the unconfident one carries the disclosure text. Silently dropping it would show a
        clip as more measurable than it was.
        """
        rows = figures.synthetic_tables(seed=7)["pose_body"].to_pylist()
        frames = sorted({row["frame_number"] for row in rows})
        unconfident = {
            frame for frame in frames
            if max(row["confidence"] or 0.0 for row in rows if row["frame_number"] == frame)
            < figures.CONFIDENCE_FLOOR
        }
        assert unconfident, "the synthetic set stopped exercising the empty-frame path"

        import matplotlib.pyplot as plt

        original = plt.subplots
        captured: list = []

        def spy(*call_args: object, **call_kwargs: object):
            result = original(*call_args, **call_kwargs)
            captured.append(result)
            return result

        monkeypatch.setattr(plt, "subplots", spy)
        figures.render_pose_skeleton_strip(figures.synthetic_tables(seed=7),
                                           tmp_path / "pose.png", label="x")
        _, axes = captured[-1]
        axes = [axes] if hasattr(axes, "texts") else list(axes)
        assert len(axes) == figures.POSE_FRAMES, "a sampled frame was dropped from the strip"
        texts = " ".join(t.get_text() for ax in axes for t in ax.texts)
        assert "no detection" in texts and str(min(unconfident)) in texts
        plt.close("all")


@requires_matplotlib
class TestCommittedAssetBudget:
    """Committed images stay small enough to review, and readable by a PNG parser."""

    def test_synthetic_figures_are_small(self, figures: object, tmp_path: Path) -> None:
        for path in figures.render_synthetic(tmp_path, seed=7):
            assert path.stat().st_size < 400_000, f"{path.name} is too big to commit"

    def test_png_headers_are_sane(self, figures: object, tmp_path: Path) -> None:
        # Structural check without a second imaging dependency: IHDR dimensions and IDAT.
        for path in figures.render_synthetic(tmp_path, seed=7):
            raw = path.read_bytes()
            assert raw[12:16] == b"IHDR"
            width = int.from_bytes(raw[16:20], "big")
            height = int.from_bytes(raw[20:24], "big")
            assert width > 200 and height > 200, f"{path.name} is {width}x{height}"
            assert b"IDAT" in raw

    def test_the_committed_assets_are_reproducible_from_the_script(self, figures: object,
                                                                   tmp_path: Path) -> None:
        """docs/assets/*.png are this command's output, byte for byte.

        The claim in docs/assets/README.md is only worth making if a test checks it:
        a committed image nobody can regenerate is indistinguishable from one that came
        from a broadcast frame.
        """
        committed = ROOT / "docs" / "assets"
        present = sorted(p.name for p in committed.glob("*.png")) if committed.is_dir() else []
        if not present:
            pytest.skip("docs/assets holds no PNGs yet")
        expected = sorted(f"{name}.png" for name in figures.FIGURES)
        assert present == expected, f"committed set drifted: {present} != {expected}"
        figures.render_synthetic(tmp_path, seed=7)
        for name in expected:
            assert digest(tmp_path / name) == digest(committed / name), (
                f"docs/assets/{name} is not what `--synthetic --seed 7` produces")
