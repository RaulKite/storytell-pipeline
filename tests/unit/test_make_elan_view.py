"""Tests for ``scripts/make_elan_view.py``.

The script exists so someone without ELAN can look at an export, which makes its real
failure mode *a view that lies about the file it is showing*: a time axis silently
normalised to the last annotation so every tier appears to fill the clip, a bar holding
three simultaneous labels drawn as one label, an empty tier indistinguishable from a
missing one, pympi's always-present `default` tier counted as a real tier. Each test
attacks one of those.

pympi is a project dependency (the `elan` stage is written against it), so nothing here
skips. Every test builds its own `.eaf` in a tmp dir and asserts against the rendered
HTML, because a bug in a throwaway viewer is invisible to the rest of the suite.
"""

from __future__ import annotations

import html
import importlib.util
import json
import re
from pathlib import Path

import pytest
from pympi.Elan import Eaf, to_eaf

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_elan_view.py"


@pytest.fixture(scope="module")
def view():
    """The script as a module. Loaded from path because scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("make_elan_view_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_eaf(path: Path, tiers: dict[str, list[tuple[int, int, str]]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    eaf = Eaf()
    for tier, rows in tiers.items():
        # pympi raises KeyError for an unknown tier: a tier must be declared first, the
        # same way the pipeline's own exporter declares its 17 before filling them.
        eaf.add_tier(tier)
        for start, end, value in rows:
            eaf.add_annotation(tier, start, end, value)
    to_eaf(str(path), eaf)
    return path


def render(view, eaf: Path, tmp_path: Path) -> tuple[dict, str]:
    """Render and return both the summary dict and the page text."""
    out = tmp_path / "out.html"
    summary = view.main(str(eaf), str(out))
    return summary, out.read_text(encoding="utf-8")


def dataset(tmp_path: Path, tiers: dict, *, duration: float | None = 8.008008) -> Path:
    """Lay files out the way the pipeline does: <dataset>/elan + <dataset>/source."""
    root = tmp_path / "mydataset"
    eaf = write_eaf(root / "elan" / "annotations.eaf", tiers)
    if duration is not None:
        source = root / "source"
        source.mkdir(parents=True, exist_ok=True)
        (source / "metadata.json").write_text(
            json.dumps({"duration_seconds": duration}), encoding="utf-8")
    return eaf


# --- the axis: the defect this script had ---------------------------------


class TestTimeAxis:
    def test_the_axis_is_the_media_duration_when_metadata_is_present(self, view, tmp_path):
        eaf = dataset(tmp_path, {"words": [(0, 1000, "hello")]}, duration=8.008008)
        summary = view.main(str(eaf), str(tmp_path / "out.html"))
        assert summary["axis_ms"] == pytest.approx(8008.008)
        assert summary["axis_kind"] == "source media duration"

    def test_a_tier_that_stops_early_does_not_touch_the_right_edge(self, view, tmp_path):
        """The whole point: with an 8 s axis, a 1 s bar occupies ~12%, not 100%."""
        eaf = dataset(tmp_path, {"words": [(0, 1000, "hello")]}, duration=8.008008)
        html_text = Path(view.main(str(eaf), str(tmp_path / "out.html")) and
                         tmp_path / "out.html").read_text(encoding="utf-8")
        (left, width) = re.search(r'left:([\d.]+)%;width:([\d.]+)%', html_text).groups()
        assert float(left) == pytest.approx(0.0)
        assert float(width) == pytest.approx(12.49, abs=0.05)

    def test_it_says_so_when_it_had_to_use_the_annotation_end(self, view, tmp_path):
        """A bundle of .eaf files travels without the dataset; the page must admit it."""
        eaf = dataset(tmp_path, {"words": [(0, 1000, "hello")]}, duration=None)
        summary, html_text = render(view, eaf, tmp_path)
        assert summary["axis_ms"] == 1000
        assert summary["axis_kind"] == "latest annotation end"
        assert "latest annotation end" in html_text

    def test_an_unusable_duration_falls_back_instead_of_crashing(self, view, tmp_path):
        for payload in ("not json", "{}", '{"duration_seconds": null}',
                        '{"duration_seconds": 0}', '{"duration_seconds": -3}',
                        '{"duration_seconds": "8.0"}'):
            eaf = dataset(tmp_path, {"words": [(0, 1000, "hello")]}, duration=None)
            (eaf.parent.parent / "source").mkdir(parents=True, exist_ok=True)
            (eaf.parent.parent / "source" / "metadata.json").write_text(
                payload, encoding="utf-8")
            assert render(view, eaf, tmp_path)[0]["axis_kind"] == "latest annotation end"

    def test_an_annotation_beyond_the_media_duration_widens_the_axis(self, view, tmp_path):
        """A diarizer turn can run past the declared duration; clipping it would lie."""
        eaf = dataset(tmp_path, {"turns": [(0, 9000, "turn")]}, duration=8.008008)
        summary, html_text = render(view, eaf, tmp_path)
        assert summary["axis_ms"] == 9000

    def test_it_does_not_call_a_widened_axis_the_media_duration(self, view, tmp_path):
        """The axis stopped being the media duration; the header must stop claiming it.

        Measured on this corpus: La-1's latest annotation ends at 8097 ms on an
        8008 ms clip, so the truthful page says the axis is an annotation end past
        the duration, not the duration itself.
        """
        eaf = dataset(tmp_path, {"turns": [(0, 9000, "turn")]}, duration=8.008008)
        summary, html_text = render(view, eaf, tmp_path)
        assert summary["axis_kind"] == "annotation end past the media duration"
        assert "= <b>annotation end past the media duration</b>" in html_text


# --- simultaneity: the other thing the export works hard to preserve ------


class TestSimultaneousBars:
    def test_a_json_list_bar_keeps_every_label_visible(self, view, tmp_path):
        labels = ["speaker_0 · turn1", "speaker_1 · turn2"]
        eaf = dataset(tmp_path, {"turns": [(100, 200, json.dumps(labels))]})
        summary, html_text = render(view, eaf, tmp_path)
        assert summary["simultaneous_bars"] == 1
        assert "2 simultaneous labels" in html_text
        assert 'class="bS"' in html_text
        # The block body is truncated for layout, so the full labels must be in the title.
        for label in labels:
            assert html.escape(label) in html_text, f"{label} lost from the view"

    def test_a_single_label_string_is_not_reported_as_simultaneous(self, view, tmp_path):
        eaf = dataset(tmp_path, {"words": [(0, 5, "hello")]})
        summary, html_text = render(view, eaf, tmp_path)
        assert summary["simultaneous_bars"] == 0
        assert 'class="b"' in html_text and 'class="bS"' not in html_text

    def test_a_one_element_list_is_a_plain_bar(self, view, tmp_path):
        assert view.simultaneous(json.dumps(["only"])) is None

    def test_unparseable_or_non_string_json_degrades_to_a_plain_bar(self, view):
        assert view.simultaneous("[not json") is None
        assert view.simultaneous(json.dumps([1, 2])) is None
        assert view.simultaneous(json.dumps({"a": 1})) is None
        assert view.simultaneous("plain text") is None

    def test_markup_inside_a_label_cannot_escape_into_the_page(self, view, tmp_path):
        evil = '<script>alert(1)</script> · seg000001'
        eaf = dataset(tmp_path, {"words": [(0, 10, evil)]})
        _summary, html_text = render(view, eaf, tmp_path)
        assert "<script>alert(1)</script>" not in html_text
        assert "&lt;script&gt;" in html_text


# --- tier accounting ------------------------------------------------------


class TestTierAccounting:
    def test_pymepis_default_tier_is_never_counted_as_one_of_ours(self, view, tmp_path):
        eaf = dataset(tmp_path, {"words": [(0, 10, "a")]})
        raw = Eaf(str(eaf))
        assert "default" in raw.tiers, "pympi no longer emits a default tier?"
        summary = view.main(str(eaf), str(tmp_path / "out.html"))
        assert summary["tiers"] == 1

    def test_an_empty_tier_is_reported_as_empty_not_omitted(self, view, tmp_path):
        eaf = write_eaf(tmp_path / "b.eaf", {"words": [(0, 10, "a")]})
        extra = Eaf(str(eaf))
        extra.add_tier("pose_presence")
        to_eaf(str(eaf), extra)
        summary, html_text = render(view, eaf, tmp_path)
        assert summary["empty_tiers"] == ["pose_presence"]
        assert "vacía" in html_text

    def test_the_annotation_total_is_the_sum_over_real_tiers(self, view, tmp_path):
        eaf = dataset(tmp_path, {"words": [(0, 5, "a"), (5, 10, "b")],
                                 "turns": [(0, 10, "t")]})
        assert render(view, eaf, tmp_path)[0]["annotations"] == 3

    def test_the_media_url_is_shown_whether_absolute_or_relative(self, view, tmp_path):
        eaf = dataset(tmp_path, {"words": [(0, 10, "a")]})
        doc = Eaf(str(eaf))
        # Absolute only: RELATIVE_MEDIA_URL stays absent, so this exercises the
        # viewer's fallback to MEDIA_URL, which a bundle with relative paths never hits.
        doc.add_linked_file("file:///abs/path/clip.mp4", mimetype="video/mp4")
        to_eaf(str(eaf), doc)
        assert "clip.mp4" in render(view, eaf, tmp_path)[1]

    def test_a_media_descriptor_less_export_still_renders(self, view, tmp_path):
        eaf = dataset(tmp_path, {"words": [(0, 10, "a")]})
        doc = Eaf(str(eaf))
        doc.media_descriptors = []
        to_eaf(str(eaf), doc)
        assert "video: ?" in render(view, eaf, tmp_path)[1]
