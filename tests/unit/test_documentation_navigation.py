"""Navigation guards for the README + the eight topic guides under ``docs/``.

The README used to be 2112 lines: one file that was simultaneously the landing page, the
install chain, the dataset schema reference and the ELAN tier semantics. The fix is
structure — a short README that *routes*, and eight guides that hold the evidence moved
out of it, so a reader can reach the long reference material without scrolling past it to
find out what the pipeline does.

Structure has its own rot, and none of it is caught by ``test_readme_claims.py``:

* a guide that is renamed, or never written, leaves a dead link in the README;
* a guide that does not link back strands a reader who arrived at it from a search engine;
* a moved heading leaves a fragment that resolves to nothing, which is the failure mode
  *within* a document that no link checker without anchor support can see;
* an image whose caption does not say it was drawn from a synthetic schema invites the
  reading that it is a frame of a real clip — the one claim this repository cannot make,
  because its input is copyrighted broadcast material (see ``docs/assets/README.md``).

So the guard is mechanical and closed: an explicit allowlist of documents, every local
link and image resolving from the file that wrote it, every ``#fragment`` matching a real
heading in the document that owns it, fenced code excluded (a ``#`` inside a bash block is
a comment, not a heading, and a ``[text](thing)`` inside a Python block is not a link).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS = ROOT / "docs"

#: The eight topic guides the README routes to. Not a cap on how many guides may exist:
#: add a ninth here and link it from the README. What this list makes checkable is that
#: every document under ``docs/`` is reachable from the landing page, because a guide
#: nobody links is content the repository ships and no reader finds.
GUIDES: tuple[str, ...] = (
    "getting-started.md",
    "cli.md",
    "datasets.md",
    "modalities.md",
    "elan.md",
    "architecture.md",
    "configuration.md",
    "development.md",
)

#: How the README announces the guides, and how every guide points home.
ROOT_LINK_RE = re.compile(r"\]\(docs/([a-z-]+\.md)\)")
BACKLINK_RE = re.compile(r"\]\(\.\./README\.md\)")

#: The sentence every command-bearing guide owes the reader: these commands are written
#: relative to the repository root, which is the only place they work.
ROOT_CWD_SENTENCE = "run them from the repository root"

#: The stale figure. ``docs/assets/stage_graph.png`` is regenerated from ``STAGE_ORDER``
#: and ``STAGE_DEPENDENCIES``, but the committed PNG predates the ``stories`` stage: it
#: draws a 17-stage graph for an 18-stage pipeline, and its labels overlap badly enough to
#: be unreadable. Displaying a DAG that contradicts the pipeline is worse than displaying
#: none, so no document may embed it while it is stale.
STALE_FIGURE = "stage_graph.png"

FENCE_RE = re.compile(r"^\s*```")
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
TOC_ITEM_RE = re.compile(r"^- \[[^\]]+\]\(#([a-z0-9_\-]+)\)", re.MULTILINE)


def strip_fences(text: str) -> str:
    """The document with every fenced code block removed.

    Needed in both directions: a ``# comment`` line inside a bash block is not a heading,
    and a bare URL or bracketed expression inside a code block is not a link.
    """
    kept: list[str] = []
    inside = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            inside = not inside
            continue
        if not inside:
            kept.append(line)
    return "\n".join(kept)


def slugify(title: str) -> str:
    """GitHub's heading slug: drop the markup, lowercase, spaces to hyphens."""
    title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", title)
    title = title.replace("`", "").strip().lower()
    title = re.sub(r"[^\w\s-]", "", title, flags=re.UNICODE)
    return title.replace(" ", "-")


def anchors_in(text: str) -> set[str]:
    """Slugs for every heading outside a fenced block.

    Deliberately reimplemented rather than imported: the point is that a fragment written
    in a document matches what a renderer would generate from the heading that document
    owns, and the renderer's rule is short enough to state exactly.
    """
    return {
        slugify(match.group(2))
        for line in strip_fences(text).splitlines()
        if (match := HEADING_RE.match(line))
    }


def documents() -> list[Path]:
    return [README, *(DOCS / name for name in GUIDES)]


def body(path: Path) -> str:
    return strip_fences(path.read_text(encoding="utf-8"))


class TestLandingPage:
    def test_the_readme_routes_instead_of_being_the_manual(self) -> None:
        """The design decision, pinned.

        The limit is generous on purpose (the page as written is ~185 lines). It exists
        because length is what made the old README hard to start from, and length grows
        one "useful paragraph" at a time until the file is a manual again.
        """
        lines = README.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 220, (
            f"README.md is {len(lines)} lines; it is a landing page that routes to the "
            "eight guides under docs/, not the manual itself. Move the new content into "
            "the guide that owns the topic and link it.")

    def test_the_readme_titles_the_project_and_names_the_cli_it_ships(self) -> None:
        text = README.read_text(encoding="utf-8")
        assert text.lstrip().startswith("# Storytel Pipeline")
        # The console script keeps its own name; the page title does not rename it.
        assert "multimodal-pipeline" in text

    def test_the_readme_links_every_guide(self) -> None:
        # Repeats are fine — a row in the navigation table and a pointer in the
        # limitations list are both legitimate. What is not fine is a guide the README
        # never routes to.
        linked = set(ROOT_LINK_RE.findall(body(README)))
        missing = [name for name in GUIDES if name not in linked]
        assert not missing, f"the README does not route to: {missing}"
        extra = sorted(linked - set(GUIDES))
        assert not extra, f"the README links a guide outside this test's allowlist: {extra}"

    def test_the_readme_links_the_license(self) -> None:
        assert re.search(r"\]\(\.?/?LICENSE\)", body(README)), "the License line lost its link"


class TestGuidesExistAndAreNavigable:
    @pytest.mark.parametrize("name", GUIDES)
    def test_the_guide_exists_with_a_title_and_a_purpose(self, name: str) -> None:
        path = DOCS / name
        assert path.is_file(), f"docs/{name} is missing; the README links it"
        text = path.read_text(encoding="utf-8").lstrip()
        assert text.startswith("# "), f"docs/{name} has no H1 title"
        # Title, then a sentence saying what the guide is for — before any moved content.
        blocks = text.split("\n\n", 2)
        assert len(blocks) > 1 and len(blocks[1].strip()) > 40, (
            f"docs/{name} has no purpose paragraph under its title")

    @pytest.mark.parametrize("name", GUIDES)
    def test_the_guide_links_back_to_the_readme(self, name: str) -> None:
        assert BACKLINK_RE.search(body(DOCS / name)), (
            f"docs/{name} has no [back to README](../README.md) link; a reader who lands "
            "here from a search engine has no way back to the overview")

    @pytest.mark.parametrize("name", GUIDES)
    def test_the_guide_has_a_complete_table_of_contents(self, name: str) -> None:
        """The TOC lists every ``##`` and ``###`` heading, in document order.

        Completeness rather than a minimum count: a section added later must not be
        invisible at the top of the file, which is what keeps a long guide scannable.
        Dangling fragments are caught by the link resolver below.
        """
        text = (DOCS / name).read_text(encoding="utf-8")
        toc = TOC_ITEM_RE.findall(strip_fences(text))
        assert toc, f"docs/{name} has no in-page TOC"
        headings = [
            slugify(match.group(2))
            for line in strip_fences(text).splitlines()
            if (match := HEADING_RE.match(line))
            and 2 <= len(match.group(1)) <= 3 and match.group(2) != "Contents"
        ]
        assert toc == headings, (
            f"docs/{name} TOC {toc} does not list its sections in order {headings}")

    def test_every_document_under_docs_is_in_the_allowlist(self) -> None:
        """A new guide has to be routed to, not merely written.

        Adding a ninth guide means adding it to ``GUIDES`` and linking it from the README;
        this fails while the document exists but is unreachable, which is the state where
        content ships and no reader finds it.
        """
        present = sorted(p.name for p in DOCS.glob("*.md"))
        assert present == sorted(GUIDES), (
            "docs/ holds a document that is not in GUIDES and therefore not routed to by "
            f"the README (or lost one): {present}")


class TestLinksResolve:
    @pytest.mark.parametrize("path", documents(), ids=lambda p: str(p.relative_to(ROOT)))
    def test_every_local_link_and_image_resolves_from_its_own_document(self, path: Path) -> None:
        assert path.is_file(), f"{path.relative_to(ROOT)} is missing"
        text = body(path)
        broken: list[str] = []
        slugs: dict[Path, set[str]] = {}
        for target in LINK_RE.findall(text) + [m[1] for m in IMAGE_RE.findall(text)]:
            if target.startswith(("http://", "https://", "mailto:", "file://")):
                continue
            file_part, _, fragment = target.partition("#")
            resolved = (path.parent / file_part).resolve() if file_part else path
            if file_part and not resolved.exists():
                broken.append(f"{target} -> no such file from {path.relative_to(ROOT)}")
                continue
            if not fragment:
                continue
            slugs.setdefault(resolved, anchors_in(resolved.read_text(encoding="utf-8")))
            if fragment not in slugs[resolved]:
                broken.append(f"{target} -> no heading #{fragment} in "
                              f"{resolved.relative_to(ROOT)}")
        assert not broken, f"unresolved links in {path.relative_to(ROOT)}: {broken}"

    @pytest.mark.parametrize("name", GUIDES)
    def test_guide_links_are_written_relative_to_their_own_directory(self, name: str) -> None:
        """No ``/docs/...`` or ``README.md`` shortcuts: guides live one level down.

        A root-relative link is not a browser-relative one, and ``README.md`` inside
        ``docs/`` would be a different file from the one at the root.
        """
        text = body(DOCS / name)
        targets = LINK_RE.findall(text) + [m[1] for m in IMAGE_RE.findall(text)]
        for target in targets:
            if target.startswith(("http://", "https://", "mailto:", "file://", "#")):
                continue
            file_part = target.partition("#")[0]
            assert not file_part.startswith("/"), f"docs/{name} links root-relative: {target}"
            assert file_part != "README.md", (
                f"docs/{name} links docs/README.md rather than ../README.md: {target}")
            assert file_part != "LICENSE", f"docs/{name} links docs/LICENSE: {target}"


class TestFigureHonesty:
    @pytest.mark.parametrize("path", documents(), ids=lambda p: str(p.relative_to(ROOT)))
    def test_every_embedded_figure_says_it_is_synthetic(self, path: Path) -> None:
        """No committed PNG may read as a frame of a real clip.

        Every figure is drawn by ``scripts/make_dataset_figures.py --synthetic`` from an
        in-memory dataset; the pipeline's real input is copyrighted broadcast material, so
        nothing derived from it is committed here. The caption is where a reader learns
        that before drawing a conclusion from the picture.
        """
        offenders = [
            f"{caption!r} -> {target}"
            for caption, target in IMAGE_RE.findall(body(path))
            if target.endswith(".png") and "synthetic" not in caption.lower()
        ]
        assert not offenders, (
            f"figures in {path.relative_to(ROOT)} are displayed without naming their "
            f"synthetic origin in the alt text: {offenders}")

    def test_the_stale_stage_graph_is_not_displayed_anywhere(self) -> None:
        offenders = [str(p.relative_to(ROOT)) for p in documents() if STALE_FIGURE in body(p)]
        assert not offenders, (
            f"{STALE_FIGURE} is embedded in {offenders}. The committed PNG predates the "
            "stories stage, so it draws a 17-stage graph for an 18-stage pipeline; the "
            "text graph in docs/architecture.md follows STAGE_DEPENDENCIES.")

    def test_the_committed_figures_are_the_ones_the_assets_index_lists(self) -> None:
        listed = (DOCS / "assets" / "README.md").read_text(encoding="utf-8")
        for name in ("speaker_turn_strip.png", "active_speaker_strip.png",
                     "pose_skeleton_strip.png"):
            assert (DOCS / "assets" / name).is_file(), f"docs/assets/{name} vanished"
            assert name in listed, f"docs/assets/README.md no longer indexes {name}"


class TestCommandsAreRunnableAsWritten:
    @pytest.mark.parametrize("name", GUIDES)
    def test_a_guide_with_commands_says_they_run_from_the_repository_root(self, name: str) -> None:
        text = (DOCS / name).read_text(encoding="utf-8")
        blocks = re.findall(r"^```bash\n(.*?)^```", text, re.DOTALL | re.MULTILINE)
        executable = [b for b in blocks
                      if any(line.strip() and not line.lstrip().startswith("#")
                             for line in b.splitlines())]
        if not executable:
            pytest.skip(f"docs/{name} shows no runnable command")
        assert ROOT_CWD_SENTENCE in text, (
            f"docs/{name} shows runnable commands without stating that they are written "
            f"to {ROOT_CWD_SENTENCE}")
