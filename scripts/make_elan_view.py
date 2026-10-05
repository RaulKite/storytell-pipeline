#!/usr/bin/env python3
"""Render one .eaf as a self-contained HTML page: tiers as rows, annotations as blocks.

Not a replacement for ELAN — a way to eyeball the export on a machine that has no ELAN,
and a way to see the export's *own* claims (how many bars, which tiers are empty) laid out
against the time axis. Media is never embedded: the videos are gigabytes and a copy would
be a second, staler source of truth. Each block carries its full label in a hover title,
so content can still be checked against the Parquet tables.

Two things this view deliberately refuses to imply:

* **That the page width is the video.** The strip is normalised to the latest annotation
  end unless the dataset's `source/metadata.json` can be found next to the `.eaf`, in
  which case the real media duration is used — unless an annotation runs past it, which
  real diarizer turns do. The header always names which of the three the axis actually
  is. A tier that stops early then reads as stopping early, instead of always touching
  the right edge.
* **That a bar holds one label.** The export partitions simultaneous intervals onto
  independent (non-overlapping) tiers, so one bar can legitimately carry several labels as
  a JSON list. Such a bar is drawn with a `×N` count and a striped edge: collapsing it to
  the first label would hide exactly the simultaneity the export worked to preserve.

Usage:
    uv run python scripts/make_elan_view.py data/processed/<dataset>/elan/annotations.eaf out.html
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

from pympi.Elan import Eaf

COLORS = ["#4f8ef7", "#e0653a", "#3fa66a", "#b072d0", "#d9a13b", "#5aa8c9",
          "#c75a86", "#7a9c3f", "#8a8f98", "#d0603f", "#3d8f8f", "#9a6b3f"]

# pympi always emits a `default` tier in every document; it is not one of ours.
NON_TIER = "default"


def tier_names(eaf: Eaf) -> list[str]:
    """The document's real tiers, in document order, with pympi's `default` removed."""
    return [t for t in eaf.tiers if t != NON_TIER]


def simultaneous(value: object) -> list[str] | None:
    """The labels of a projected bar, or None when the bar carries a single label.

    The exporter writes a plain string for one active label and a JSON list when several
    are simultaneous, so the shape of the value *is* the simultaneity signal.
    """
    if not isinstance(value, str) or not value.startswith("["):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        return None
    return parsed if len(parsed) > 1 else None


def media_duration(eaf_path: Path) -> tuple[float | None, str]:
    """The real media duration in seconds, or how long this page's axis actually is.

    The `.eaf` lives at `<dataset>/elan/annotations.eaf` and the duration at
    `<dataset>/source/metadata.json`, so one `parent.parent` reaches it. Absent is a
    normal state (the bundle of .eaf files travels without the dataset), which is why the
    caller must print which of the two it got.
    """
    metadata = eaf_path.parent.parent / "source" / "metadata.json"
    try:
        document = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "latest annotation end"
    duration = document.get("duration_seconds")
    if not isinstance(duration, (int, float)) or duration <= 0:
        return None, "latest annotation end"
    return float(duration) * 1000.0, "source media duration"


def main(eaf_path: str, out_path: str) -> dict:
    """Render and return a summary of what was rendered (printed by ``__main__``)."""
    source = Path(eaf_path)
    eaf = Eaf(str(source))
    tiers = tier_names(eaf)

    # One pass to learn the axis, because a bar's position is meaningless without it.
    per_tier = {t: sorted(eaf.get_annotation_data_for_tier(t), key=lambda a: a[0])
                for t in tiers}
    ends = [a[1] for anns in per_tier.values() for a in anns]
    media_ms, axis_kind = media_duration(source)
    candidates = [1] + ends + ([media_ms] if media_ms else [])
    duration = max(candidates)
    if media_ms and duration > media_ms:
        # A diarizer turn can run past the declared duration (measured on this corpus:
        # La-1's latest annotation ends at 8097 ms on an 8008 ms clip). Saying the axis
        # *is* the media duration then would be false by 89 ms; say what widened it.
        axis_kind = "annotation end past the media duration"

    rows, total, simultaneous_bars = [], 0, 0
    for i, tier in enumerate(tiers):
        anns = per_tier[tier]
        total += len(anns)
        blocks = []
        for start, end, value in anns:
            labels = simultaneous(value)
            if labels:
                simultaneous_bars += 1
                text = f"{len(labels)} labels · {labels[0]}"
                title = f"{tier}  {start}–{end} ms  {len(labels)} simultaneous labels:\n" \
                    + "\n".join(f"  • {l}" for l in labels)
                striped = "S"
            else:
                text = value if isinstance(value, str) else ""
                title = f"{tier}  {start}-{end} ms  {text}"
                striped = ""
            left = start / duration * 100
            width = max((end - start) / duration * 100, 0.35)
            blocks.append(
                f'<div class="b{striped}" style="left:{left:.2f}%;width:{width:.2f}%;'
                f'background:{COLORS[i % len(COLORS)]}" title="{html.escape(title)}">'
                f'{html.escape(text[:40])}</div>')
        rows.append(f'<div class="row"><div class="lane">{html.escape(tier)} '
                    f'<span class="n">{len(anns)}</span></div>'
                    f'<div class="track">{"".join(blocks) or "<i>vacía</i>"}</div></div>')

    media = eaf.media_descriptors[0] if eaf.media_descriptors else {}
    media_note = html.escape(media.get("RELATIVE_MEDIA_URL") or media.get("MEDIA_URL") or "?")
    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(source.as_posix())}</title>
<style>
 body{{font:13px system-ui;margin:24px;color:#1c2126}}
 .meta{{color:#5a6570;margin-bottom:14px;line-height:1.5}}
 .row{{display:flex;align-items:center;margin:4px 0}}
 .lane{{width:210px;flex:none;font-family:ui-monospace,monospace}}
 .n{{color:#8a94a0}}
 .track{{position:relative;height:26px;background:#f1f3f5;border-radius:3px;width:100%;overflow:hidden}}
 .b{{position:absolute;top:2px;height:22px;border-radius:3px;color:#fff;font-size:10px;
     line-height:22px;padding:0 3px;white-space:nowrap;overflow:hidden}}
 .b:hover{{outline:2px solid #1c2126;z-index:5}}
 .S{{box-shadow:inset -5px 0 0 rgba(255,255,255,.85)}}
</style>
<h3>{html.escape(source.parent.parent.name)}</h3>
<div class="meta">{total} annotations · {len(tiers)} tiers · {simultaneous_bars} bars
 with simultaneous labels · axis {duration:.0f} ms = <b>{axis_kind}</b><br>
 video: {media_note}<br>
 <b>no video embedded</b>: copy the .eaf to a machine with ELAN for the real check.</div>
{chr(10).join(rows)}
"""
    Path(out_path).write_text(page, encoding="utf-8")
    return {"annotations": total, "tiers": len(tiers), "axis_ms": duration,
            "axis_kind": axis_kind, "simultaneous_bars": simultaneous_bars,
            "empty_tiers": [t for t in tiers if not per_tier[t]]}


if __name__ == "__main__":
    summary = main(sys.argv[1], sys.argv[2])
    print(f"{sys.argv[2]}: {summary['annotations']} annotations, {summary['tiers']} tiers, "
          f"axis {summary['axis_ms']:.0f} ms ({summary['axis_kind']}), "
          f"{summary['simultaneous_bars']} simultaneous-label bars")
