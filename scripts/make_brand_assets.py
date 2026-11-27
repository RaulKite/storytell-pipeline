"""Generate the README brand SVGs (light + dark variants share one geometry).

Run from the repository root: python scripts/make_brand_assets.py docs/assets/brand
Every picture is synthetic: invented values, no recording.
"""
import math
import random
import sys
from pathlib import Path

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

THEMES = {
    "light": dict(
        bg="#ffffff", plate_stroke="#d0d7de", panel="#f6f8fa", border="#d8dee4",
        text="#1f2328", text2="#59636e", grid="#d0d7de", faint="#eaeef2",
        speech="#0b6bcb", language="#1a7f37", acoustics="#8250df",
        vision="#c4450c", speakers="#9a6700", stories="#57606a",
        speech_soft="#ddeeff", language_soft="#dcf3e3", vision_soft="#fde4d6",
        speakers_soft="#fbefc9", stories_soft="#eef1f4", accent="#cf222e",
    ),
    "dark": dict(
        bg="#0d1117", plate_stroke="#30363d", panel="#161b22", border="#2a313c",
        text="#e6edf3", text2="#9198a1", grid="#3d444d", faint="#21262d",
        speech="#58a6ff", language="#3fb950", acoustics="#bc8cff",
        vision="#f0883e", speakers="#e3b341", stories="#9198a1",
        speech_soft="#12263f", language_soft="#10291a", vision_soft="#33200f",
        speakers_soft="#2e2508", stories_soft="#1c2128", accent="#ff7b72",
    ),
}


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=12, fill="#000", weight=400, anchor="start", family=FONT,
         style="", extra=""):
    st = f' font-style="{style}"' if style else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{st}{extra}>{esc(s)}</text>')


def svg_open(w, h, title, desc, tid):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="{tid}-t {tid}-d">\n'
            f'<title id="{tid}-t">{esc(title)}</title>\n<desc id="{tid}-d">{esc(desc)}</desc>\n')


# ----------------------------------------------------------------------------- mark
def mark(c, x, y, s):
    """The logo mark: waveform, pose figure and text lines converging on one time axis.

    Drawn in a 96-unit box, scaled by s/96, placed at (x, y)."""
    k = s / 96
    g = [f'<g transform="translate({x},{y}) scale({k:.4f})">']
    g.append(f'<rect x="0" y="0" width="96" height="96" rx="22" fill="{c["panel"]}" stroke="{c["plate_stroke"]}" stroke-width="1.5"/>')
    # waveform (speech)
    bars = [6, 13, 22, 13, 8]
    for i, hh in enumerate(bars):
        bx = 15 + i * 5.2
        g.append(f'<rect x="{bx:.1f}" y="{40 - hh/2:.1f}" width="3" height="{hh}" rx="1.5" fill="{c["speech"]}"/>')
    # pose figure (body)
    px, py = 48, 26
    st = f'stroke="{c["vision"]}" stroke-width="3" stroke-linecap="round" fill="none"'
    g.append(f'<circle cx="{px}" cy="{py}" r="4.6" {st}/>')
    g.append(f'<path d="M{px} {py+5} V{py+19} M{px-8} {py+9} H{px+8} M{px-8} {py+9} L{px-11} {py+16} M{px+8} {py+9} L{px+11} {py+16} M{px} {py+19} L{px-6} {py+29} M{px} {py+19} L{px+6} {py+29}" {st}/>')
    # text lines (language)
    for i, ww in enumerate([20, 13, 17]):
        g.append(f'<rect x="64" y="{31 + i*7.5:.1f}" width="{ww}" height="3.6" rx="1.8" fill="{c["language"]}"/>')
    # shared time axis + drop lines
    g.append(f'<path d="M14 74 H82" stroke="{c["text2"]}" stroke-width="2.4" stroke-linecap="round"/>')
    for i, tx in enumerate([25, 48, 72]):
        col = [c["speech"], c["vision"], c["language"]][i]
        g.append(f'<path d="M{tx} 57 V71" stroke="{col}" stroke-width="1.6" stroke-dasharray="2 2.4" stroke-linecap="round"/>')
        g.append(f'<circle cx="{tx}" cy="74" r="3.4" fill="{col}"/>')
    g.append('</g>')
    return "\n".join(g)


def logo(theme):
    c = THEMES[theme]
    out = [svg_open(96, 96, "Storytel Pipeline logo",
                    "Original mark: a speech waveform, a pose figure and lines of text drop onto one shared time axis.",
                    "logo")]
    out.append(mark(c, 0, 0, 96))
    out.append("</svg>\n")
    return "\n".join(out)


# ----------------------------------------------------------------------------- banner
def banner(theme):
    c = THEMES[theme]
    W, H = 1280, 300
    o = [svg_open(W, H, "Storytel Pipeline",
                  "Banner: the Storytel Pipeline mark and name with the tagline video recordings to traceable, "
                  "time-aligned multimodal datasets, over a faint score of speech, body and text tiers.",
                  "banner")]
    o.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" fill="{c["bg"]}" stroke="{c["plate_stroke"]}" stroke-width="1.5"/>')
    # faint multimodal score on the right
    rnd = random.Random(11)
    x0, x1 = 760, 1230
    tiers = [("speech", 70), ("language", 112), ("acoustics", 154), ("vision", 196), ("speakers", 238)]
    o.append('<g opacity="0.9">')
    for name, ty in tiers:
        col = c[name]
        o.append(f'<path d="M{x0} {ty+14} H{x1}" stroke="{c["faint"]}" stroke-width="1"/>')
        if name == "acoustics":
            pts, x, on = [], x0, True
            segs = []
            while x < x1:
                ln = rnd.randint(40, 90)
                if on:
                    seg = []
                    for xx in range(x, min(x + ln, x1), 4):
                        yy = ty + 4 + 7 * math.sin(xx / 17.0) + 3 * math.sin(xx / 5.3)
                        seg.append(f"{xx},{yy:.1f}")
                    if len(seg) > 1:
                        segs.append(seg)
                x += ln + (rnd.randint(14, 30) if on else 0)
                on = not on or True
            for seg in segs:
                o.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="{col}" stroke-width="2" stroke-linecap="round" opacity="0.55"/>')
        elif name == "vision":
            for fx in range(x0 + 20, x1, 78):
                o.append(f'<g stroke="{col}" stroke-width="1.8" stroke-linecap="round" fill="none" opacity="0.55">'
                         f'<circle cx="{fx}" cy="{ty-2}" r="3"/><path d="M{fx} {ty+1} V{ty+10} M{fx-5} {ty+4} H{fx+5} M{fx} {ty+10} L{fx-4} {ty+17} M{fx} {ty+10} L{fx+4} {ty+17}"/></g>')
        else:
            x = x0 + rnd.randint(0, 20)
            while x < x1 - 20:
                ln = rnd.randint(16, 70) if name != "speakers" else rnd.randint(6, 14)
                ln = min(ln, x1 - x)
                hgt = 8 if name != "speakers" else rnd.randint(4, 16)
                yy = ty + 8 - hgt / 2 if name != "speakers" else ty + 14 - hgt
                o.append(f'<rect x="{x}" y="{yy:.1f}" width="{ln}" height="{hgt}" rx="{min(4, hgt/2):.1f}" fill="{col}" opacity="0.5"/>')
                x += ln + rnd.randint(6, 26)
    o.append('</g>')
    # playhead across the score
    o.append(f'<path d="M1010 52 V262" stroke="{c["accent"]}" stroke-width="1.6" stroke-dasharray="4 4" opacity="0.8"/>')
    o.append(f'<circle cx="1010" cy="52" r="4" fill="{c["accent"]}" opacity="0.9"/>')
    # fade the score into the plate on its left edge
    o.append(f'<defs><linearGradient id="fade-{theme}" x1="0" x2="1"><stop offset="0" stop-color="{c["bg"]}" stop-opacity="1"/><stop offset="1" stop-color="{c["bg"]}" stop-opacity="0"/></linearGradient></defs>')
    o.append(f'<rect x="740" y="40" width="120" height="230" fill="url(#fade-{theme})"/>')
    # mark + words
    o.append(mark(c, 64, 82, 136))
    o.append(text(236, 142, "Storytel Pipeline", 54, c["text"], 700, extra=' letter-spacing="-1"'))
    o.append(text(238, 184, "Video recordings → traceable, time-aligned", 22, c["text2"]))
    o.append(text(238, 214, "multimodal datasets", 22, c["text2"]))
    # modality chips
    chips = [("words", "speech"), ("voice", "acoustics"), ("body", "vision"), ("faces", "speakers"), ("stories", "stories")]
    cx = 238
    for label, key in chips:
        w = 14 + len(label) * 8.2 + 14
        o.append(f'<rect x="{cx}" y="232" width="{w:.0f}" height="24" rx="12" fill="none" stroke="{c[key]}" stroke-width="1.4"/>')
        o.append(f'<circle cx="{cx+13}" cy="244" r="3.5" fill="{c[key]}"/>')
        o.append(text(cx + 22, 249, label, 13, c["text"]))
        cx += w + 8
    o.append("</svg>\n")
    return "\n".join(o)


# ----------------------------------------------------------------------------- modalities
def modalities(theme):
    c = THEMES[theme]
    W, H = 1200, 572
    AX0, AX1 = 280, 880   # 0..60 s -> 10 px per second
    def X(t): return AX0 + t * 10
    lane_top, lane_h, gap = 112, 56, 10
    T = 33.0   # the highlighted moment
    o = [svg_open(W, H, "Six modalities on one clock",
                  "Synthetic illustration with invented values, not a recording. Six lanes - speech, language, "
                  "acoustics, vision, speakers and stories - share one axis in seconds from video start. A marker "
                  "at 33 seconds reads every lane at the same moment, and the lanes feed one dataset directory of "
                  "Parquet tables, retained tool output and stage caches, provenance and an optional ELAN export.",
                  "mod")]
    o.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" fill="{c["bg"]}" stroke="{c["plate_stroke"]}" stroke-width="1.5"/>')
    o.append(text(32, 46, "Six modalities, one clock", 24, c["text"], 700, extra=' letter-spacing="-0.4"'))
    o.append(text(32, 70, "Every table is time-stamped in seconds from video start, so one moment can be read across all of them.", 13, c["text2"]))
    # axis
    ay = 98
    o.append(f'<path d="M{AX0} {ay} H{AX1}" stroke="{c["grid"]}" stroke-width="1.2"/>')
    for t in range(0, 61, 10):
        o.append(f'<path d="M{X(t)} {ay-4} V{ay+3}" stroke="{c["grid"]}" stroke-width="1.2"/>')
        o.append(text(X(t), ay - 8, f"{t} s" if t in (0, 60) else str(t), 10.5, c["text2"], anchor="middle"))
    lanes = [
        ("speech", "Speech", "WhisperX · Nemotron", "speech/words.parquet"),
        ("language", "Language", "spaCy · translation endpoint", "linguistic/source/tokens.parquet"),
        ("acoustics", "Acoustics", "Praat via Parselmouth", "acoustic/frame_features.parquet"),
        ("vision", "Vision", "OpenPose · YOLO (optional)", "pose/body.parquet"),
        ("speakers", "Speakers", "TalkNet (optional)", "speaker/active_speaker_frames.parquet"),
        ("stories", "Stories", "LLM endpoint", "stories/stories.parquet"),
    ]
    rnd = random.Random(7)
    segments = [(1.0, 9.5, 0), (11.0, 21.5, 1), (23.0, 36.5, 0), (38.0, 47.0, 1), (48.5, 58.0, 0)]
    for i, (key, name, tool, path) in enumerate(lanes):
        y = lane_top + i * (lane_h + gap)
        col = c[key]
        # label column
        o.append(f'<rect x="32" y="{y+8}" width="4" height="{lane_h-16}" rx="2" fill="{col}"/>')
        o.append(text(46, y + 22, name, 15, c["text"], 700))
        o.append(text(46, y + 37, tool, 11.5, c["text2"]))
        o.append(text(46, y + 50, path, 9.5, c["text2"], family=MONO))
        # lane plate
        o.append(f'<rect x="{AX0-8}" y="{y}" width="{AX1-AX0+16}" height="{lane_h}" rx="10" fill="{c["panel"]}" stroke="{c["border"]}" stroke-width="1"/>')
        top, mid = y + 10, y + lane_h / 2
        if key == "speech":
            for (a, b, spk) in segments:
                t = a + 0.3
                while t < b - 0.4:
                    w = rnd.uniform(0.35, 0.9)
                    o.append(f'<rect x="{X(t):.1f}" y="{top+2}" width="{w*10:.1f}" height="12" rx="3" fill="{col}" opacity="0.85"/>')
                    t += w + rnd.uniform(0.15, 0.6)
                fill = col if spk == 0 else "none"
                tc = c["bg"] if spk == 0 else col
                o.append(f'<rect x="{X(a):.1f}" y="{top+20}" width="{(b-a)*10:.1f}" height="18" rx="5" fill="{fill}" stroke="{col}" stroke-width="1.3"/>')
                o.append(text(X(a) + 6, top + 33, f"SPEAKER_0{spk}", 9.5, tc, 700, family=MONO))
        elif key == "language":
            for (a, b, spk) in segments:
                t = a + 0.3
                while t < b - 0.4:
                    w = rnd.uniform(0.3, 0.7)
                    o.append(f'<rect x="{X(t):.1f}" y="{top+3}" width="{w*10:.1f}" height="9" rx="2" fill="{col}" opacity="0.8"/>')
                    t += w + rnd.uniform(0.2, 0.5)
                o.append(f'<rect x="{X(a):.1f}" y="{top+20}" width="{(b-a)*10:.1f}" height="18" rx="5" fill="{c["language_soft"]}" stroke="{col}" stroke-width="1"/>')
                o.append(text(X(a) + 6, top + 33, "EN translation", 9.5, col, 600))
        elif key == "acoustics":
            for (a, b, _) in segments:
                pts = []
                t = a + 0.2
                base = mid + rnd.uniform(-3, 3)
                while t < b - 0.2:
                    yy = base + 8 * math.sin(t * 0.9 + a) + 4 * math.sin(t * 2.7)
                    pts.append(f"{X(t):.1f},{yy:.1f}")
                    t += 0.25
                o.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        elif key == "vision":
            for t in range(3, 60, 8):
                fx, fy = X(t), top + 4
                o.append(f'<g stroke="{col}" stroke-width="1.7" stroke-linecap="round" fill="none">'
                         f'<circle cx="{fx}" cy="{fy+2}" r="2.8"/>'
                         f'<path d="M{fx} {fy+5} V{fy+13} M{fx-6} {fy+8} L{fx+6} {fy+8} M{fx} {fy+13} L{fx-4} {fy+20} M{fx} {fy+13} L{fx+4} {fy+20}"/></g>')
            for (a, b, lab, dy) in [(0.5, 34.0, "person track p1", 0), (28.0, 59.0, "person track p2", 0)]:
                yy = top + 27 + (0 if lab.endswith("p1") else 0)
            o.append(f'<rect x="{X(0.5):.1f}" y="{top+28}" width="{33.5*10:.1f}" height="9" rx="4.5" fill="{c["vision_soft"]}" stroke="{col}" stroke-width="1"/>')
            o.append(f'<rect x="{X(36.0):.1f}" y="{top+28}" width="{23.0*10:.1f}" height="9" rx="4.5" fill="{c["vision_soft"]}" stroke="{col}" stroke-width="1"/>')
            o.append(text(X(0.5) + 6, top + 35.5, "track 1", 8.5, col, 700))
            o.append(text(X(36.0) + 6, top + 35.5, "track 2", 8.5, col, 700))
        elif key == "speakers":
            t = 0.5
            while t < 59.5:
                if 15.5 <= t <= 20.5:
                    t += 1.0
                    continue
                speaking = any(a <= t <= b and s == 0 for a, b, s in segments)
                hgt = rnd.uniform(14, 30) if speaking else rnd.uniform(2, 7)
                o.append(f'<rect x="{X(t):.1f}" y="{y+lane_h-10-hgt:.1f}" width="6" height="{hgt:.1f}" rx="2" fill="{col}" opacity="0.85"/>')
                t += 1.0
            o.append(f'<rect x="{X(15.6):.1f}" y="{top+6}" width="{5.0*10:.1f}" height="30" rx="6" fill="none" stroke="{c["text2"]}" stroke-width="1" stroke-dasharray="3 3"/>')
            o.append(text(X(18.1), top + 25, "no face", 9.5, c["text2"], anchor="middle"))
        elif key == "stories":
            a, b = 14.0, 55.0
            o.append(f'<rect x="{X(a):.1f}" y="{top+2}" width="{(b-a)*10:.1f}" height="32" rx="8" fill="{c["stories_soft"]}" stroke="{col}" stroke-width="1.4" stroke-dasharray="6 4"/>')
            o.append(text(X(36.5), top + 16, "story candidate", 12, c["text"], 700))
            o.append(text(X(36.5), top + 29, "LLM span, not a validated detection", 9.5, c["text2"]))
    lanes_bottom = lane_top + 6 * (lane_h + gap) - gap
    # playhead
    px = X(T)
    o.append(f'<path d="M{px} {ay+6} V{lanes_bottom+6}" stroke="{c["accent"]}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    o.append(f'<rect x="{px-30}" y="{lanes_bottom+8}" width="60" height="20" rx="10" fill="{c["accent"]}"/>')
    o.append(text(px, lanes_bottom + 22, "t = 33 s", 11, "#ffffff", 700, anchor="middle"))
    # right column: readout card
    RX, RW = 912, 256
    ry = lane_top
    rh = 236
    o.append(f'<rect x="{RX}" y="{ry}" width="{RW}" height="{rh}" rx="12" fill="{c["panel"]}" stroke="{c["border"]}" stroke-width="1"/>')
    o.append(f'<rect x="{RX+16}" y="{ry+16}" width="60" height="20" rx="10" fill="{c["accent"]}"/>')
    o.append(text(RX + 46, ry + 30, "t = 33 s", 11, "#ffffff", 700, anchor="middle"))
    o.append(text(RX + 86, ry + 31, "one row per table", 12, c["text2"]))
    reads = [
        ("speech", "SPEAKER_00 turn, aligned word"),
        ("language", "source tokens, EN segment"),
        ("acoustics", "f0 and intensity frame"),
        ("vision", "BODY_25 keypoints, track 1"),
        ("speakers", "visible face scored active"),
        ("stories", "inside a story candidate"),
    ]
    for j, (key, s) in enumerate(reads):
        yy = ry + 62 + j * 28
        o.append(f'<circle cx="{RX+22}" cy="{yy-4}" r="4.5" fill="{c[key]}"/>')
        o.append(text(RX + 34, yy, s, 12, c["text"]))
    # arrow from lanes to dataset card
    dy = ry + rh + 22
    dh = lanes_bottom - dy
    o.append(f'<rect x="{RX}" y="{dy}" width="{RW}" height="{dh}" rx="12" fill="{c["panel"]}" stroke="{c["border"]}" stroke-width="1"/>')
    o.append(f'<path d="M{AX1+12} {dy+dh/2} H{RX-6}" stroke="{c["text2"]}" stroke-width="1.6"/>')
    o.append(f'<path d="M{RX-12} {dy+dh/2-5} L{RX-4} {dy+dh/2} L{RX-12} {dy+dh/2+5} Z" fill="{c["text2"]}"/>')
    o.append(text(RX + 16, dy + 26, "one dataset directory", 14, c["text"], 700))
    o.append(text(RX + 16, dy + 43, "<video_id>/", 10.5, c["text2"], family=MONO))
    items = ["normalized Parquet tables", "tool output and stage caches", "manifest, status, provenance", "optional ELAN .eaf, last"]
    for j, a in enumerate(items):
        yy = dy + 68 + j * 19
        o.append(f'<circle cx="{RX+21}" cy="{yy-4}" r="3" fill="{c["text2"]}"/>')
        o.append(text(RX + 32, yy, a, 12, c["text"]))
    # footer
    o.append(text(32, H - 22, "Synthetic illustration: invented values, no recording.", 11, c["text2"], style="italic"))
    o.append(text(W - 32, H - 22, "Optional stages are marked; paths are dataset-relative.", 11, c["text2"], anchor="end"))
    o.append("</svg>\n")
    return "\n".join(o)


# ----------------------------------------------------------------------------- how it works
def flow(theme):
    c = THEMES[theme]
    W, H = 1200, 300
    o = [svg_open(W, H, "How one video becomes a dataset",
                  "Five steps: probe the video and extract audio with ffprobe and ffmpeg; run each tool in its own "
                  "isolated uv environment or binary; normalize results to Parquet and retain tool output; validate and "
                  "record status, manifest and provenance so valid results are reused; optionally export an ELAN file, with an HTML preview helper.",
                  "flow")]
    o.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="18" fill="{c["bg"]}" stroke="{c["plate_stroke"]}" stroke-width="1.5"/>')
    o.append(text(32, 46, "How one video becomes a dataset", 24, c["text"], 700, extra=' letter-spacing="-0.4"'))
    o.append(text(32, 70, "Sequential: one video, one stage at a time. Valid earlier results are reused, failures are recorded, not hidden.", 13, c["text2"]))
    steps = [
        ("1", "Probe", ["ffprobe metadata,", "frame index,", "ffmpeg WAV"], "speech"),
        ("2", "Run tools", ["7 isolated uv envs", "+ OpenPose binary,", "one at a time"], "vision"),
        ("3", "Normalize", ["typed Parquet,", "seconds from start,", "tool output retained"], "language"),
        ("4", "Validate", ["status.json, manifest,", "provenance; reuse", "what is still valid"], "acoustics"),
        ("5", "Inspect", ["optional ELAN .eaf,", "HTML preview helper,", "explanation guides"], "speakers"),
    ]
    n = len(steps)
    x0, bw, gapx, by, bh = 32, 196, 40, 100, 160
    for i, (num, title, lines, key) in enumerate(steps):
        x = x0 + i * (bw + gapx)
        col = c["speech"]
        o.append(f'<rect x="{x}" y="{by}" width="{bw}" height="{bh}" rx="14" fill="{c["panel"]}" stroke="{c["border"]}" stroke-width="1"/>')
        o.append(f'<rect x="{x}" y="{by}" width="{bw}" height="5" rx="2.5" fill="{col}"/>')
        o.append(f'<circle cx="{x+30}" cy="{by+40}" r="15" fill="none" stroke="{col}" stroke-width="2"/>')
        o.append(text(x + 30, by + 45.5, num, 15, col, 700, anchor="middle"))
        o.append(text(x + 54, by + 46, title, 17, c["text"], 700))
        for j, ln in enumerate(lines):
            o.append(text(x + 18, by + 84 + j * 20, ln, 12.5, c["text2"]))
        if i < n - 1:
            ax = x + bw + 6
            o.append(f'<path d="M{ax} {by+bh/2} H{ax+gapx-14}" stroke="{c["text2"]}" stroke-width="1.6"/>')
            o.append(f'<path d="M{ax+gapx-18} {by+bh/2-5} L{ax+gapx-10} {by+bh/2} L{ax+gapx-18} {by+bh/2+5} Z" fill="{c["text2"]}"/>')
    o.append(text(32, H - 18, "Stage graph, environments and reuse rules: docs/architecture.md", 11, c["text2"], style="italic"))
    o.append("</svg>\n")
    return "\n".join(o)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "logo.svg").write_text(logo("dark"), encoding="utf-8")
    for th in THEMES:
        (out / f"banner-{th}.svg").write_text(banner(th), encoding="utf-8")
        (out / f"modalities-{th}.svg").write_text(modalities(th), encoding="utf-8")
        (out / f"flow-{th}.svg").write_text(flow(th), encoding="utf-8")


if __name__ == "__main__":
    main()
