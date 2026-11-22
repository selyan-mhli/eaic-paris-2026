"""Generate the README charts as static SVG, one light and one dark version each.

Values are public leaderboard scores logged in docs/experiences.md and docs/CLOTURE-20261122.md.
Run: python3 docs/figures/make_figures.py
"""

from pathlib import Path

OUT = Path(__file__).parent

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", series="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", series="#3987e5"),
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

T1 = [  # (short label, public mIoU), in submission order
    ("synth head", 0.5307), ("holes head", 0.5572), ("pixseg SS", 0.5845), ("ConvNeXt", 0.6137),
    ("4×4k ens.", 0.6377), ("ens. E", 0.6739), ("SS 6k", 0.6788), ("TM F 6k", 0.6880),
    ("SS 9 seeds", 0.6886), ("final Y", 0.6893),
]
T1_TOP = 0.803  # best team on the public leaderboard, Sunday 03:17

T3_STRENGTH = [(1.25, 0.0567), (1.5, 0.0827), (1.65, 0.0923), (1.75, 0.0941), (2.0, 0.0792)]
T3_TEMP = [(0.8, 0.0898), (0.9, 0.0986), (0.95, 0.0972), (1.0, 0.0941), (1.2, 0.0754)]


def fmt(v):
    return f"{v:.4f}".rstrip("0") if v < 0.2 else f"{v:.3f}"


def svg_open(w, h, t, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'font-family=\'{FONT}\' role="img" aria-label="{title}">',
        f"<title>{title}</title>",
        f'<rect width="{w}" height="{h}" rx="8" fill="{t["surface"]}"/>',
    ]


def line_panel(t, x0, y0, w, h, xs, ys, xlabels, ymin, ymax, yticks, best_i, label_is=()):
    """One line chart: hairline grid, 2px line, 8px markers with a 2px surface ring."""
    s = []
    px = lambda i: x0 + 24 + (w - 48) * xs[i]  # inset keeps end labels off the y ticks
    py = lambda v: y0 + h - h * (v - ymin) / (ymax - ymin)
    for v in yticks:
        y = py(v)
        s.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" font-size="11" fill="{t["muted"]}" text-anchor="end" '
                 f'style="font-variant-numeric: tabular-nums">{v:.2f}</text>')
    s.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="{t["axis"]}" stroke-width="1"/>')
    for i, lab in enumerate(xlabels):
        s.append(f'<text x="{px(i):.1f}" y="{y0 + h + 18}" font-size="11" fill="{t["ink2"]}" '
                 f'text-anchor="middle">{lab}</text>')
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{t["series"]}" stroke-width="2" '
             f'stroke-linejoin="round" stroke-linecap="round"/>')
    for i, v in enumerate(ys):
        r = 5 if i == best_i else 4
        s.append(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="{r}" fill="{t["series"]}" '
                 f'stroke="{t["surface"]}" stroke-width="2"/>')
    for i in set(label_is) | {best_i}:
        weight = "600" if i == best_i else "400"
        s.append(f'<text x="{px(i):.1f}" y="{py(ys[i]) - 12:.1f}" font-size="12" font-weight="{weight}" '
                 f'fill="{t["ink"]}" text-anchor="middle">{fmt(ys[i])}</text>')
    return s


def chart_t1(t):
    w, h = 760, 330
    title = "Task 1 public mIoU, submission by submission"
    s = svg_open(w, h, t, title)
    s.append(f'<text x="24" y="30" font-size="15" font-weight="600" fill="{t["ink"]}">{title}</text>')
    s.append(f'<text x="24" y="50" font-size="12" fill="{t["ink2"]}">Only submissions that set a new best; '
             f'the dashed line is the best team at Sunday 03:17.</text>')
    x0, y0, pw, ph = 64, 78, 660, 200
    n = len(T1)
    xs = [i / (n - 1) for i in range(n)]
    ymin, ymax = 0.50, 0.82
    py = lambda v: y0 + ph - ph * (v - ymin) / (ymax - ymin)
    s += line_panel(t, x0, y0, pw, ph, xs, [v for _, v in T1], [lab for lab, _ in T1],
                    ymin, ymax, [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80], n - 1, label_is=(0, 3, 5))
    y = py(T1_TOP)
    s.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["muted"]}" '
             f'stroke-width="1.5" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{x0 + pw}" y="{y - 6:.1f}" font-size="11" fill="{t["ink2"]}" text-anchor="end">'
             f'best team 0.803</text>')
    s.append("</svg>")
    return "\n".join(s)


def chart_t3(t):
    w, h = 760, 330
    title = "Task 3 public score of the head push, two sweeps"
    s = svg_open(w, h, t, title)
    s.append(f'<text x="24" y="30" font-size="15" font-weight="600" fill="{t["ink"]}">{title}</text>')
    s.append(f'<text x="24" y="50" font-size="12" fill="{t["ink2"]}">Left: push strength at T = 1.0. '
             f'Right: head temperature at strength ×1.75. Top teams scored 0.43-0.48.</text>')
    ymin, ymax, ticks = 0.04, 0.11, [0.04, 0.06, 0.08, 0.10]
    for k, (data, xlab, sub) in enumerate([
        (T3_STRENGTH, lambda v: f"×{v:g}", "Push strength"),
        (T3_TEMP, lambda v: f"{v:g}", "Head temperature T"),
    ]):
        x0 = 64 + k * 360
        s.append(f'<text x="{x0}" y="84" font-size="12" font-weight="600" fill="{t["ink2"]}">{sub}</text>')
        lo, hi = data[0][0], data[-1][0]
        xs = [(x - lo) / (hi - lo) for x, _ in data]
        ys = [v for _, v in data]
        best = max(range(len(ys)), key=ys.__getitem__)
        s += line_panel(t, x0, 100, 290, 180, xs, ys, [xlab(x) for x, _ in data],
                        ymin, ymax, ticks, best, label_is=(0,))
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    for mode, t in THEMES.items():
        (OUT / f"t1_progression_{mode}.svg").write_text(chart_t1(t), encoding="utf-8")
        (OUT / f"t3_sweeps_{mode}.svg").write_text(chart_t3(t), encoding="utf-8")
    print("written to", OUT)
