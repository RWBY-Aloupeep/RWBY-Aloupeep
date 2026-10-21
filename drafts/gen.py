"""Build the three profile drafts (paper / minimal / log), each in a dark and a light theme.

Usage:  python drafts/gen.py
Needs numpy, fonttools and brotli; the OFL fonts are fetched once with `npm pack`
into drafts/.fontcache and subset into each SVG, so the text renders identically
everywhere without depending on the viewer's installed fonts.
"""

import base64
import io
import logging
import math
import subprocess
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from fontTools import subset
from fontTools.ttLib import TTFont

import sims

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".fontcache"
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)

FONT_FILES = {  # key: (npm package, file inside it, css family name)
    "serif": ("computer-modern@0.1.3", "package/fonts/cmu-serif-500-roman.woff2", "pf-serif"),
    "serif-i": ("computer-modern@0.1.3", "package/fonts/cmu-serif-500-italic.woff2", "pf-serif-i"),
    "serif-b": ("computer-modern@0.1.3", "package/fonts/cmu-serif-700-roman.woff2", "pf-serif-b"),
    "sans": ("@fontsource/inter@5.3.0", "package/files/inter-latin-400-normal.woff2", "pf-sans"),
    "sans-b": ("@fontsource/inter@5.3.0", "package/files/inter-latin-600-normal.woff2", "pf-sans-b"),
    "mono": ("@fontsource/jetbrains-mono@5.3.0", "package/files/jetbrains-mono-latin-400-normal.woff2", "pf-mono"),
    "mono-b": ("@fontsource/jetbrains-mono@5.3.0", "package/files/jetbrains-mono-latin-700-normal.woff2", "pf-mono-b"),
}
FALLBACK = {"serif": "Georgia, serif", "sans": "'Segoe UI', Helvetica, Arial, sans-serif",
            "mono": "Menlo, Consolas, monospace"}

THEMES = {
    "dark": dict(text="#E6EDF3", muted="#8B949E", faint="#30363D", panel="#161B22",
                 a="#7B5CFF", b="#22D3EE", ok="#3FB950"),
    "light": dict(text="#1F2328", muted="#59636E", faint="#D0D7DE", panel="#F6F8FA",
                  a="#6E4FF6", b="#0891B2", ok="#1A7F37"),
}


# --------------------------------------------------------------------------- fonts

class Font:
    def __init__(self, key):
        pkg, member, self.family = FONT_FILES[key]
        folder = CACHE / pkg.replace("/", "_").replace("@", "_")
        self.path = folder / member
        if not self.path.exists():
            CACHE.mkdir(exist_ok=True)
            out = subprocess.run(["npm", "pack", pkg, "--pack-destination", str(CACHE)],
                                 check=True, capture_output=True, text=True).stdout.split()[-1]
            with tarfile.open(CACHE / out) as tar:
                tar.extractall(folder, filter="data")
        self.css_family = f"{self.family}, {FALLBACK[key.split('-')[0]]}"
        tt = TTFont(self.path)
        self.cmap, self.hmtx, self.upm = tt.getBestCmap(), tt["hmtx"], tt["head"].unitsPerEm

    def width(self, s, size):
        return sum(self.hmtx[self.cmap[ord(" " if c == "\u00a0" else c)]][0] for c in s) * size / self.upm

    def face(self, chars):
        tt = TTFont(self.path)
        opts = subset.Options()
        opts.flavor, opts.layout_features, opts.name_IDs = "woff2", ["kern", "liga"], []
        sub = subset.Subsetter(opts)
        sub.populate(text="".join(sorted(chars)))
        sub.subset(tt)
        buf = io.BytesIO()
        tt.save(buf)
        data = base64.b64encode(buf.getvalue()).decode()
        return f'@font-face{{font-family:{self.family};src:url(data:font/woff2;base64,{data}) format("woff2")}}'


FONTS = {k: Font(k) for k in FONT_FILES}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h, label):
        self.w, self.h, self.label = w, h, label
        self.defs, self.css, self.body = [], [], []
        self.used = defaultdict(set)

    def text(self, x, y, s, font, size, fill, anchor=None, extra=""):
        self.used[font].update(s)
        a = f' text-anchor="{anchor}"' if anchor else ""
        self.body.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONTS[font].css_family}" '
                         f'font-size="{size}" fill="{fill}"{a}{extra}>{esc(s)}</text>')

    def paragraph(self, x, y, width, runs, size, leading, fill, base="serif"):
        """Greedy line breaking, fully justified (word spacing), last line ragged; `~` ties words."""
        words = [(w.replace("~", "\u00a0"), f) for text, f in runs for w in text.split()]
        space = FONTS[base].width(" ", size)
        lines, cur, cur_w = [], [], 0.0
        for w, f in words:
            ww = FONTS[f].width(w, size)
            if cur and cur_w + space + ww > width:
                lines.append(cur)
                cur, cur_w = [], 0.0
            cur_w += (space if cur else 0) + ww
            cur.append((w, f, ww))
        lines.append(cur)
        for i, line in enumerate(lines):
            last = i == len(lines) - 1
            gap = space if last or len(line) < 2 else (width - sum(w for *_, w in line)) / (len(line) - 1)
            xx = x
            for j, (w, f, ww) in enumerate(line):
                if not last and j == len(line) - 1 and len(line) > 1:
                    self.text(x + width, y + i * leading, w, f, size, fill, anchor="end")
                else:
                    self.text(xx, y + i * leading, w, f, size, fill)
                xx += ww + gap
        return y + (len(lines) - 1) * leading

    def render(self):
        faces = "".join(FONTS[f].face(chars) for f, chars in sorted(self.used.items()))
        return "\n".join([
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
            f'height="{self.h}" role="img" aria-label="{esc(self.label)}">',
            "<defs>", *self.defs, f"<style>{faces}", *self.css,
            "@media (prefers-reduced-motion:reduce){*{animation:none!important}}</style>", "</defs>",
            *self.body, "</svg>"]) + "\n"


def mix(c1, c2, t):
    p = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    q = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a + (b - a) * t):02X}" for a, b in zip(p, q))


def rel_path(xy, closed=False):
    """Absolute move then relative segments: compact and interpolable by SMIL."""
    x0, y0 = round(xy[0][0], 1), round(xy[0][1], 1)
    out, px, py = [f"M{x0:g} {y0:g}l"], x0, y0
    for x, y in xy[1:]:
        x, y = round(x, 1), round(y, 1)
        out.append(f"{x - px:.1f} {y - py:.1f}".replace(".0 ", " ").removesuffix(".0"))
        px, py = x, y
    return " ".join(out) + ("z" if closed else "")


def gradient(svg, gid, x1, x2, c):
    svg.defs.append(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{x1}" y1="0" x2="{x2}" y2="0">'
                    f'<stop offset="0" stop-color="{c["a"]}"/><stop offset="1" stop-color="{c["b"]}"/></linearGradient>')


# --------------------------------------------------------------------------- figures

def flow_figure(svg, box, c, scale=46):
    """Streamlines past a cambered Joukowski airfoil, with tracer dots moving along them."""
    x0, y0, w, h = box
    cx, cy = x0 + w / 2 + 8, y0 + h / 2 + 4
    zmin, zmax = (x0 - cx) / scale - 0.3, (x0 + w - cx) / scale + 0.3
    t, mu = 1.0, complex(-0.08, 0.16)
    f = sims.flow.Flow(mu)
    body_psi = f.psi(mu + f.r)
    px = lambda z: (cx + scale * z.real, cy - scale * z.imag)
    gradient(svg, "gflow", x0, x0 + w, c)
    svg.defs.append(f'<clipPath id="cflow"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath>')
    svg.css.append(".tr{fill:none;stroke:url(#gflow);stroke-width:2.4;stroke-linecap:round;"
                   "stroke-dasharray:0 24;animation:flow linear infinite}@keyframes flow{to{stroke-dashoffset:-24}}")
    svg.body.append('<g clip-path="url(#cflow)">')
    for off in [s * d for d in (0.14, 0.34, 0.58, 0.86, 1.18, 1.55, 1.95, 2.4, 2.9) for s in (1, -1)]:
        pts, speed = sims.flow.trace(f, sims.flow.seed(f, body_psi + off), t)
        keep = [(p, v) for p, v in zip(pts, speed) if zmin < sims.flow.to_z(p, t).real < zmax]
        zs = sims.flow.resample([sims.flow.to_z(p, t) for p, _ in keep], 70)
        mean = sum(v for _, v in keep) / len(keep)
        d = rel_path([px(z) for z in zs])
        svg.body.append(f'<path d="{d}" fill="none" stroke="{c["muted"]}" stroke-opacity="0.3" stroke-width="1"/>'
                        f'<path d="{d}" class="tr" pathLength="600" style="animation-duration:{1.1 / mean:.2f}s"/>')
    circle = [mu + f.r * np.exp(2j * math.pi * k / 120) for k in range(120)]
    body = rel_path([px(sims.flow.to_z(z, t)) for z in circle], closed=True)
    svg.body.append(f'<path d="{body}" fill="url(#gflow)" fill-opacity="0.18" stroke="url(#gflow)" '
                    f'stroke-width="1.8" stroke-linejoin="round"/></g>')


def pendulum_figure(svg, pivot, arm, c, cycle=13.0, fps=30, n=7, trace_w=1.2, bob=4.0, fade=True):
    """n double pendulums released 0.0002 rad apart; traces of the outer bobs accumulate."""
    runs, _ = sims.double_pendulums(n=n, duration=cycle, fps=fps)
    px, py = pivot
    timing = f'dur="{cycle}s" repeatCount="indefinite"'
    op = 'values="0;1;1;0" keyTimes="0;0.03;0.94;1"' if fade else 'values="1;1"'
    svg.body.append(f'<g><animate attributeName="opacity" {op} {timing}/>')
    for k, frames in enumerate(runs):
        col = mix(c["a"], c["b"], k / max(n - 1, 1))
        tips = [(px + arm * (math.sin(a) + math.sin(b)), py + arm * (math.cos(a) + math.cos(b))) for a, b in frames]
        s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(np.array(tips), axis=0).T))])
        dash = ";".join(f"{1000 * (1 - v / s[-1]):.0f}" for v in s)
        svg.body.append(f'<path d="{rel_path(tips)}" fill="none" stroke="{col}" stroke-opacity="0.55" '
                        f'stroke-width="{trace_w}" pathLength="1000" stroke-dasharray="1000 1000">'
                        f'<animate attributeName="stroke-dashoffset" values="{dash}" {timing}/></path>')
    for k, frames in enumerate(runs):
        col = mix(c["a"], c["b"], k / max(n - 1, 1))
        r1 = ";".join(f"{-math.degrees(a):.1f}" for a, _ in frames)
        r2 = ";".join(f"{-math.degrees(b - a):.1f}" for a, b in frames)
        rot = lambda v: f'<animateTransform attributeName="transform" type="rotate" values="{v}" {timing}/>'
        line = f'<line x1="0" y1="0" x2="0" y2="{arm}" stroke="{c["text"]}" stroke-opacity="0.35" stroke-width="1.4"/>'
        svg.body.append(f'<g transform="translate({px} {py})"><g>{rot(r1)}{line}'
                        f'<g transform="translate(0 {arm})"><g>{rot(r2)}{line}'
                        f'<circle cy="{arm}" r="{bob}" fill="{col}"/></g>'
                        f'<circle r="{bob * 0.7}" fill="{col}" fill-opacity="0.8"/></g></g></g>')
    svg.body.append(f'<circle cx="{px}" cy="{py}" r="{bob * 0.8}" fill="{c["text"]}"/></g>')


def curtain_figure(svg, box, c, frames, info, gid="gcloth"):
    """Projected wireframe of the mass-spring curtain, morphing through one loop."""
    x0, y0, w, h = box
    yaw, pitch = math.radians(28), math.radians(14)

    def project(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        sx = x * math.cos(yaw) - z * math.sin(yaw)
        depth = x * math.sin(yaw) + z * math.cos(yaw)
        return np.stack([sx, -y * math.cos(pitch) + depth * math.sin(pitch)], -1)

    proj = project(frames)
    lo, hi = proj.reshape(-1, 2).min(0), proj.reshape(-1, 2).max(0)
    scale = 0.86 * min(w / (hi - lo)[0], h / (hi - lo)[1])
    off = np.array([x0 + w / 2, y0 + h / 2]) - scale * (lo + hi) / 2
    pix = proj * scale + off
    gradient(svg, gid, x0, x0 + w, c)

    def mesh(f):
        lines = [f[i, :] for i in range(f.shape[0])] + [f[:, j] for j in range(f.shape[1])]
        return " ".join(rel_path(l) for l in lines)

    values = ";".join(mesh(f) for f in list(pix) + [pix[0]])
    dur = f'dur="{info["period"]}s" repeatCount="indefinite"'
    svg.body.append(f'<path d="{mesh(pix[0])}" fill="none" stroke="url(#{gid})" stroke-width="1.1" '
                    f'stroke-opacity="0.9" stroke-linejoin="round"><animate attributeName="d" values="{values}" {dur}/></path>')
    pins = pix[0, 0, [0, -1]]
    (ax, ay), (bx, by) = pins
    svg.body.append(f'<line x1="{ax - 18:.1f}" y1="{ay:.1f}" x2="{bx + 18:.1f}" y2="{by:.1f}" '
                    f'stroke="{c["muted"]}" stroke-width="2.5" stroke-linecap="round"/>'
                    + "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{c["text"]}"/>' for x, y in pins))


# --------------------------------------------------------------------------- pages

ABSTRACT = ("I explore the integration of numerical and neural methods for physics-based simulation, "
            "bridging graphics, robotics, and artificial intelligence.")


def paper(theme, cloth):
    c = THEMES[theme]
    frames, info = cloth
    svg = Svg(1200, 0, "Diwen (Steven) Xu. Abstract: " + ABSTRACT)
    svg.text(600, 82, "Diwen (Steven) Xu", "serif", 50, c["text"], anchor="middle")
    svg.text(600, 158, "Abstract", "serif-b", 19, c["text"], anchor="middle")
    y = svg.paragraph(190, 192, 820, [(ABSTRACT, "serif")], 19, 28, c["text"])
    svg.paragraph(190, y + 38, 820, [("Keywords", "serif-b"),
                                     ("— physics-based simulation · graphics · robotics · AI", "serif")],
                  19, 28, c["text"])
    top, pw, ph, gap = 300, 340, 250, 30
    boxes = [(60 + i * (pw + gap), top, pw, ph) for i in range(3)]
    for bx, by, bw, bh in boxes:
        svg.body.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="none" stroke="{c["faint"]}"/>')
    flow_figure(svg, boxes[0], c)
    pendulum_figure(svg, (boxes[1][0] + pw / 2, top + ph / 2 - 6), 54, c, trace_w=1, bob=3.2)
    curtain_figure(svg, boxes[2], c, frames, info)
    for (bx, by, bw, bh), label in zip(boxes, ["(a) Potential flow", "(b) Chaotic pendulums", "(c) Cloth in a breeze"]):
        svg.text(bx + bw / 2, by + bh + 30, label, "serif", 18, c["text"], anchor="middle")
    last = svg.paragraph(60, 630, 1080, [
        ("Figure 1.", "serif-b"),
        ("Every panel is simulated, not drawn. (a)~Streamlines of potential flow past a cambered Joukowski "
         "airfoil with Kutta circulation. (b)~Seven double pendulums released from rest 0.0002 rad apart: "
         "indistinguishable for four seconds, then chaotic. (c)~A mass–spring curtain in a periodic breeze, "
         "integrated with position-based dynamics; it locks onto a periodic orbit, so the loop is exact.",
         "serif")], 17, 25, c["text"])
    svg.h = round(last + 24)
    return svg.render()


def minimal(theme):
    c = THEMES[theme]
    svg = Svg(1200, 540, "Diwen (Steven) Xu. Seven double pendulums released 0.0002 rad apart.")
    svg.text(80, 258, "Diwen (Steven) Xu", "sans-b", 48, c["text"], extra=' letter-spacing="-1"')
    svg.text(82, 300, "Physics-Based Simulation · Graphics & Robotics · AI", "sans", 19, c["muted"])
    pendulum_figure(svg, (870, 262), 116, c, trace_w=1.4, bob=5.5)
    svg.text(1160, 522, "seven double pendulums, released 0.0002 rad apart", "mono", 13, c["muted"],
             anchor="end", extra=' fill-opacity="0.8"')
    return svg.render()


def log(theme, cloth):
    c = THEMES[theme]
    frames, info = cloth
    svg = Svg(1200, 520, "Terminal: ./simulate --scene profile. Diwen (Steven) Xu; physics-based simulation; "
                         "graphics & robotics; AI; C++ and Python.")
    svg.body.append(f'<rect x="20.5" y="20.5" width="1159" height="479" rx="12" fill="{c["panel"]}" stroke="{c["faint"]}"/>'
                    f'<line x1="21" y1="62.5" x2="1179" y2="62.5" stroke="{c["faint"]}"/>'
                    + "".join(f'<circle cx="{46 + i * 22}" cy="41.5" r="6.5" fill="{col}"/>'
                              for i, col in enumerate(["#FF5F57", "#FEBC2E", "#28C840"])))
    svg.text(600, 47, "diwen@lab: ~/sim", "mono", 14, c["muted"], anchor="middle")

    per_sec = info["fps"]
    stretch = lambda t: info["stretch"][round(t * per_sec) % len(info["stretch"])] * 100
    tag, val, num, ok = c["a"], c["text"], c["b"], c["ok"]
    lines = [
        [("$ ", ok), ("./simulate --scene profile", val)],
        [("[init ] ", tag), ("Diwen (Steven) Xu", val)],
        [("[field] ", tag), ("physics-based simulation", val)],
        [("[field] ", tag), ("graphics & robotics", val)],
        [("[field] ", tag), ("AI", val)],
        [("[lang ] ", tag), ("C++ · Python", val)],
        [("[scene] ", tag), ("curtain · ", val), (f"{info['nodes']}", num), (" nodes · ", val),
         (f"{info['springs']}", num), (" springs", val)],
        *[[("[step ] ", tag), ("t = ", val), (f"{t:.1f}", num), (" s   max stretch ", val),
           (f"{stretch(t):.2f}%", num), ("   ok", ok)] for t in (1.0, 2.0, 3.0)],
        [("[loop ] ", tag), ("periodic orbit · seam ", val), (f"{info['seam']:.0e}", num)],
    ]
    cycle, x0, y0, lh, size = 16.0, 52, 104, 30, 17
    cw = FONTS["mono"].width("0", size)
    svg.css.append(".cur{animation:blink 1s steps(1) infinite}@keyframes blink{50%{opacity:0}}")
    for i, runs in enumerate(lines + [[("$ ", ok)]]):
        t_on = 0.6 + 0.45 * i
        y = y0 + i * lh
        svg.body.append(f'<g opacity="0"><animate attributeName="opacity" values="0;0;1;1;0" '
                        f'keyTimes="0;{t_on / cycle:.4f};{(t_on + 0.08) / cycle:.4f};0.96;1" '
                        f'dur="{cycle}s" repeatCount="indefinite"/>')
        x = x0
        for text, col in runs:
            svg.text(x, y, text, "mono-b" if col == tag else "mono", size, col, extra=' xml:space="preserve"')
            x += cw * len(text)
        if i == len(lines):
            svg.body.append(f'<rect class="cur" x="{x + 1:.1f}" y="{y - size + 3}" width="{cw:.1f}" '
                            f'height="{size + 2}" fill="{c["text"]}" fill-opacity="0.8"/>')
        svg.body.append("</g>")

    vx, vy, vw, vh = 700, 86, 450, 390
    svg.body.append(f'<rect x="{vx}.5" y="{vy}.5" width="{vw}" height="{vh}" rx="8" fill="none" stroke="{c["faint"]}"/>')
    svg.text(vx + 16, vy + 26, "viewport · curtain", "mono", 13, c["muted"])
    svg.text(vx + vw - 16, vy + 26, f"dt = 1/{round(1 / info['dt'])} s", "mono", 13, c["muted"], anchor="end")
    curtain_figure(svg, (vx + 20, vy + 40, vw - 40, vh - 56), c, frames, info)
    return svg.render()


def main():
    cloth = sims.curtain(nx=13, ny=10, fps=10)
    print(f"curtain: seam {cloth[1]['seam']:.1e}, max stretch {cloth[1]['stretch'].max():.3%}")
    for name, build in [("paper", lambda t: paper(t, cloth)), ("minimal", minimal), ("log", lambda t: log(t, cloth))]:
        for theme in THEMES:
            out = HERE / name / f"{name}-{theme}.svg"
            out.parent.mkdir(exist_ok=True)
            out.write_text(build(theme), encoding="utf-8")
            print(f"{out.relative_to(HERE)}: {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
