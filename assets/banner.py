"""Generate the profile banner: potential flow past a morphing Joukowski body.

Every curve in the banner is computed, not drawn. In the zeta-plane we solve
uniform flow past a cylinder with circulation (Kutta condition at zeta = 1);
the conformal map z = zeta + t * lam^2 / zeta then carries the cylinder and
its streamlines into the physical plane. Sweeping t and the cylinder centre
morphs a circle into a symmetric and then a cambered airfoil, and because the
stream function is invariant under conformal maps, every frame is an exact
solution of Laplace's equation.

Usage:  python assets/banner.py   (writes banner-{dark,light}.svg)
Needs fonttools and brotli. The OFL fonts (Inter, JetBrains Mono) are fetched
once with `npm pack` into assets/.fontcache and subset into the SVG, so the
text renders the same everywhere.
"""

import base64
import cmath
import io
import math
import subprocess
import tarfile
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent
W, H = 1200, 540                 # SVG canvas (px)
CX, CY, SCALE = 880, 272, 78     # pixel position of the z-plane origin, px per unit
U, ALPHA, LAM = 1.0, math.radians(7), 1.0

# (t, centre of the zeta-plane circle): circle -> symmetric airfoil -> cambered airfoil
FRAMES = [(0.0, complex(-0.10, 0.00)),
          (1.0, complex(-0.10, 0.00)),
          (1.0, complex(-0.08, 0.16))]

# stream-function offsets from the body streamline; one entry per streamline
OFFSETS = [s * d for d in (0.15, 0.38, 0.65, 0.95, 1.3, 1.7, 2.15, 2.65, 3.2) for s in (1, -1)]
X_MIN, X_MAX = (480 - CX) / SCALE, (W + 20 - CX) / SCALE   # visible stretch of each streamline
X_SEED = -16.0
N_LINE, N_BODY = 90, 160
CYCLE = 15                       # seconds for one full morph cycle

THEMES = {
    "dark": dict(text="#E6EDF3", muted="#8B949E", a="#7B5CFF", b="#22D3EE", line_op=0.22, body_op=0.18),
    "light": dict(text="#1F2328", muted="#59636E", a="#6E4FF6", b="#0891B2", line_op=0.2, body_op=0.12),
}

FONTS = {  # key: (npm package, file inside it, css family, fallback)
    "sans": ("@fontsource/inter@5.3.0", "package/files/inter-latin-400-normal.woff2", "pf-sans",
             "'Segoe UI', Helvetica, Arial, sans-serif"),
    "sans-b": ("@fontsource/inter@5.3.0", "package/files/inter-latin-600-normal.woff2", "pf-sans-b",
               "'Segoe UI', Helvetica, Arial, sans-serif"),
    "mono": ("@fontsource/jetbrains-mono@5.3.0", "package/files/jetbrains-mono-latin-400-normal.woff2", "pf-mono",
             "Menlo, Consolas, monospace"),
}


# --------------------------------------------------------------------------- physics

class Flow:
    """Cylinder flow in the zeta-plane with Kutta circulation."""

    def __init__(self, mu):
        self.mu = mu
        self.r = abs(LAM - mu)
        te = LAM - mu
        a = U * (cmath.exp(-1j * ALPHA) - self.r ** 2 * cmath.exp(1j * ALPHA) / te ** 2)
        self.gamma = (2j * math.pi * te * a).real   # makes dW/dzeta vanish at zeta = LAM

    def potential(self, zeta):
        d = zeta - self.mu
        return (U * (d * cmath.exp(-1j * ALPHA) + self.r ** 2 * cmath.exp(1j * ALPHA) / d)
                + 1j * self.gamma / (2 * math.pi) * cmath.log(d))

    def velocity(self, zeta):
        d = zeta - self.mu
        dw = (U * (cmath.exp(-1j * ALPHA) - self.r ** 2 * cmath.exp(1j * ALPHA) / d ** 2)
              + 1j * self.gamma / (2 * math.pi * d))
        return dw.conjugate()

    def psi(self, zeta):
        return self.potential(zeta).imag


def to_z(zeta, t):
    # Joukowski map, then rotate so the free stream is horizontal (airfoil pitched nose-up)
    return (zeta + t * LAM ** 2 / zeta) * cmath.exp(-1j * ALPHA)


def to_px(z):
    return CX + SCALE * z.real, CY - SCALE * z.imag


def seed(flow, target):
    """Point far upstream whose stream function equals `target` (psi is monotone in y there)."""
    x = X_SEED * math.cos(ALPHA)  # upstream along the free-stream direction
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if flow.psi(complex(x, mid) * cmath.exp(1j * ALPHA)) < target:
            lo = mid
        else:
            hi = mid
    return complex(x, 0.5 * (lo + hi)) * cmath.exp(1j * ALPHA)


def trace(flow, start, t, ds=0.01):
    """RK4 on the unit direction field; keeps the stretch that is visible on the canvas."""
    def f(p):
        v = flow.velocity(p)
        return v / max(abs(v), 1e-9)

    pts, speeds, p = [], [], start
    for _ in range(20000):
        z = to_z(p, t)
        if z.real > X_MAX:
            break
        if z.real > X_MIN:
            pts.append(z)
            speeds.append(abs(flow.velocity(p)))
        k1 = f(p)
        k2 = f(p + 0.5 * ds * k1)
        k3 = f(p + 0.5 * ds * k2)
        k4 = f(p + ds * k3)
        p += ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return pts, speeds


def resample(pts, n):
    """n points evenly spaced in arc length, so every frame has matching path commands."""
    acc = [0.0]
    for a, b in zip(pts, pts[1:]):
        acc.append(acc[-1] + abs(b - a))
    out, j = [], 0
    for i in range(n):
        s = acc[-1] * i / (n - 1)
        while j < len(acc) - 2 and acc[j + 1] < s:
            j += 1
        w = (s - acc[j]) / max(acc[j + 1] - acc[j], 1e-12)
        out.append(pts[j] + w * (pts[j + 1] - pts[j]))
    return out


def path_d(points, closed=False):
    """Absolute move followed by relative line segments: compact, and interpolable by SMIL."""
    xy = [to_px(p) for p in points]
    x0, y0 = round(xy[0][0], 1), round(xy[0][1], 1)
    parts, px, py = [f"M{x0:g} {y0:g}l"], x0, y0
    for x, y in xy[1:]:
        x, y = round(x, 1), round(y, 1)
        parts.append(f"{x - px:.1f} {y - py:.1f}".replace(".0 ", " ").removesuffix(".0"))
        px, py = x, y
    return " ".join(parts) + ("z" if closed else "")


def compute():
    lines = [[] for _ in OFFSETS]
    speeds = [0.0] * len(OFFSETS)
    bodies = []
    for t, mu in FRAMES:
        flow = Flow(mu)
        body_psi = flow.psi(mu + flow.r)
        circle = [mu + flow.r * cmath.exp(2j * math.pi * k / N_BODY) for k in range(N_BODY)]
        bodies.append(path_d([to_z(c, t) for c in circle], closed=True))
        for i, off in enumerate(OFFSETS):
            pts, sp = trace(flow, seed(flow, body_psi + off), t)
            lines[i].append(path_d(resample(pts, N_LINE)))
            speeds[i] += sum(sp) / len(sp) / len(FRAMES)
    return lines, speeds, bodies


# --------------------------------------------------------------------------- fonts

def font_face(key, chars):
    pkg, member, family, _ = FONTS[key]
    folder = HERE / ".fontcache" / pkg.replace("/", "_").replace("@", "_")
    path = folder / member
    if not path.exists():
        folder.parent.mkdir(exist_ok=True)
        out = subprocess.run(["npm", "pack", pkg, "--pack-destination", str(folder.parent)],
                             check=True, capture_output=True, text=True).stdout.split()[-1]
        with tarfile.open(folder.parent / out) as tar:
            tar.extractall(folder, filter="data")
    tt = TTFont(path)
    opts = subset.Options()
    opts.flavor, opts.layout_features, opts.name_IDs = "woff2", ["kern", "liga"], []
    sub = subset.Subsetter(opts)
    sub.populate(text="".join(sorted(set(chars))))
    sub.subset(tt)
    buf = io.BytesIO()
    tt.save(buf)
    data = base64.b64encode(buf.getvalue()).decode()
    return f'@font-face{{font-family:{family};src:url(data:font/woff2;base64,{data}) format("woff2")}}'


def family(key):
    return f"{FONTS[key][2]}, {FONTS[key][3]}"


# --------------------------------------------------------------------------- svg

def morph(frames):
    vals = ";".join(frames + frames[:1])
    n = len(frames)
    times = ";".join(f"{k / n:.4f}" for k in range(n + 1))
    splines = ";".join(["0.8 0 0.2 1"] * n)  # ease in/out so each shape lingers
    return (f'<animate attributeName="d" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="spline" keyTimes="{times}" keySplines="{splines}" values="{vals}"/>')


NAME = "Diwen (Steven) Xu"
TAGLINE = "Physics-Based Simulation · Graphics & Robotics · AI"
CAPTION = "potential flow past a morphing Joukowski airfoil"


def render(theme, lines, speeds, bodies):
    c = THEMES[theme]
    faces = font_face("sans-b", NAME) + font_face("sans", TAGLINE) + font_face("mono", CAPTION)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="{NAME}: {TAGLINE.replace("&", "&amp;")}. Potential flow past a shape '
           f'morphing from a cylinder into an airfoil.">',
           "<defs>",
           f'<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="560" y1="0" x2="{W}" y2="0">'
           f'<stop offset="0" stop-color="{c["a"]}"/><stop offset="1" stop-color="{c["b"]}"/></linearGradient>',
           # keep the left half clean for the text: the flow fades in from x = 500 to 760
           f'<linearGradient id="fade" gradientUnits="userSpaceOnUse" x1="500" y1="0" x2="760" y2="0">'
           '<stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff"/></linearGradient>',
           f'<mask id="m"><rect width="{W}" height="{H}" fill="url(#fade)"/></mask>',
           f"<style>{faces}",
           f'.sl{{fill:none;stroke:{c["muted"]};stroke-opacity:{c["line_op"]};stroke-width:1.2}}',
           ".tr{fill:none;stroke:url(#g);stroke-width:2.8;stroke-linecap:round;stroke-dasharray:0 30;"
           "animation:flow linear infinite}",
           "@keyframes flow{to{stroke-dashoffset:-30}}",
           "@media (prefers-reduced-motion:reduce){.tr{animation:none}}",
           "</style>"]
    # each streamline is defined once and drawn twice: a faint line plus moving tracer dots
    out += [f'<path id="s{i}" pathLength="1000" d="{frames[0]}">{morph(frames)}</path>'
            for i, frames in enumerate(lines)]
    out += ["</defs>", '<g mask="url(#m)">']
    for i, speed in enumerate(speeds):
        dur = 1.4 / speed  # faster tracers where the flow is faster (e.g. over the suction side)
        out.append(f'<use href="#s{i}" class="sl"/>'
                   f'<use href="#s{i}" class="tr" style="animation-duration:{dur:.2f}s"/>')
    out.append("</g>")
    out.append(f'<path d="{bodies[0]}" fill="url(#g)" fill-opacity="{c["body_op"]}" stroke="url(#g)" '
               f'stroke-width="2.4" stroke-linejoin="round">{morph(bodies)}</path>')
    out += [f'<text x="80" y="262" font-family="{family("sans-b")}" font-size="48" letter-spacing="-1" '
            f'fill="{c["text"]}">{NAME}</text>',
            f'<text x="82" y="304" font-family="{family("sans")}" font-size="19" '
            f'fill="{c["muted"]}">{TAGLINE.replace("&", "&amp;")}</text>',
            f'<text x="82" y="{H - 24}" font-family="{family("mono")}" font-size="13" '
            f'fill="{c["muted"]}" fill-opacity="0.8">{CAPTION}</text>',
            "</svg>"]
    return "\n".join(out) + "\n"


def main():
    import logging
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    lines, speeds, bodies = compute()
    for theme in THEMES:
        out = HERE / f"banner-{theme}.svg"
        out.write_text(render(theme, lines, speeds, bodies), encoding="utf-8")
        print(f"{out.name}: {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
