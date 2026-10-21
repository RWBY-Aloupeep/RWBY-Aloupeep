"""Generate the profile banner: potential flow past a morphing Joukowski body.

Every curve in the banner is computed, not drawn. In the zeta-plane we solve
uniform flow past a cylinder with circulation (Kutta condition at zeta = 1);
the conformal map z = zeta + t * lam^2 / zeta then carries the cylinder and
its streamlines into the physical plane. Sweeping t and the cylinder centre
morphs a circle into a symmetric and then a cambered airfoil, and because the
stream function is invariant under conformal maps, every frame is an exact
solution of Laplace's equation.

Usage:  python assets/banner.py   (stdlib only; writes banner-{dark,light}.svg)
"""

import cmath
import math
from pathlib import Path

W, H = 1200, 320                 # SVG canvas (px)
CX, CY, SCALE = 860, 168, 60     # pixel position of the z-plane origin, px per unit
U, ALPHA, LAM = 1.0, math.radians(7), 1.0

# (t, centre of the zeta-plane circle): circle -> symmetric airfoil -> cambered airfoil
FRAMES = [(0.0, complex(-0.10, 0.00)),
          (1.0, complex(-0.10, 0.00)),
          (1.0, complex(-0.08, 0.16))]

# stream-function offsets from the body streamline; one entry per streamline
OFFSETS = [s * d for d in (0.2, 0.45, 0.75, 1.1, 1.5, 1.95, 2.45, 3.0) for s in (1, -1)]
N_LINE, N_BODY = 110, 180
X_SEED, X_END = -16.0, 7.0
CYCLE = 15                       # seconds for one full morph cycle


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
    """RK4 on the unit direction field; stops once the mapped point leaves the canvas."""
    def f(p):
        v = flow.velocity(p)
        return v / max(abs(v), 1e-9)

    pts, speeds, p = [start], [abs(flow.velocity(start))], start
    for _ in range(20000):
        k1 = f(p)
        k2 = f(p + 0.5 * ds * k1)
        k3 = f(p + 0.5 * ds * k2)
        k4 = f(p + ds * k3)
        p += ds / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts.append(p)
        speeds.append(abs(flow.velocity(p)))
        if to_z(p, t).real > X_END:
            break
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
    x0, y0 = xy[0]
    parts = [f"M{x0:.1f} {y0:.1f}l"]
    px, py = round(x0, 1), round(y0, 1)
    for x, y in xy[1:]:
        x, y = round(x, 1), round(y, 1)
        parts.append(f"{x - px:.1f} {y - py:.1f}")
        px, py = x, y
    return " ".join(parts).replace(".0 ", " ").replace("-0 ", "0 ") + ("z" if closed else "")


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
            lines[i].append(path_d(resample([to_z(p, t) for p in pts], N_LINE)))
            speeds[i] += sum(sp) / len(sp) / len(FRAMES)
    return lines, speeds, bodies


THEMES = {
    "dark": dict(text="#E6EDF3", muted="#8B949E", line="#8B949E", line_op=0.22,
                 a="#7B5CFF", b="#22D3EE", body_op=0.18),
    "light": dict(text="#1F2328", muted="#59636E", line="#59636E", line_op=0.2,
                  a="#6E4FF6", b="#0891B2", body_op=0.12),
}

MONO = "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"


def morph(frames, attr="d"):
    vals = ";".join(frames + frames[:1])
    n = len(frames)
    times = ";".join(f"{k / n:.4f}" for k in range(n + 1))
    splines = ";".join(["0.8 0 0.2 1"] * n)  # ease in/out so each shape lingers
    return (f'<animate attributeName="{attr}" dur="{CYCLE}s" repeatCount="indefinite" '
            f'calcMode="spline" keyTimes="{times}" keySplines="{splines}" values="{vals}"/>')


def render(theme, lines, speeds, bodies):
    c = THEMES[theme]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="Diwen (Steven) Xu: potential flow past a shape morphing from a cylinder into an airfoil">',
           "<defs>",
           f'<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="560" y1="0" x2="{W}" y2="0">'
           f'<stop offset="0" stop-color="{c["a"]}"/><stop offset="1" stop-color="{c["b"]}"/></linearGradient>',
           # fade the flow out behind the text block
           '<linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">'
           '<stop offset="0" stop-color="#fff" stop-opacity="0.12"/><stop offset="0.42" stop-color="#fff" stop-opacity="0.18"/>'
           '<stop offset="0.6" stop-color="#fff" stop-opacity="1"/></linearGradient>',
           f'<mask id="m"><rect width="{W}" height="{H}" fill="url(#fade)"/></mask>',
           "<style>",
           f'.sl{{fill:none;stroke:{c["line"]};stroke-opacity:{c["line_op"]};stroke-width:1.2}}',
           ".tr{fill:none;stroke:url(#g);stroke-width:2.6;stroke-linecap:round;stroke-dasharray:0 26;animation:flow linear infinite}",
           "@keyframes flow{to{stroke-dashoffset:-26}}",
           "@media (prefers-reduced-motion:reduce){.tr{animation:none}}",
           "</style>"]
    # each streamline is defined once and drawn twice: a faint line plus moving tracer dots
    out += [f'<path id="s{i}" pathLength="1300" d="{frames[0]}">{morph(frames)}</path>'
            for i, frames in enumerate(lines)]
    out += ["</defs>", '<g mask="url(#m)">']
    for i, speed in enumerate(speeds):
        dur = 1.6 / speed  # faster tracers where the flow is faster (e.g. over the suction side)
        out.append(f'<use href="#s{i}" class="sl"/>'
                   f'<use href="#s{i}" class="tr" style="animation-duration:{dur:.2f}s"/>')
    out.append("</g>")

    out.append(f'<path d="{bodies[0]}" fill="url(#g)" fill-opacity="{c["body_op"]}" stroke="url(#g)" '
               f'stroke-width="2.2" stroke-linejoin="round">{morph(bodies)}</path>')

    out += [f'<text x="64" y="128" font-family="{MONO}" font-size="18" fill="{c["muted"]}">Hi, I\'m</text>',
            f'<text x="60" y="190" font-family="{SANS}" font-size="54" font-weight="700" fill="{c["text"]}" '
            f'letter-spacing="-0.5">Diwen (Steven) Xu</text>',
            f'<text x="{W - 28}" y="{H - 22}" text-anchor="end" font-family="{MONO}" font-size="13" '
            f'fill="{c["muted"]}" fill-opacity="0.8">∇²φ = 0 · z = ζ + λ²/ζ</text>',
            "</svg>"]
    return "\n".join(out) + "\n"


def main():
    lines, speeds, bodies = compute()
    here = Path(__file__).parent
    for theme in THEMES:
        (here / f"banner-{theme}.svg").write_text(render(theme, lines, speeds, bodies), encoding="utf-8")


if __name__ == "__main__":
    main()
