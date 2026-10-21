"""Generate the Kármán vortex street banner: dye streaks behind a cylinder at Re = 150.

The flow is a real viscous simulation, not a sketch: a D2Q9 lattice Boltzmann
solver (BGK collision, bounce-back cylinder, Zou-He inflow) is run until the
vortex shedding is periodic, one shedding period of the velocity field is
stored, and dye particles are traced through it with RK2. Because the flow is
periodic, one period of the dye animation loops seamlessly.

Usage:  python assets/karman.py   (writes karman-{dark,light}.svg; needs numpy)
The flow field is cached in assets/.cache, so layout tweaks rerun in seconds.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np

from banner import NAME, TAGLINE, THEMES, H, W, family, font_face

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache" / "karman.npz"

NX, NY = 380, 225                     # lattice size (cells)
CX, CY, R = 90, 112, 10               # cylinder centre and radius (cells)
U0, RE = 0.08, 150                    # inflow speed (lattice units), Reynolds number
PX = 2.4                              # pixels per cell
ORIGIN = (650, 270)                   # canvas position of the cylinder centre

V = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
WT = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float32)[:, None, None]
VX = V[:, 0].astype(np.float32)[:, None, None]
VY = V[:, 1].astype(np.float32)[:, None, None]
OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]
LEFT, MID, RIGHT = [3, 6, 7], [0, 2, 4], [1, 5, 8]


# --------------------------------------------------------------------------- lattice Boltzmann

def equilibrium(rho, ux, uy):
    cu = 3 * (VX * ux + VY * uy)
    return rho * WT * (1 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy))


def simulate(develop=20000, stride=10):
    """Run to periodic shedding; return one period of (ux, uy) sampled every `stride` steps."""
    omega = 1 / (3 * U0 * 2 * R / RE + 0.5)
    y, x = np.mgrid[0:NY, 0:NX]
    solid = (x - CX) ** 2 + (y - CY) ** 2 < R ** 2
    u_in = np.full(NY, U0, dtype=np.float32)
    kick = 0.2 * U0 * np.exp(-((x - CX - 3 * R) ** 2 + (y - CY) ** 2) / (2 * R * R))  # breaks symmetry
    f = equilibrium(np.float32(1), u_in[:, None] * np.ones((NY, NX), np.float32), kick.astype(np.float32))

    def step(f):
        f[LEFT, :, -1] = f[LEFT, :, -2]                               # outflow: zero gradient
        rho = f.sum(0)
        ux = (f * VX).sum(0) / rho
        uy = (f * VY).sum(0) / rho
        ux[:, 0], uy[:, 0] = u_in, 0                                   # Zou-He velocity inlet
        rho[:, 0] = (f[MID, :, 0].sum(0) + 2 * f[LEFT, :, 0].sum(0)) / (1 - ux[:, 0])
        feq = equilibrium(rho, ux, uy)
        f[RIGHT, :, 0] = feq[RIGHT, :, 0] + f[[3, 7, 6], :, 0] - feq[[3, 7, 6], :, 0]
        out = f - omega * (f - feq)                                    # BGK collision
        out[:, solid] = f[OPP][:, solid]                               # bounce-back
        for i in range(9):                                             # streaming
            f[i] = np.roll(out[i], (V[i, 1], V[i, 0]), axis=(0, 1))
        return ux, uy

    probe = (CY, CX + 8 * R)
    t0 = time.time()
    for n in range(develop):
        step(f)
        if n % 5000 == 0:
            print(f"  develop {n}/{develop}  ({time.time() - t0:.0f}s)", file=sys.stderr)

    # measure the shedding period from upward zero crossings of the wake's transverse velocity
    sig = []
    for _ in range(6000):
        sig.append(step(f)[1][probe])
    sig = np.array(sig) - np.mean(sig)
    ups = np.nonzero((sig[:-1] < 0) & (sig[1:] >= 0))[0]
    period = float(np.mean(np.diff(ups)))
    n_snap = round(period / stride)

    # start at an upward crossing, then record exactly one period
    prev = step(f)[1][probe]
    while True:
        cur = step(f)[1][probe]
        if prev < 0 <= cur:
            break
        prev = cur
    ux_s, uy_s = [], []
    for k in range(n_snap * stride + 1):
        ux, uy = step(f)
        if k % stride == 0:
            ux_s.append(ux.astype(np.float32))
            uy_s.append(uy.astype(np.float32))
    seam = float(np.abs(uy_s[-1] - uy_s[0]).max() / U0)
    print(f"  period {period:.1f} steps, {n_snap} snapshots, seam {seam:.2%} of U", file=sys.stderr)
    return np.array(ux_s[:-1]), np.array(uy_s[:-1]), stride, seam


def flow_field():
    if CACHE.exists():
        d = np.load(CACHE)
        return d["ux"], d["uy"], int(d["stride"]), float(d["seam"])
    ux, uy, stride, seam = simulate()
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, ux=ux, uy=uy, stride=stride, seam=seam)
    return ux, uy, stride, seam


# --------------------------------------------------------------------------- dye tracing

M = 24        # animation frames per shedding period; dye is released once per frame
LIFE = 70     # frames a dye particle is followed (long enough to leave the canvas)
SUB = 6       # RK2 substeps per frame
SHOWN = 2.4   # seconds per shedding period on screen


def trace(ux, uy, stride, seeds):
    """Positions at frames 0..M of particles released from each seed at frames -LIFE..M.

    Returns an array (M + 1, seeds, releases, 2); a particle sits on its seed until released.
    """
    n = len(ux)
    delta = n * stride / M
    rel = np.arange(-LIFE, M + 1)

    def vel(t, p):
        s = (t / stride) % n
        i0 = int(s)
        a, i1 = s - i0, (i0 + 1) % n
        x = np.clip(p[..., 0], 0, NX - 1.001)
        y = np.clip(p[..., 1], 0, NY - 1.001)
        x0, y0 = x.astype(int), y.astype(int)
        fx, fy = x - x0, y - y0

        def bilinear(g):
            return ((g[y0, x0] * (1 - fx) + g[y0, x0 + 1] * fx) * (1 - fy)
                    + (g[y0 + 1, x0] * (1 - fx) + g[y0 + 1, x0 + 1] * fx) * fy)

        return np.stack([bilinear(ux[i0]) * (1 - a) + bilinear(ux[i1]) * a,
                         bilinear(uy[i0]) * (1 - a) + bilinear(uy[i1]) * a], -1)

    pos = np.repeat(np.asarray(seeds, float)[:, None, :], len(rel), 1)
    out = np.empty((M + 1, len(seeds), len(rel), 2))
    h = delta / SUB
    for j in range(-LIFE, M + 1):
        if j >= 0:
            out[j] = pos
        if j == M:
            break
        live = rel <= j
        p = pos[:, live]
        for k in range(SUB):
            t = (j + k / SUB) * delta
            mid = p + 0.5 * h * vel(t, p)
            p = p + h * vel(t + 0.5 * h, mid)
        pos[:, live] = p
    return out


def to_px(p):
    return np.stack([ORIGIN[0] + (p[..., 0] - CX) * PX, ORIGIN[1] + (p[..., 1] - CY) * PX], -1)


def rel_path(xy):
    """Absolute move then relative segments: compact and interpolable by SMIL."""
    x0, y0 = round(float(xy[0][0]), 1), round(float(xy[0][1]), 1)
    out, px, py = [f"M{x0:g} {y0:g}l"], x0, y0
    for x, y in xy[1:]:
        x, y = round(float(x), 1), round(float(y), 1)
        out.append(f"{x - px:.1f} {y - py:.1f}".replace(".0 ", " ").removesuffix(".0"))
        px, py = x, y
    return " ".join(out)


def animate_d(frames):
    return (f'<animate attributeName="d" dur="{SHOWN}s" repeatCount="indefinite" '
            f'values="{";".join(frames)}"/>')


# --------------------------------------------------------------------------- svg

DYE_SEEDS = [(CX + (R + d) * math.cos(math.radians(a)), CY - s * (R + d) * math.sin(math.radians(a)))
             for s in (1, -1) for a, d in ((96, 1.0), (92, 2.5), (88, 4.5))]
RAKE_SEEDS = [(CX - 4 * R, CY + s * k * R) for s in (1, -1) for k in (1.5, 2.5, 4.0, 6.0)]


def compute():
    ux, uy, stride, seam = flow_field()
    y, x = np.mgrid[0:NY, 0:NX]
    solid = (x - CX) ** 2 + (y - CY) ** 2 < R ** 2
    ux[:, solid] = 0
    uy[:, solid] = 0
    pix = to_px(trace(ux, uy, stride, DYE_SEEDS + RAKE_SEEDS))
    n_dye = len(DYE_SEEDS)
    # dye: vertex k is the particle released at frame M - k, so vertices move with the fluid and
    # the shift by M particles at the loop point lands on an identical picture
    dye = [[rel_path(pix[f, s, ::-1]) for f in range(M + 1)] for s in range(n_dye)]
    # rake lines carry no dots, so they are drawn as loci of constant age (every other release)
    rake = [[rel_path(pix[f, s, f + LIFE::-2][:LIFE // 2]) for f in range(M + 1)]
            for s in range(n_dye, pix.shape[1])]
    return dye, rake, seam


def render(theme, dye, rake):
    c = THEMES[theme]
    faces = font_face("sans-b", NAME) + font_face("sans", TAGLINE)
    cyl = (ORIGIN[0], ORIGIN[1], R * PX)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="{NAME}: {TAGLINE.replace("&", "&amp;")}. Kármán vortex street behind '
           f'a cylinder, simulated with the lattice Boltzmann method.">',
           "<defs>",
           f'<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="600" y1="0" x2="{W}" y2="0">'
           f'<stop offset="0" stop-color="{c["a"]}"/><stop offset="1" stop-color="{c["b"]}"/></linearGradient>',
           '<linearGradient id="fade" gradientUnits="userSpaceOnUse" x1="470" y1="0" x2="620" y2="0">'
           '<stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#fff"/></linearGradient>',
           f'<mask id="m"><rect width="{W}" height="{H}" fill="url(#fade)"/></mask>',
           *[f'<marker id="d{k}" viewBox="0 0 4 4" refX="2" refY="2" markerWidth="4" markerHeight="4" '
             f'markerUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.6" fill="{col}"/></marker>'
             for k, col in enumerate((c["a"], c["b"]))],
           f"<style>{faces}</style>",
           "</defs>",
           '<g mask="url(#m)" fill="none">']
    for frames in rake:
        out.append(f'<path d="{frames[0]}" stroke="{c["muted"]}" stroke-opacity="{c["line_op"]}" '
                   f'stroke-width="1.2">{animate_d(frames)}</path>')
    for i, frames in enumerate(dye):
        k = 0 if i < len(dye) // 2 else 1
        col = (c["a"], c["b"])[k]
        out.append(f'<path d="{frames[0]}" stroke="{col}" stroke-opacity="0.35" stroke-width="1" '
                   f'marker-start="url(#d{k})" marker-mid="url(#d{k})" marker-end="url(#d{k})">'
                   f'{animate_d(frames)}</path>')
    out.append("</g>")
    out.append(f'<circle cx="{cyl[0]}" cy="{cyl[1]}" r="{cyl[2]:g}" fill="url(#g)" fill-opacity="{c["body_op"]}" '
               f'stroke="url(#g)" stroke-width="2.4"/>')
    out += [f'<text x="80" y="262" font-family="{family("sans-b")}" font-size="48" letter-spacing="-1" '
            f'fill="{c["text"]}">{NAME}</text>',
            f'<text x="82" y="304" font-family="{family("sans")}" font-size="19" '
            f'fill="{c["muted"]}">{TAGLINE.replace("&", "&amp;")}</text>',
            "</svg>"]
    return "\n".join(out) + "\n"


def main():
    import logging
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    dye, rake, seam = compute()
    print(f"shedding loop seam: {seam:.2%} of the inflow speed")
    for theme in THEMES:
        out = HERE / f"karman-{theme}.svg"
        out.write_text(render(theme, dye, rake), encoding="utf-8")
        print(f"{out.name}: {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
