"""Generate the profile banner: a Kármán vortex street behind a cylinder at Re = 200.

The flow is a real viscous simulation, not a sketch: a D2Q9 lattice Boltzmann
solver (BGK collision, bounce-back cylinder, Zou-He inflow) is run until the
vortex shedding is periodic, and one shedding period of the velocity field is
stored. Dye particles are traced through it with RK2, and its vorticity is
drawn as a soft haze behind them. Because the flow is periodic, one period of
the animation loops seamlessly.

Usage:  python assets/banner.py   (writes banner-{dark,light}.svg)
Needs numpy, pillow, fonttools and brotli. The flow field is cached in assets/.cache
(the first run takes a few minutes), so layout tweaks rerun in seconds. The
OFL font Inter is fetched once with `npm pack` into assets/.fontcache and
subset into the SVG, so the text renders the same everywhere.
"""

import base64
import gzip
import io
import logging
import math
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from fontTools import subset
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent

W, H = 1200, 540                      # SVG canvas (px)
NAME = "Diwen (Steven) Xu"
TAGLINE = "Physics-Based Simulation · Graphics & Robotics · AI"

THEMES = {
    "dark": dict(text="#E6EDF3", muted="#8B949E", a="#7B5CFF", b="#22D3EE", line_op=0.2, body_op=0.18,
                 dye_op=0.85, haze=0.5),
    "light": dict(text="#1F2328", muted="#59636E", a="#6E4FF6", b="#0891B2", line_op=0.2, body_op=0.12,
                  dye_op=0.85, haze=0.35),
}

NX, NY = 430, 270                     # lattice size (cells)
CX, CY, R = 80, 134, 16               # cylinder centre and radius (cells)
U0, RE = 0.08, 200                    # inflow speed (lattice units), Reynolds number
PX = 2.0                              # pixels per cell
ORIGIN = (620, 270)                   # canvas position of the cylinder centre
CACHE = HERE / ".cache" / f"karman-{NX}x{NY}-r{R}-re{RE}.npz"

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


def simulate(develop=36000, stride=16):
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
    for _ in range(8000):
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

M = 16        # animation frames per shedding period; dye is released once per frame
LIFE = 60     # frames a dye particle is followed (long enough that ~95% have left the canvas)
SUB = 8       # RK2 substeps per frame
SHOWN = 3.0   # seconds per shedding period on screen


RELEASES = np.arange(-LIFE, M + 1)   # release frames of the particles on every path


def trace(ux, uy, stride, starts):
    """Positions at frames 0..M of particles released at frames -LIFE..M from `starts`.

    `starts` has shape (paths, releases, 2). Returns (M + 1, paths, releases, 2); a particle
    sits on its start point until released.
    """
    n = len(ux)
    delta = n * stride / M
    rel = RELEASES

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

    pos = np.array(starts, float)
    out = np.empty((M + 1, len(pos), len(rel), 2))
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
    xy = np.stack([ORIGIN[0] + (p[..., 0] - CX) * PX, ORIGIN[1] + (p[..., 1] - CY) * PX], -1)
    # park everything that has left the canvas on its edge: long runs of "0 0" gzip to nothing
    return np.stack([np.clip(xy[..., 0], 0, W + 8), np.clip(xy[..., 1], -8, H + 8)], -1)


def rel_path(xy):
    """Absolute move then relative whole-pixel segments: compact and interpolable by SMIL."""
    q = np.round(np.asarray(xy)).astype(int)
    return f"M{q[0, 0]} {q[0, 1]}l" + " ".join(f"{dx} {dy}" for dx, dy in np.diff(q, axis=0))


def animate(attr, values, **extra):
    more = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in extra.items())
    return (f'<animate attributeName="{attr}" dur="{SHOWN}s" repeatCount="indefinite"{more} '
            f'values="{";".join(values)}"/>')


# --------------------------------------------------------------------------- vorticity haze

HAZE_BLOCK = 5    # lattice cells per haze texel; the browser's smooth upscaling does the rest
HAZE_LEFT = 470   # canvas x where the haze (and the flow) starts
HAZE_AQ = 60     # WebP alpha quality: lossless alpha keeps the empty background truly empty


def vorticity(ux, uy):
    """Vorticity at the M frame times, block-averaged over the visible part of the lattice."""
    n = len(ux)
    x0 = int(CX + (HAZE_LEFT - ORIGIN[0]) / PX)
    x1 = int(CX + (W - ORIGIN[0]) / PX) + HAZE_BLOCK
    half = 6 * R - (6 * R) % HAZE_BLOCK          # the wake never strays further than this
    y0, y1 = CY - half, CY + half
    x1 -= (x1 - x0) % HAZE_BLOCK
    y, x = np.mgrid[0:NY, 0:NX]
    # fade out the wall's boundary layer, which would otherwise swamp the scale and glow blockily,
    # and everything upstream of the shoulders: only the wake is drawn
    ring = np.clip((np.hypot(x - CX, y - CY) - R - 0.5) / (0.7 * R), 0, 1)
    taper = ring * ring * (3 - 2 * ring) * np.clip((x - CX + R) / (1.2 * R), 0, 1)
    frames = []
    for f in range(M):
        s = f * n / M
        i0 = int(s)
        a, i1 = s - i0, (i0 + 1) % n
        u = ux[i0] * (1 - a) + ux[i1] * a
        v = uy[i0] * (1 - a) + uy[i1] * a
        om = (np.gradient(v, axis=1) - np.gradient(u, axis=0)) * taper
        for axis in (0, 1):  # binomial blur, so the upscaled texels blend into soft blobs
            om = sum(w * np.roll(om, k, axis) for k, w in zip(range(-2, 3), (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)))
        om = om[y0:y1, x0:x1]
        frames.append(om.reshape((y1 - y0) // HAZE_BLOCK, HAZE_BLOCK, -1, HAZE_BLOCK).mean((1, 3)))
    box = (ORIGIN[0] + (x0 - CX - 0.5) * PX, ORIGIN[1] + (y0 - CY - 0.5) * PX, (x1 - x0) * PX, (y1 - y0) * PX)
    return np.array(frames), box


def haze_images(frames, c, alpha):
    """WebP frames: hue by sign of vorticity (top vortices in colour a), opacity by magnitude."""
    ref = np.percentile(np.abs(frames), 99.0)
    rgb_a = np.array([int(c["a"][i:i + 2], 16) for i in (1, 3, 5)])
    rgb_b = np.array([int(c["b"][i:i + 2], 16) for i in (1, 3, 5)])
    out = []
    for om in frames:
        # smoothstep from 8% to 60% of the reference: the lattice's faint acoustic noise drops out
        t = np.clip((np.abs(om) / ref - 0.08) / 0.52, 0, 1)
        mag = t * t * (3 - 2 * t)
        # hue from a widely blurred sign, so even transparent texels carry a smooth colour:
        # WebP subsamples chroma, and a speckled sign there would bleed cyan into purple
        wide = om
        for _ in range(4):
            wide = sum(np.roll(np.roll(wide, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) / 9
        hue = (0.5 + 0.5 * np.tanh(wide / (0.05 * ref)))[..., None]
        rgba = np.concatenate([rgb_b + hue * (rgb_a - rgb_b), 255 * alpha * mag[..., None]], -1)
        buf = io.BytesIO()
        Image.fromarray(rgba.astype(np.uint8), "RGBA").save(buf, "WEBP", quality=50, alpha_quality=HAZE_AQ, method=6)
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


# --------------------------------------------------------------------------- fonts

FONTS = {  # key: (npm package, file inside it, css family)
    "sans": ("@fontsource/inter@5.3.0", "package/files/inter-latin-400-normal.woff2", "pf-sans"),
    "sans-b": ("@fontsource/inter@5.3.0", "package/files/inter-latin-600-normal.woff2", "pf-sans-b"),
}


def font_face(key, chars):
    pkg, member, fam = FONTS[key]
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
    return f'@font-face{{font-family:{fam};src:url(data:font/woff2;base64,{data}) format("woff2")}}'


def family(key):
    return f"{FONTS[key][2]}, 'Segoe UI', Helvetica, Arial, sans-serif"


# --------------------------------------------------------------------------- svg

# dye seeps out of the shoulders, ahead of separation: 7 paths per side, each spread over a band
# of angles and wall distances (jittered per release phase, so frame M still matches frame 0)
DYE_SITES = [(s, a, d) for s in (1, -1) for a, d in zip(np.linspace(100, 86, 7), np.linspace(0.6, 0.3 * R, 7))]


def dye_starts():
    rng = np.random.default_rng(7)
    jitter_a = rng.uniform(-3.5, 3.5, (len(DYE_SITES), M))
    jitter_d = rng.uniform(0.7, 1.3, (len(DYE_SITES), M))
    phase = RELEASES % M
    a = np.radians([[site[1]] for site in DYE_SITES] + jitter_a[:, phase])
    r = R + np.array([[site[2]] for site in DYE_SITES]) * jitter_d[:, phase]
    side = np.array([[site[0]] for site in DYE_SITES])
    return np.stack([CX + r * np.cos(a), CY - side * r * np.sin(a)], -1)

RAKE_SEEDS = [(CX - 4 * R, CY + s * k * R) for s in (1, -1) for k in (5.0, 6.5, 8.0)]


def compute():
    ux, uy, stride, seam = flow_field()
    y, x = np.mgrid[0:NY, 0:NX]
    solid = (x - CX) ** 2 + (y - CY) ** 2 < R ** 2
    ux[:, solid] = 0
    uy[:, solid] = 0
    dye0 = dye_starts()
    rake0 = np.repeat(np.array(RAKE_SEEDS, float)[:, None], len(RELEASES), 1)
    pos = trace(ux, uy, stride, np.concatenate([dye0, rake0]))
    n_dye = len(dye0)
    # dye not yet released waits just inside the wall, where the cylinder masks it out
    centre = np.array([CX, CY])
    park = centre + (R - 2) * (dye0 - centre) / np.linalg.norm(dye0 - centre, axis=-1, keepdims=True)
    for f in range(M + 1):
        pos[f, :n_dye, LIFE + f + 1:] = park[:, LIFE + f + 1:]
    pix = to_px(pos)
    # dye: vertex k is always the particle released at frame M - k, so vertices move with the
    # fluid; frame M equals frame 0 shifted by M particles, so the jump back is invisible
    dye = [[rel_path(pix[f, s, ::-1]) for f in range(M + 1)] for s in range(n_dye)]
    # streak lines carry no dots, so they are drawn as loci of constant age; frame M == frame 0
    rake = [[rel_path(pix[f, s, f + LIFE::-1][:LIFE]) for f in range(M)] for s in range(n_dye, pix.shape[1])]
    om, box = vorticity(ux, uy)
    return dye, rake, om, box, seam


def haze_svg(images, box):
    """Cyclic triangle cross-fade: frame i peaks at i/M and is gone by its neighbours' peaks."""
    bx, by, bw, bh = box
    out = []
    for i, img in enumerate(images):
        times = [(i - 1) / M, i / M, (i + 1) / M]
        if i == 0:
            kt, vals = [0, 1 / M, 1 - 1 / M, 1], ["1", "0", "0", "1"]
        elif i == M - 1:
            kt, vals = [0, times[0], times[1], 1], ["0", "0", "1", "0"]
        else:
            kt, vals = [0, *times, 1], ["0", "0", "1", "0", "0"]
        out.append(f'<image href="data:image/webp;base64,{img}" x="{bx:g}" y="{by:g}" width="{bw:g}" '
                   f'height="{bh:g}" preserveAspectRatio="none" opacity="0">'
                   f'{animate("opacity", vals, keyTimes=";".join(f"{t:.4f}" for t in kt))}</image>')
    return out


def render(theme, dye, rake, om, box):
    c = THEMES[theme]
    faces = font_face("sans-b", NAME) + font_face("sans", TAGLINE)
    marker = ('<marker id="d{k}" viewBox="-2 -2 4 4" markerWidth="4" markerHeight="4" '
              'markerUnits="userSpaceOnUse"><circle r="1.4" fill="{col}"/></marker>')
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="{NAME}: {TAGLINE.replace("&", "&amp;")}. Kármán vortex street behind '
           f'a cylinder, simulated with the lattice Boltzmann method.">',
           "<defs>",
           f'<linearGradient id="g" gradientUnits="userSpaceOnUse" x1="560" y1="0" x2="{W}" y2="0">'
           f'<stop offset="0" stop-color="{c["a"]}"/><stop offset="1" stop-color="{c["b"]}"/></linearGradient>',
           # fade in from behind the text, then let the dye thin out downstream like real smoke
           f'<linearGradient id="fade" gradientUnits="userSpaceOnUse" x1="{HAZE_LEFT}" y1="0" x2="{W}" y2="0">'
           '<stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="0.17" stop-color="#fff"/>'
           '<stop offset="0.45" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0.4"/>'
           '</linearGradient>',
           f'<mask id="m"><rect width="{W}" height="{H}" fill="url(#fade)"/></mask>',
           # the dye condenses out of the boundary layer: clear at the wall, full strength 0.7 R out
           f'<radialGradient id="wall" gradientUnits="userSpaceOnUse" cx="{ORIGIN[0]}" cy="{ORIGIN[1]}" '
           f'r="{1.7 * R * PX:g}"><stop offset="{1 / 1.7:.3f}"/><stop offset="1" stop-opacity="0"/></radialGradient>',
           f'<mask id="md"><rect width="{W}" height="{H}" fill="url(#fade)"/>'
           f'<rect width="{W}" height="{H}" fill="url(#wall)"/></mask>',
           *[marker.format(k=k, col=col) for k, col in enumerate((c["a"], c["b"]))],
           f"<style>{faces}</style>",
           "</defs>",
           '<g mask="url(#m)" fill="none">',
           *haze_svg(haze_images(om, c, c["haze"]), box)]
    for frames in rake:
        out.append(f'<path d="{frames[0]}" stroke="{c["muted"]}" stroke-opacity="{c["line_op"]}" '
                   f'stroke-width="1.2" stroke-linejoin="round">{animate("d", frames + frames[:1])}</path>')
    out.append('</g><g mask="url(#md)" fill="none">')
    for i, frames in enumerate(dye):
        k = 0 if i < len(dye) // 2 else 1
        out.append(f'<path d="{frames[0]}" stroke="none" opacity="{c["dye_op"]}" '
                   f'marker-start="url(#d{k})" marker-mid="url(#d{k})" marker-end="url(#d{k})">'
                   f'{animate("d", frames)}</path>')
    out.append("</g>")
    out.append(f'<circle cx="{ORIGIN[0]}" cy="{ORIGIN[1]}" r="{R * PX:g}" fill="url(#g)" '
               f'fill-opacity="{c["body_op"]}" stroke="url(#g)" stroke-width="2.4"/>')
    out += [f'<text x="80" y="262" font-family="{family("sans-b")}" font-size="48" letter-spacing="-1" '
            f'fill="{c["text"]}">{NAME}</text>',
            f'<text x="82" y="304" font-family="{family("sans")}" font-size="19" '
            f'fill="{c["muted"]}">{TAGLINE.replace("&", "&amp;")}</text>',
            "</svg>"]
    return "\n".join(out) + "\n"


def main():
    logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
    dye, rake, om, box, seam = compute()
    print(f"shedding loop seam: {seam:.2%} of the inflow speed")
    for theme in THEMES:
        out = HERE / f"banner-{theme}.svg"
        text = render(theme, dye, rake, om, box)
        out.write_text(text, encoding="utf-8")
        print(f"{out.name}: {len(text.encode()) / 1024:.0f} KB, {len(gzip.compress(text.encode(), 9)) / 1024:.0f} KB gzipped")


if __name__ == "__main__":
    main()
