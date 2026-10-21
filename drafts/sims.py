"""Physics behind the profile drafts. Every frame in the SVGs comes from here."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
import banner as flow  # noqa: E402  (potential flow + Joukowski map)


# --------------------------------------------------------------------------- double pendulum

G = 9.81


def _pendulum_rhs(s):
    t1, t2, w1, w2 = s
    d = t1 - t2
    den = 3 - math.cos(2 * d)  # unit masses and lengths
    a1 = (-3 * G * math.sin(t1) - G * math.sin(t1 - 2 * t2)
          - 2 * math.sin(d) * (w2 * w2 + w1 * w1 * math.cos(d))) / den
    a2 = 2 * math.sin(d) * (2 * w1 * w1 + 2 * G * math.cos(t1) + w2 * w2 * math.cos(d)) / den
    return (w1, w2, a1, a2)


def _rk4(s, h):
    k1 = _pendulum_rhs(s)
    k2 = _pendulum_rhs([a + 0.5 * h * b for a, b in zip(s, k1)])
    k3 = _pendulum_rhs([a + 0.5 * h * b for a, b in zip(s, k2)])
    k4 = _pendulum_rhs([a + h * b for a, b in zip(s, k3)])
    return [a + h / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(s, k1, k2, k3, k4)]


def double_pendulums(n=7, theta=(2.3, 2.3), spread=2e-4, duration=12.0, fps=30, substeps=40):
    """n double pendulums released from rest, initial angles `spread` rad apart.

    Returns per pendulum a list of (theta1, theta2) sampled at `fps`, plus the
    relative energy drift of the integrator (a sanity check).
    """
    runs, drift = [], 0.0
    for k in range(n):
        s = [theta[0] + k * spread, theta[1] + k * spread, 0.0, 0.0]
        e0 = _energy(s)
        frames = []
        for _ in range(int(duration * fps) + 1):
            frames.append((s[0], s[1]))
            for _ in range(substeps):
                s = _rk4(s, 1 / (fps * substeps))
        drift = max(drift, abs(_energy(s) - e0) / abs(e0))
        runs.append(frames)
    return runs, drift


def _energy(s):
    t1, t2, w1, w2 = s
    kinetic = 0.5 * (2 * w1 * w1 + w2 * w2 + 2 * w1 * w2 * math.cos(t1 - t2))
    potential = -G * (2 * math.cos(t1) + math.cos(t2))
    return kinetic + potential


# --------------------------------------------------------------------------- curtain (mass-spring cloth)

def curtain(nx=15, ny=11, size=(1.5, 1.1), pin_gap=0.8, period=4.0, fps=20, settle=10,
            dt=1 / 400, iters=10, breeze=(0.8, 2.6, 0.9), drag=0.8, keep=0.99):
    """Position-based mass-spring cloth hung from its two top corners, in a periodic breeze.

    The corners are pinned closer together than the cloth is wide, so it drapes.
    The breeze is periodic with `period` and the cloth is damped, so after
    `settle` periods it follows a periodic orbit; we record exactly one period,
    which then loops seamlessly. Returns frames of shape (frames, ny, nx, 3) and
    stats: node/spring counts, the loop seam (first vs. last recorded state) and
    the worst spring stretch in each recorded frame.
    """
    w, h = size
    xs, ys = np.meshgrid(np.linspace(0, w, nx), np.linspace(0, -h, ny))
    x = np.stack([xs, ys, np.zeros_like(xs)], -1).reshape(-1, 3)
    idx = np.arange(nx * ny).reshape(ny, nx)
    pairs = [(idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :]),        # structural
             (idx[:-1, :-1], idx[1:, 1:]), (idx[:-1, 1:], idx[1:, :-1]),   # shear
             (idx[:, :-2], idx[:, 2:]), (idx[:-2, :], idx[2:, :])]         # bending
    a = np.concatenate([p[0].ravel() for p in pairs])
    b = np.concatenate([p[1].ravel() for p in pairs])
    rest = np.linalg.norm(x[a] - x[b], axis=1)
    pinned = np.array([idx[0, 0], idx[0, -1]])
    anchor = np.array([[w * (1 - pin_gap) / 2, 0, 0], [w * (1 + pin_gap) / 2, 0, 0]])
    x[:, 0] = w / 2 + (x[:, 0] - w / 2) * pin_gap   # start already squeezed, so no snap
    prev = x.copy()
    gravity = np.array([0, -9.81, 0])
    count = np.bincount(np.concatenate([a, b]), minlength=nx * ny)[:, None]

    def normals(p):
        g = p.reshape(ny, nx, 3)
        n = np.cross(np.gradient(g, axis=1), np.gradient(g, axis=0))
        return (n / np.linalg.norm(n, axis=-1, keepdims=True)).reshape(-1, 3)

    steps_per_frame = round(1 / (fps * dt))
    record_from = round(settle * period / dt)
    frames = []
    for step in range(round((settle + 1) * period / dt) + 1):
        if step >= record_from and (step - record_from) % steps_per_frame == 0:
            frames.append(x.reshape(ny, nx, 3).copy())
        phase = 2 * math.pi * step * dt / period
        wind = np.array([0.0, 0.0, breeze[0] + breeze[1] * math.sin(phase)
                         + breeze[2] * math.sin(2 * phase + 0.7)])
        n = normals(x)
        aero = drag * np.sum((wind - (x - prev) / dt) * n, axis=1, keepdims=True) * n
        nxt = x + keep * (x - prev) + (gravity + aero) * dt * dt   # damped Verlet
        prev, x = x, nxt
        for _ in range(iters):                                       # Jacobi distance projection
            d = x[b] - x[a]
            ln = np.linalg.norm(d, axis=1, keepdims=True)
            corr = 0.5 * (ln - rest[:, None]) / ln * d
            delta = np.zeros_like(x)
            np.add.at(delta, a, corr)
            np.add.at(delta, b, -corr)
            x = x + 1.5 * delta / count   # over-relaxed average of the corrections
            x[pinned] = anchor
    frames = np.array(frames)
    flat = frames.reshape(len(frames), -1, 3)
    stretch = np.abs(np.linalg.norm(flat[:, a] - flat[:, b], axis=2) / rest - 1).max(axis=1)
    info = dict(nodes=nx * ny, springs=len(a), dt=dt, fps=fps, period=period,
                seam=float(np.abs(frames[-1] - frames[0]).max()), stretch=stretch[:-1])
    return frames[:-1], info  # the last frame duplicates the first
