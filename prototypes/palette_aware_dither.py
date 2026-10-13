"""Palette-structure-aware error diffusion: prototype and evaluation.

Compares three error-diffusion variants in Oklab:
  1. Vector Floyd-Steinberg (baseline): full error vector propagated.
  2. Global-covariance shaping: e' = M e, M from palette covariance eigenstructure.
  3. Per-cell shaping: one M_i per palette color from its k nearest palette
     neighbors (robust stand-in for Delaunay neighbors, which degenerate
     for rank-deficient palettes).

Test design: a deliberately rank-deficient palette lying on the b=const plane
in Oklab, and a test image containing a smooth off-plane (b) component that
no palette color can represent. The hypothesis: the baseline accumulates the
unrecoverable b error and lets it pollute quantization decisions in L and a
(visible noise, worms); the shaped variants damp the off-plane component and
produce cleaner output with identical systematic bias.

Historical record (see README.md); the original hardcoded output paths were
sanitized to prototypes/out/. near_degenerate.py and
photo_pipeline.py exec this file's Oklab section by splitting its source on
the 74-character "# ---...--- palette" marker line, so that marker is
load-bearing: do not edit or move it.
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(7)

# ---------------------------------------------------------------- Oklab

def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(c):
    c = np.asarray(c, dtype=np.float64)
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)

M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])

def srgb_to_oklab(rgb):
    lin = srgb_to_linear(rgb)
    lms = lin @ M1.T
    lms_ = np.cbrt(lms)
    return lms_ @ M2.T

def oklab_to_srgb(lab):
    lms_ = lab @ np.linalg.inv(M2).T
    lms = lms_ ** 3
    lin = lms @ np.linalg.inv(M1).T
    return linear_to_srgb(lin)

def in_gamut(lab, tol=1e-6):
    """sRGB-gamut test with a dual return: a single Oklab color (ndim 1)
    gives a Python bool; an (N, 3) array gives a per-row boolean array."""
    lms_ = lab @ np.linalg.inv(M2).T
    lms = lms_ ** 3
    lin = lms @ np.linalg.inv(M1).T
    return np.all(lin >= -tol) and np.all(lin <= 1.0 + tol) if lab.ndim == 1 \
        else np.all((lin >= -tol) & (lin <= 1 + tol), axis=-1)

# ---------------------------------------------------------------- palette
# Rank-deficient by construction: all colors on the plane b = 0.02 in Oklab.

B_PLANE = 0.02
Ls = [0.40, 0.55, 0.70]
As = [-0.08, 0.00, 0.08]
palette = np.array([[L, a, B_PLANE] for L in Ls for a in As])
assert all(in_gamut(p) for p in palette), "palette color out of gamut"
N = len(palette)
print(f"Palette: {N} colors on plane b={B_PLANE} in Oklab")

# ---------------------------------------------------------------- shaping matrices

EPS = 0.02 ** 2  # variance scale below which an axis is considered degenerate

def shaping_matrix(points, centre=None):
    """Covariance-eigenstructure shaping matrix with gains lam/(lam+EPS)."""
    if centre is None:
        d = points - points.mean(axis=0)
    else:
        d = points - centre
    cov = d.T @ d / len(d)
    lam, V = np.linalg.eigh(cov)
    g = lam / (lam + EPS)
    return V @ np.diag(g) @ V.T, lam, g

M_global, lam_g, g_g = shaping_matrix(palette)
print("Global eigenvalues:", np.round(lam_g, 6), " gains:", np.round(g_g, 3))

# Per-cell: k nearest palette neighbors of each color (Delaunay stand-in;
# qhull degenerates on rank-deficient point sets, kNN does not).
K = 4
M_cell = np.zeros((N, 3, 3))
for i in range(N):
    d = np.linalg.norm(palette - palette[i], axis=1)
    nbr = np.argsort(d)[1:K + 1]
    M_cell[i], lam_i, g_i = shaping_matrix(palette[nbr], centre=palette[i])

# ---------------------------------------------------------------- test image

H = W = 256
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
xn, yn = xx / (W - 1), yy / (H - 1)

img = np.zeros((H, W, 3))
img[..., 0] = 0.40 + 0.30 * xn                       # L: horizontal ramp
img[..., 1] = -0.08 + 0.16 * yn                      # a: vertical ramp
blob = np.exp(-(((xn - 0.5) ** 2 + (yn - 0.5) ** 2) / (2 * 0.18 ** 2)))
img[..., 2] = B_PLANE + 0.09 * blob                  # b: smooth OFF-PLANE demand

# Clip through sRGB so ground truth is representable, then back to Oklab.
img = srgb_to_oklab(np.clip(oklab_to_srgb(img), 0, 1))
offplane_demand = img[..., 2] - B_PLANE
print(f"Off-plane demand: max {offplane_demand.max():.4f}, mean {offplane_demand.mean():.4f}")

# ---------------------------------------------------------------- error diffusion

FS = [(0, 1, 7 / 16), (1, -1, 3 / 16), (1, 0, 5 / 16), (1, 1, 1 / 16)]

def dither(image, mode):
    """mode: 'baseline' | 'global' | 'cell'"""
    buf = image.copy()
    out_idx = np.zeros((H, W), dtype=np.int64)
    for y in range(H):
        rng_x = range(W) if y % 2 == 0 else range(W - 1, -1, -1)
        sgn = 1 if y % 2 == 0 else -1
        for x in rng_x:
            px = buf[y, x]
            i = np.argmin(((palette - px) ** 2).sum(axis=1))
            out_idx[y, x] = i
            e = px - palette[i]
            if mode == "global":
                e = M_global @ e
            elif mode == "cell":
                e = M_cell[i] @ e
            for dy, dx, w in FS:
                ny, nx = y + dy, x + sgn * dx
                if 0 <= ny < H and 0 <= nx < W:
                    buf[ny, nx] += w * e
    return palette[out_idx], out_idx

results = {}
for mode in ["baseline", "global", "cell"]:
    print(f"dithering: {mode} ...")
    results[mode] = dither(img, mode)[0]

# ---------------------------------------------------------------- metrics

def metrics(dith):
    E = dith - img
    bias = E.mean(axis=(0, 1))
    # tone fidelity: blur both, mean Oklab distance
    blur_d = np.stack([gaussian_filter(dith[..., c], 1.5) for c in range(3)], -1)
    blur_o = np.stack([gaussian_filter(img[..., c], 1.5) for c in range(3)], -1)
    tone = np.linalg.norm(blur_d - blur_o, axis=-1).mean()
    # noise: per-channel std of error
    stds = E.std(axis=(0, 1))
    # spatial spectrum of a channel's error: low-frequency (worm) energy fraction
    def lowfreq_frac(ch):
        e = E[..., ch] - E[..., ch].mean()
        P = np.abs(np.fft.fftshift(np.fft.fft2(e))) ** 2
        cy, cx = H // 2, W // 2
        r = np.hypot(*np.mgrid[-cy:H - cy, -cx:W - cx])
        P[cy, cx] = 0
        return P[(r > 0) & (r < 0.10 * (H / 2))].sum() / P[r > 0].sum()
    # angular anisotropy of mid-frequency spectrum (L channel)
    def anisotropy(ch):
        e = E[..., ch] - E[..., ch].mean()
        P = np.abs(np.fft.fftshift(np.fft.fft2(e))) ** 2
        cy, cx = H // 2, W // 2
        gy, gx = np.mgrid[-cy:H - cy, -cx:W - cx]
        r = np.hypot(gy, gx)
        th = np.arctan2(gy, gx) % np.pi
        band = (r > 0.15 * (H / 2)) & (r < 0.45 * (H / 2))
        hist, _ = np.histogram(th[band], bins=18, weights=P[band])
        cnt, _ = np.histogram(th[band], bins=18)
        prof = hist / np.maximum(cnt, 1)
        return prof.std() / prof.mean()
    return dict(bias=bias, bias_norm=np.linalg.norm(bias), tone=tone,
                stds=stds, lowfreq_L=lowfreq_frac(0), lowfreq_b=lowfreq_frac(2),
                aniso_L=anisotropy(0))

table = {m: metrics(d) for m, d in results.items()}
print(f"\n{'metric':<26}{'baseline':>12}{'global':>12}{'cell':>12}")
rows = [
    ("systematic bias |mean E|", "bias_norm", "{:.5f}"),
    ("bias b component", None, None),
    ("tone error (blurred dE)", "tone", "{:.5f}"),
    ("noise std L", None, None),
    ("noise std a", None, None),
    ("noise std b", None, None),
    ("low-freq energy frac (L)", "lowfreq_L", "{:.4f}"),
    ("low-freq energy frac (b)", "lowfreq_b", "{:.4f}"),
    ("spectral anisotropy (L)", "aniso_L", "{:.4f}"),
]
def val(m, key, comp=None):
    t = table[m]
    if key == "bias_b": return t["bias"][2]
    if key and key.startswith("std"): return t["stds"][int(key[-1])]
    return t[key]
for label, key, fmt in rows:
    if label == "bias b component":
        vals = [table[m]["bias"][2] for m in results]; fmt = "{:.5f}"
    elif label.startswith("noise std"):
        c = {"L": 0, "a": 1, "b": 2}[label[-1]]
        vals = [table[m]["stds"][c] for m in results]; fmt = "{:.5f}"
    else:
        vals = [table[m][key] for m in results]
    print(f"{label:<26}" + "".join(f"{fmt.format(v):>12}" for v in vals))

# ---------------------------------------------------------------- figures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def to_rgb(lab):
    return np.clip(oklab_to_srgb(lab), 0, 1)

crop = (slice(64, 192), slice(64, 192))  # center crop over the off-plane blob

fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
panels = [("original (continuous)", img)] + [
    (name, results[m]) for name, m in
    [("vector Floyd-Steinberg", "baseline"),
     ("global-covariance shaped", "global"),
     ("per-cell shaped", "cell")]]
for ax, (title, lab) in zip(axes[0], panels):
    ax.imshow(to_rgb(lab)); ax.set_title(title, fontsize=11); ax.axis("off")
for ax, (title, lab) in zip(axes[1], panels):
    ax.imshow(to_rgb(lab[crop])); ax.set_title(title + " (center crop)", fontsize=10); ax.axis("off")
fig.suptitle("Rank-deficient palette (9 colors on b=0.02 plane), image with smooth off-plane b demand",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "comparison.png", dpi=130)

# error spectra, L channel
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, m in zip(axes2, ["baseline", "global", "cell"]):
    e = (results[m] - img)[..., 0]
    P = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(e - e.mean()))) ** 2 + 1e-12)
    ax.imshow(P, cmap="magma"); ax.set_title(f"{m}: log power, L error", fontsize=10); ax.axis("off")
fig2.suptitle("Error power spectra (L channel): blue-noise ring vs low-frequency worm energy", fontsize=11)
fig2.tight_layout()
fig2.savefig(OUT / "spectra.png", dpi=130)
print("\nfigures written")
