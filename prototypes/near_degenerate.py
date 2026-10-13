"""Near-degenerate condition: palette b jittered slightly off the plane.

Historical record (see README.md); the original hardcoded output paths were
sanitized to prototypes/out/. Run from the prototypes/ folder: the exec
line below reads palette_aware_dither.py by relative path and splits its
source on a 74-character marker line.
"""
import numpy as np
from pathlib import Path
exec(open("palette_aware_dither.py").read().split("# ---------------------------------------------------------------- palette")[0])

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

B_PLANE = 0.02
Ls = [0.40, 0.55, 0.70]
As = [-0.08, 0.00, 0.08]
jit = np.array([+1,-1,+1,-1,0,-1,+1,-1,+1]) * 0.006   # tiny b spread
palette = np.array([[L, a, B_PLANE] for L in Ls for a in As])
palette[:, 2] += jit
assert all(in_gamut(p) for p in palette)
N = len(palette)
print(f"Palette: {N} colors, b spread std={palette[:,2].std():.4f} (near-degenerate)")

EPS = 0.02 ** 2
def shaping_matrix(points, centre=None):
    """Covariance-eigenstructure shaping matrix with gains lam/(lam+EPS)."""
    d = points - (points.mean(axis=0) if centre is None else centre)
    cov = d.T @ d / len(d)
    lam, V = np.linalg.eigh(cov)
    g = lam / (lam + EPS)
    return V @ np.diag(g) @ V.T, lam, g

M_global, lam_g, g_g = shaping_matrix(palette)
print("Global eigenvalues:", np.round(lam_g, 6), " gains:", np.round(g_g, 3))
K = 4
M_cell = np.zeros((N, 3, 3))
for i in range(N):
    d = np.linalg.norm(palette - palette[i], axis=1)
    nbr = np.argsort(d)[1:K+1]
    M_cell[i], _, _ = shaping_matrix(palette[nbr], centre=palette[i])

H = W = 256
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
xn, yn = xx/(W-1), yy/(H-1)
img = np.zeros((H, W, 3))
img[...,0] = 0.40 + 0.30*xn
img[...,1] = -0.08 + 0.16*yn
blob = np.exp(-(((xn-0.5)**2 + (yn-0.5)**2)/(2*0.18**2)))
img[...,2] = B_PLANE + 0.09*blob
img = srgb_to_oklab(np.clip(oklab_to_srgb(img), 0, 1))

FS = [(0,1,7/16),(1,-1,3/16),(1,0,5/16),(1,1,1/16)]
def dither(image, mode):
    """mode: 'baseline' | 'global' | 'cell'"""
    buf = image.copy()
    out = np.zeros((H, W), dtype=np.int64)
    for y in range(H):
        rx = range(W) if y%2==0 else range(W-1,-1,-1)
        sgn = 1 if y%2==0 else -1
        for x in rx:
            px = buf[y,x]
            i = np.argmin(((palette-px)**2).sum(axis=1))
            out[y,x] = i
            e = px - palette[i]
            if mode=="global": e = M_global @ e
            elif mode=="cell": e = M_cell[i] @ e
            for dy,dx,w in FS:
                ny,nx = y+dy, x+sgn*dx
                if 0<=ny<H and 0<=nx<W: buf[ny,nx] += w*e
    return palette[out]

results = {m: dither(img,m) for m in ["baseline","global","cell"]}
print("dithering done")

from scipy.ndimage import gaussian_filter
def metrics(dith):
    E = dith - img
    bias = E.mean(axis=(0,1))
    bd = np.stack([gaussian_filter(dith[...,c],1.5) for c in range(3)],-1)
    bo = np.stack([gaussian_filter(img[...,c],1.5) for c in range(3)],-1)
    tone = np.linalg.norm(bd-bo, axis=-1).mean()
    stds = E.std(axis=(0,1))
    def lowfreq(ch):
        e = E[...,ch]-E[...,ch].mean()
        P = np.abs(np.fft.fftshift(np.fft.fft2(e)))**2
        cy,cx = H//2,W//2
        r = np.hypot(*np.mgrid[-cy:H-cy,-cx:W-cx]); P[cy,cx]=0
        return P[(r>0)&(r<0.10*(H/2))].sum()/P[r>0].sum()
    return bias, tone, stds, lowfreq(0), lowfreq(1)

print(f"\n{'metric':<26}{'baseline':>12}{'global':>12}{'cell':>12}")
tab = {m: metrics(d) for m,d in results.items()}
names = ["baseline","global","cell"]
print(f"{'bias norm':<26}" + "".join(f"{np.linalg.norm(tab[m][0]):>12.5f}" for m in names))
print(f"{'bias b':<26}" + "".join(f"{tab[m][0][2]:>12.5f}" for m in names))
print(f"{'tone error (blurred dE)':<26}" + "".join(f"{tab[m][1]:>12.5f}" for m in names))
for c,ch in enumerate("Lab"):
    print(f"{'noise std '+ch:<26}" + "".join(f"{tab[m][2][c]:>12.5f}" for m in names))
print(f"{'low-freq frac L':<26}" + "".join(f"{tab[m][3]:>12.4f}" for m in names))
print(f"{'low-freq frac a':<26}" + "".join(f"{tab[m][4]:>12.4f}" for m in names))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def to_rgb(lab): return np.clip(oklab_to_srgb(lab),0,1)
crop = (slice(64,192), slice(64,192))
fig, axes = plt.subplots(2,4, figsize=(16,8.2))
panels = [("original (continuous)", img),
          ("vector Floyd-Steinberg", results["baseline"]),
          ("global-covariance shaped", results["global"]),
          ("per-cell shaped", results["cell"])]
for ax,(t,lab) in zip(axes[0],panels):
    ax.imshow(to_rgb(lab)); ax.set_title(t,fontsize=11); ax.axis("off")
for ax,(t,lab) in zip(axes[1],panels):
    ax.imshow(to_rgb(lab[crop])); ax.set_title(t+" (center crop)",fontsize=10); ax.axis("off")
fig.suptitle("Near-degenerate palette (b spread std 0.0057) with smooth off-plane b demand", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "comparison_near.png", dpi=130)

fig2, axes2 = plt.subplots(1,3, figsize=(13,4.4))
for ax,m in zip(axes2, names):
    e = (results[m]-img)[...,0]
    P = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(e-e.mean())))**2 + 1e-12)
    ax.imshow(P, cmap="magma"); ax.set_title(f"{m}: log power, L error", fontsize=10); ax.axis("off")
fig2.suptitle("Error power spectra (L channel), near-degenerate palette", fontsize=11)
fig2.tight_layout(); fig2.savefig(OUT / "spectra_near.png", dpi=130)
print("figures written")
