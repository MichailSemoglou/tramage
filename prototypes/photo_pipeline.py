"""Photo pipeline: k-means palette in Oklab, three dithering variants."""
import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.ndimage import gaussian_filter

exec(open("palette_aware_dither.py").read().split("# ---------------------------------------------------------------- palette")[0])

# ---- load and downscale
im = Image.open("/mnt/user-data/uploads/45640.jpeg").convert("RGB")
scale = 320 / max(im.size)
im = im.resize((round(im.size[0]*scale), round(im.size[1]*scale)), Image.LANCZOS)
rgb = np.asarray(im, dtype=np.float64) / 255.0
H, W = rgb.shape[:2]
print(f"working size: {W}x{H}")
img = srgb_to_oklab(rgb)

# ---- k-means palette in Oklab
NPAL = 10
pix = img.reshape(-1, 3)
centroids, labels = kmeans2(pix, NPAL, minit="++", seed=11)
palette = centroids
N = len(palette)
counts = np.bincount(labels, minlength=N) / len(labels)
order = np.argsort(-counts)
palette = palette[order]; counts = counts[order]
print("palette (L, a, b) and share:")
for p, c in zip(palette, counts):
    print(f"  L={p[0]:.3f} a={p[1]:+.3f} b={p[2]:+.3f}  {100*c:5.1f}%")

# ---- shaping matrices
EPS = 0.02 ** 2
def shaping_matrix(points, centre=None):
    d = points - (points.mean(axis=0) if centre is None else centre)
    cov = d.T @ d / len(d)
    lam, V = np.linalg.eigh(cov)
    g = lam / (lam + EPS)
    return V @ np.diag(g) @ V.T, lam, g

M_global, lam_g, g_g = shaping_matrix(palette)
print("\nglobal palette eigenvalues:", np.round(lam_g, 6))
print("global gains g(lam):       ", np.round(g_g, 3))

K = 4
M_cell = np.zeros((N, 3, 3)); cell_gains = []
for i in range(N):
    d = np.linalg.norm(palette - palette[i], axis=1)
    nbr = np.argsort(d)[1:K+1]
    M_cell[i], lam_i, g_i = shaping_matrix(palette[nbr], centre=palette[i])
    cell_gains.append(g_i)
cell_gains = np.array(cell_gains)
print("per-cell gain ranges: min", np.round(cell_gains.min(axis=0),3),
      " max", np.round(cell_gains.max(axis=0),3))

# ---- dither
FS = [(0,1,7/16),(1,-1,3/16),(1,0,5/16),(1,1,1/16)]
def dither(image, mode):
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

results = {}
for m in ["baseline","global","cell"]:
    print("dithering:", m)
    results[m] = dither(img, m)

# ---- metrics
def metrics(dith):
    E = dith - img
    bias = np.linalg.norm(E.mean(axis=(0,1)))
    bd = np.stack([gaussian_filter(dith[...,c],1.5) for c in range(3)],-1)
    bo = np.stack([gaussian_filter(img[...,c],1.5) for c in range(3)],-1)
    tone = np.linalg.norm(bd-bo, axis=-1).mean()
    stds = E.std(axis=(0,1))
    return bias, tone, stds

names = ["baseline","global","cell"]
tab = {m: metrics(d) for m,d in results.items()}
print(f"\n{'metric':<26}" + "".join(f"{m:>12}" for m in names))
print(f"{'systematic bias':<26}" + "".join(f"{tab[m][0]:>12.5f}" for m in names))
print(f"{'tone error (blurred dE)':<26}" + "".join(f"{tab[m][1]:>12.5f}" for m in names))
for c,ch in enumerate("Lab"):
    print(f"{'noise std '+ch:<26}" + "".join(f"{tab[m][2][c]:>12.5f}" for m in names))

# ---- figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def to_rgb(lab): return np.clip(oklab_to_srgb(lab),0,1)

# crop around the eyes: roughly upper third, centred
cy, cx = int(H*0.30), int(W*0.52)
s = 64
crop = (slice(cy-s//2, cy+s//2), slice(cx-s, cx+s))

fig, axes = plt.subplots(2, 4, figsize=(15, 11))
panels = [("original", img), ("vector Floyd-Steinberg", results["baseline"]),
          ("global-covariance shaped", results["global"]), ("per-cell shaped", results["cell"])]
for ax,(t,lab) in zip(axes[0], panels):
    ax.imshow(to_rgb(lab)); ax.set_title(t, fontsize=11); ax.axis("off")
for ax,(t,lab) in zip(axes[1], panels):
    ax.imshow(to_rgb(lab[crop]), interpolation="nearest"); ax.set_title(t+" (eye crop)", fontsize=10); ax.axis("off")
# palette strip
fig.subplots_adjust(bottom=0.10)
strip = fig.add_axes([0.25, 0.03, 0.5, 0.035])
strip.imshow(to_rgb(palette[None,:,:]), aspect="auto"); strip.set_yticks([])
strip.set_xticks(range(N)); strip.set_xticklabels([f"{100*c:.0f}%" for c in counts], fontsize=7)
strip.set_title("extracted 10-colour Oklab palette (share of pixels)", fontsize=9)
fig.suptitle("Photo, k-means palette: baseline vs palette-structure-aware error diffusion", fontsize=12)
fig.savefig("/home/claude/photo_comparison.png", dpi=130, bbox_inches="tight")
print("figure written")

# save individual full-res-of-working-size dithers too
for m in names:
    Image.fromarray((to_rgb(results[m])*255).round().astype(np.uint8)).save(f"/home/claude/photo_{m}.png")
