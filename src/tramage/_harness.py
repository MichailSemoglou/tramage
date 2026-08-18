"""Shared plumbing for the ``experiments/`` run scripts.

The run scripts under ``experiments/`` repeat the same CSV-resume, dedupe,
scoring, and image-loading logic, differing only in their field names and
per-trial pipeline. These helpers hold that common code in one place. They
are private to the package: import from ``tramage._harness``, not as public
API (they depend on Pillow, which is an optional extra).

``score`` returns the metric fields only; callers add the identity fields
(corpus, image, K, variant/config, eps) for their CSV row.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
from collections.abc import Callable, Hashable
from pathlib import Path

import numpy as np
from PIL import Image

from .colour import oklab_to_srgb, srgb_to_oklab
from .metrics import (
    blurred_tone_error,
    lowfreq_energy_fraction,
    noise_std,
    scielab_de,
    spectral_anisotropy,
    systematic_bias,
)

# A callable that reduces one CSV row (a dict) to a hashable identity key,
# used both to skip already-computed trials and to dedupe the final CSV.
Key = Callable[[dict], Hashable]

# Image suffixes recognised across the corpora (Kodak is PNG; CQ100 may add PPM).
IMAGE_SUFFIXES = (".png", ".jpg", ".ppm")


def corpus_dirs(corpora: Path) -> list[Path]:
    """Sorted corpus subdirectories under ``corpora``."""
    return sorted(d for d in corpora.iterdir() if d.is_dir())


def images_in(corpus: Path) -> list[Path]:
    """Sorted image paths in one corpus directory, grouped by suffix then name."""
    return [
        path
        for suffix in IMAGE_SUFFIXES
        for path in sorted(corpus.glob(f"*{suffix}"))
    ]


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as a float sRGB array in [0, 1], shape ``(H, W, 3)``."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def load_oklab(path: Path) -> np.ndarray:
    """Load an image and convert it to Oklab, shape ``(H, W, 3)``."""
    return srgb_to_oklab(load_rgb(path))


def score(dithered: np.ndarray, original: np.ndarray, *, scielab: bool = True) -> dict:
    """Compute the metric fields shared by every dithering-quality CSV row.

    The bias/noise/tone/low-frequency/anisotropy metrics are space-agnostic
    (they operate on whatever colour space the caller passes in). S-CIELAB
    ΔE additionally requires Oklab input, so callers that score in a
    different working space (the CIELAB arm of the colour-space comparison)
    pass ``scielab=False`` and compute it on the rendered RGB themselves.
    """
    bias = systematic_bias(dithered, original)
    noise = noise_std(dithered, original)
    row = {
        "bias_norm": float(np.linalg.norm(bias)),
        "bias_L": float(bias[0]), "bias_a": float(bias[1]), "bias_b": float(bias[2]),
        "noise_L": float(noise[0]), "noise_a": float(noise[1]), "noise_b": float(noise[2]),
        "tone_error": float(blurred_tone_error(dithered, original)),
        "lowfreq_frac": float(lowfreq_energy_fraction(dithered, original, channel=0)),
        "anisotropy": float(spectral_anisotropy(dithered, original, channel=0)),
    }
    if scielab:
        row["scielab_de"] = float(scielab_de(dithered, original))
    return row


def load_done(csv_path: Path, key: Key) -> set[Hashable]:
    """Return the set of ``key(row)`` identities already present in the CSV."""
    if not csv_path.exists():
        return set()
    done: set[Hashable] = set()
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            done.add(key(row))
    return done


def finalise_csv(
    csv_path: Path,
    fields: list[str],
    key: Key,
    sort_key: Key | None = None,
) -> None:
    """Dedupe and sort the CSV in place. Idempotent; safe to call repeatedly."""
    if not csv_path.exists():
        return
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen: set[Hashable] = set()
    unique: list[dict] = []
    for row in rows:
        k = key(row)
        if k in seen:
            continue
        seen.add(k)
        unique.append(row)
    unique.sort(key=sort_key or key)
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(unique)

# Numeric columns shared by the dithering-quality result CSVs.
QUALITY_NUMERIC_FIELDS = (
    "K", "eps", "bias_norm", "bias_L", "bias_a", "bias_b",
    "noise_L", "noise_a", "noise_b", "tone_error", "lowfreq_frac",
    "anisotropy", "scielab_de",
)


def read_rows(csv_path: Path, numeric_fields: tuple[str, ...] = ()) -> list[dict]:
    """Read a CSV into a list of dicts, coercing ``numeric_fields`` to float."""
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found")
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for k in numeric_fields:
            row[k] = float(row[k])
    return rows


def percentile(arr: np.ndarray, p: float) -> float:
    """p-th percentile of a possibly-empty array; NaN when empty."""
    return float(np.percentile(arr, p)) if len(arr) else float("nan")

def bootstrap_ci(
    values: np.ndarray,
    *,
    ci: float = 0.90,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of paired per-image samples.

    A 90% CI corresponds to two one-sided equivalence tests at alpha = 0.05.
    Fixed ``seed`` keeps the interval reproducible for a given input.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = values.size
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = values[rng.integers(0, n, n)].mean()
    lo = 100.0 * (1.0 - ci) / 2.0
    hi = 100.0 - lo
    return (float(np.percentile(means, lo)), float(np.percentile(means, hi)))


def git_sha(repo_root: Path) -> str:
    """Git commit of the repository, suffixed ``-dirty`` if uncommitted changes exist."""
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout.strip()
    return f"{head}-dirty" if status else head


def corpus_checksum(image_paths: list[Path]) -> str:
    """Deterministic SHA-256 over a corpus: sorted paths plus each file's digest.

    Independent of read order and cheap (one pass over file bytes). Records
    exactly which image bytes a run consumed, for figure provenance.
    """
    digest = hashlib.sha256()
    for path in sorted(image_paths, key=str):
        file_digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                file_digest.update(chunk)
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(file_digest.digest())
    return digest.hexdigest()


def provenance(image_paths: list[Path], repo_root: Path) -> dict:
    """Provenance fields for a run: repo commit and corpus checksum."""
    return {"git_sha": git_sha(repo_root), "corpus_sha256": corpus_checksum(image_paths)}


def make_test_image(b_plane: float = 0.02, height: int = 256, width: int = 256) -> np.ndarray:
    """Synthetic gradient-plus-blob test image, returned in Oklab.

    A linear L ramp along x, a linear a ramp along y, and a smooth Gaussian
    blob on the b axis whose baseline is ``b_plane``. Used by the gallery and
    the epsilon sweep to exercise shaping on a controlled off-plane demand.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    xn, yn = xx / (width - 1), yy / (height - 1)
    img = np.zeros((height, width, 3))
    img[..., 0] = 0.40 + 0.30 * xn
    img[..., 1] = -0.08 + 0.16 * yn
    blob = np.exp(-(((xn - 0.5) ** 2 + (yn - 0.5) ** 2) / (2 * 0.18 ** 2)))
    img[..., 2] = b_plane + 0.09 * blob
    return srgb_to_oklab(np.clip(oklab_to_srgb(img), 0, 1))

