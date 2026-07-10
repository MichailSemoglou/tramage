# Prototypes

This folder holds the three research scripts in which the algorithm in this
package was first worked out and tested. They are kept verbatim, as the
historical record of the work, rather than tidied into the package’s style.
The package under `src/tramage/` is the maintained, tested successor; treat
these as notebooks, not as an interface.

## Files

- `palette_aware_dither.py`: the original prototype and evaluation. Builds a
  deliberately rank-deficient palette (nine colours on a single plane in
  Oklab), diffuses a test image against it under three variants (vector
  Floyd-Steinberg, global-covariance shaping, per-cell shaping), and reports
  bias, tone error, noise, and spectral measures for each. This is where the
  inertness lemma was first checked empirically.
- `near_degenerate.py`: the same test with the palette jittered slightly off
  the plane, so it is close to rank-deficient rather than exactly so. This is
  the pathological case: shaping becomes a fidelity-versus-noise trade-off
  rather than a free improvement, and the eigenvalue floor `eps` is the knob.
- `photo_pipeline.py`: the real-photograph test. Extracts a ten-colour
  palette from a photograph by k-means clustering in Oklab (which turns out
  to be naturally near-one-dimensional), then runs the same three-way
  comparison.

## Running them

Each script depends on the one before it: `near_degenerate.py` and
`photo_pipeline.py` both read `palette_aware_dither.py` from disk with
`exec(open(...).read()...)` to reuse its Oklab conversion functions, rather
than importing them, which is why they must be run from inside this folder.
They also write figures to and, for `photo_pipeline.py`, read a photograph
from, paths on the machine they were first written on (`/home/claude/...`
and `/mnt/user-data/uploads/...`), which will not resolve elsewhere. Update
those paths if you want to rerun them; the logic itself needs no changes.

They require `numpy`, `scipy`, `matplotlib`, and, for `photo_pipeline.py`,
`pillow`.
