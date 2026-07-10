# Mathematical appendix

This is the proof underneath docs/for-designers.md. It assumes familiarity
with error diffusion, nearest-neighbour quantisation, and basic linear
algebra.

## Setup

Let $P = \{p_1, \dots, p_n\} \subset \mathbb{R}^3$ be a palette in Oklab.
Nearest-neighbour quantisation of a point $x$ is

$$q(x) = \arg\min_i \lVert x - p_i \rVert^2.$$

Vector error diffusion propagates the error $e = x - p_{q(x)}$ to
neighbouring pixels by adding a weighted share of it to their running
values before they are quantised in turn. This package instead propagates
$e’ = Me$, where $M$ is built once per palette (see below), and reduces to
ordinary error diffusion when $M = I$.

## The shaping matrix

Let $\Sigma$ be the palette’s covariance in Oklab, and let

$$\Sigma = V \Lambda V^{\top}$$

be its eigendecomposition, with $\Lambda = \mathrm{diag}(\lambda_1,
\lambda_2, \lambda_3)$. Define the gain function

$$g(\lambda) = \frac{\lambda}{\lambda + \varepsilon}$$

for a fixed $\varepsilon > 0$, and set

$$M = V \, \mathrm{diag}\bigl(g(\lambda_1), g(\lambda_2), g(\lambda_3)\bigr) \, V^{\top}.$$

$M$ is symmetric, positive semi-definite, and every eigenvalue $g(\lambda_i)$
lies in $[0, 1)$. It acts as identity along directions where $\lambda_i \gg
\varepsilon$ and as a projection towards zero along directions where
$\lambda_i \ll \varepsilon$. This is the same shrinkage form used in ridge
regression and Wiener filtering: $\varepsilon$ plays the role of a noise
floor, and only the share of variance exceeding it is treated as real
signal. Here, the “signal” is palette extent and the “noise floor” is the
smallest variance worth protecting error along.

The per-cell variant applies the same construction locally: for palette
colour $p_i$, its $k$ nearest palette neighbours (a robust stand-in for a
Delaunay neighbourhood, which degenerates for rank-deficient palettes) give
a local covariance and a local matrix $M_i$, rather than one matrix for the
whole palette. This variant is exported but is **not the recommended
method**: on a Kodak24 corpus run (see `experiments/dithering_quality/`)
it matched the global variant on noise reduction but lost on tone accuracy
and the low-frequency worm metric from K=16 onward. Natural-image k-means
palettes are globally degenerate enough that the global matrix already
captures the relevant structure. The construction is documented here for
completeness; the rest of this appendix concerns the global variant.

## Inertness lemma

**Statement.** Let $P$ lie in an affine subspace $A \subset \mathbb{R}^3$,
and let $e^\perp$ be any vector orthogonal to the direction of $A$ (the
linear subspace of differences between points of $A$). Then for every $x$
and every pair $p_i, p_j \in P$,

$$\lVert (x + e^\perp) - p_i \rVert^2 - \lVert (x + e^\perp) - p_j \rVert^2 = \lVert x - p_i \rVert^2 - \lVert x - p_j \rVert^2.$$

Consequently $q(x + e^\perp) = q(x)$: adding $e^\perp$ can never change which
palette colour is nearest.

**Proof.** Expand each squared distance:

$$\lVert (x + e^\perp) - p_i \rVert^2 = \lVert x - p_i \rVert^2 + 2\, e^\perp \cdot (x - p_i) + \lVert e^\perp \rVert^2,$$

and likewise for $p_j$. Subtracting,

$$\lVert (x+e^\perp)-p_i\rVert^2 - \lVert (x+e^\perp)-p_j\rVert^2 = \lVert x-p_i\rVert^2 - \lVert x-p_j\rVert^2 + 2\, e^\perp \cdot (p_j - p_i).$$

Since $p_i, p_j \in A$, the difference $p_j - p_i$ lies in the direction of
$A$, and $e^\perp$ is orthogonal to that direction by hypothesis, so $e^\perp
\cdot (p_j - p_i) = 0$. The extra term vanishes, proving the identity. Since
this holds for every pair $(p_i, p_j)$, the ordering of distances to every
palette colour is unchanged by adding $e^\perp$, so the arg min is
unchanged. $\blacksquare$

**Consequence for shaping.** If $P$ is exactly rank-deficient, its
covariance has an eigenvalue $\lambda = 0$ exactly along the direction
orthogonal to its affine hull, so $g(0) = 0/(0+\varepsilon) = 0$ exactly,
for any $\varepsilon > 0$. $M$ therefore annihilates that direction
completely rather than merely damping it. The lemma above shows this
changes nothing that could have mattered: propagating the raw orthogonal
error and propagating none of it lead to identical quantisation decisions
at every pixel, since neither can ever change an argmin. Shaping is
provably inert here, not approximately so. This is confirmed end to end in
`tests/test_diffusion.py`: baseline and shaped diffusion produce bit-for-bit
identical output on an exactly rank-deficient palette.

## The isotropic limit

If the palette’s covariance is isotropic, $\Sigma = \sigma^2 I$ for some
$\sigma^2 \gg \varepsilon$, every eigenvalue equals $\sigma^2$ and every
gain equals $g(\sigma^2) = \sigma^2 / (\sigma^2 + \varepsilon) \to 1$.
Then

$$M = V \, \mathrm{diag}(1, 1, 1) \, V^{\top} = V V^{\top} = I,$$

since $V$ is orthogonal. Shaping collapses exactly to vector
Floyd-Steinberg. More generally, whenever every eigenvalue of a palette’s
covariance is much larger than $\varepsilon$, whether or not the palette is
isotropic, $M \to I$ and shaping has essentially nothing to do: it only has
purchase where a palette’s extent along some direction is comparable to or
smaller than $\varepsilon$.

## The pathological regime: near-degenerate palettes

The inertness lemma is exact only for exactly rank-deficient palettes.
Real palettes are rarely exactly flat; they are close to flat, with a small
but non-zero eigenvalue along the near-degenerate axis. This is the regime
in which shaping is doing real work, and where it is a genuine trade-off
rather than a free correction.

On a synthetic test with a palette jittered slightly off an exact plane
(eigenvalue along that axis raised from zero to a small positive value),
the baseline showed 1.7 times higher lightness noise than the shaped
variants, while recovering slightly more of the off-plane tone the palette
could not represent exactly. Shaping is a fidelity-versus-noise dial in
this regime, not a strict improvement, and $\varepsilon$ is the knob:
larger values damp more aggressively (more noise reduction, less off-plane
fidelity retained), smaller values approach the unshaped baseline. See
`prototypes/near_degenerate.py` for the test that produced this figure, and
`gallery/README.md` for the corresponding image.

## A real photograph

On a photograph with a ten-colour palette extracted by k-means clustering
in Oklab, the extracted palette turned out to be naturally close to
one-dimensional, with covariance eigenvalues of approximately $1 \times
10^{-6}$, $1.1 \times 10^{-4}$, and $4.1 \times 10^{-2}$: two axes nearly
degenerate, one with real extent. On this palette, shaping won on every
metric measured: 3.1 times lower lightness noise, better tone error, and
lower systematic bias than the unshaped baseline. Unlike the synthetic
near-degenerate test, there was no fidelity cost to report here; the
palette’s near-degeneracy was severe enough, and its real extent
concentrated narrowly enough in one direction, that shaping had almost pure
upside. See `prototypes/photo_pipeline.py` for the pipeline that produced
this result. This is presented as one photograph and one palette, not as a
general guarantee; the planned Kodak corpus run in `experiments/` is aimed
at a broader picture.

## Summary

| Regime                                                                       | What shaping does                                                                                            |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Isotropic, or every eigenvalue $\gg \varepsilon$                             | Nothing: $M \to I$, identical to vector Floyd-Steinberg.                                                     |
| Exactly rank-deficient                                                       | Nothing that could have mattered: provably inert, by the lemma above.                                        |
| Near-degenerate                                                              | A genuine trade: less noise, some loss of unrepresentable off-plane fidelity. $\varepsilon$ tunes the trade. |
| Naturally near-degenerate (for example, a k-means palette from a photograph) | Can win outright on every metric measured, when the palette’s real extent is concentrated narrowly enough.   |
