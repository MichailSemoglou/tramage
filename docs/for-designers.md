# Why your brand palette changes what good dithering means

## The short version

Dithering trades colors you do not have for a pattern of colors you do
have, arranged so that, at a glance, they read as the color you wanted.
Classic error diffusion (Floyd-Steinberg and its relatives) does this by a
simple rule: pick the nearest available color, then hand the leftover
difference to the next pixel, so the running average stays honest. That
rule does not ask what kind of palette it is working with. Tramage does ask,
and the answer changes where the leftover difference is allowed to go.

## A palette is a shape, not just a list

Three brand colors are not just three swatches. Plotted in a perceptual
color space, they form a shape: a triangle, most often, sitting somewhere
in the space of everything the eye can distinguish. That shape has a size
and an orientation. Some palettes are broad and roughly ball-shaped: warm to
cool, light to dark, saturated to muted, with real room in every direction.
Others are flat: a family of tints and shades of a single hue, for example,
which might vary a great deal in lightness but hardly at all in the
direction that separates warm from cool.

This matters because of what happens when a photograph or gradient asks for
a color the palette does not have. The classic algorithm computes the gap
between what was wanted and what was used, then posts that whole gap onward
as debt for nearby pixels to repay. If the gap happens to point in a
direction the palette has no room in at all, that debt can never be repaid.
It has nowhere to go. It sits in the running total, nudging every later
decision a little off course, and the visible result is a low, slow
graininess, sometimes called worms, that has nothing to do with the image
and everything to do with the arithmetic.

## What shaping does instead

Before dithering starts, tramage looks at the palette’s own shape: its
spread along every direction in Oklab, a color space built so that
distance corresponds reasonably well to how different two colors actually
look. Directions where the palette has real extent are left alone, debt
posted along them is real and repayable, so it passes through unchanged.
Directions where the palette is flat, or nearly flat, are damped. The
method does not decide this by inspecting the image; it reads it directly
from the palette’s geometry, once, before any pixel is touched.

The practical effect: unavoidable error stops accumulating in directions
the palette can never use, and instead shows up, if it shows up at all, as
finer, quieter noise rather than slow drift. The gallery’s near-degenerate
comparison (see gallery/README.md) shows this directly: the same image, the
same palette, dithered with and without shaping, with the shaped version
visibly calmer in the center crop.

## What this does not do

It does not invent colors the palette lacks. A photograph’s sunset will
not gain an orange your brand palette does not contain. What changes is
narrower and more honest than that: where the unavoidable noise from
approximating that sunset is allowed to sit. Given a color the palette
cannot represent, tramage decides whether the resulting noise is spread
finely (usually less noticeable) or left to accumulate as drift (usually
more noticeable). It cannot repay a debt the palette has no means to repay;
no method can. See the gallery’s null-result comparison for a palette where
this is exactly and provably true: shaping has nothing to do, because there
is nothing unpayable to protect against.

## When this matters and when it does not

If your palette is broad, spanning real variation in lightness, hue and
chroma, this method will behave almost exactly like ordinary error
diffusion, because there is little near-degeneracy for it to act on. Where
it matters is the case many brand and print palettes actually fall into: a
family of related tones that, plotted honestly, turn out to be close to
flat. That is precisely the case where classic error diffusion’s blind
rule does the most damage, and precisely the case this method was built to
address.

## Where to go next

`docs/quickstart.md` gets you from an image to a dithered result. This
document is the intuition; `docs/mathematical-appendix.md` is the proof, for
anyone who wants to see exactly why error orthogonal to a palette’s shape
can never change a decision, and exactly what is being traded away when a
palette is close to, but not exactly, that shape.
