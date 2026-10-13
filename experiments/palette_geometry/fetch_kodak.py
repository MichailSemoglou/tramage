"""Download the Kodak Lossless True Color Image Suite (Kodak24).

The 24 images are hosted at r0k.us and are in the public domain, released
by the Eastman Kodak Company. They are the standard test set in the
dithering and color-quantization literature and are used here unchanged.
Files already present are skipped, so this is safe to re-run.

Run from the repository root::

    python experiments/palette_geometry/fetch_kodak.py
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

# The r0k.us mirror serves the index at /graphics/kodak/ and the actual
# PNG files one directory deeper, at /graphics/kodak/kodak/kodimNN.png.
BASE = "https://r0k.us/graphics/kodak/kodak/"
OUT = Path(__file__).resolve().parent.parent / "corpora" / "kodak"
N = 24


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(1, N + 1):
        name = f"kodim{i:02d}.png"
        target = OUT / name
        if target.exists():
            print(f"have  {name}")
            continue
        url = BASE + name
        print(f"fetch  {name} <- {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - trusted fixed URL
            target.write_bytes(resp.read())
    print(f"\n{N} images in {OUT}")


if __name__ == "__main__":
    main()
