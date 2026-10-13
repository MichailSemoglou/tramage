"""Fetch the CQ100 dataset (manual download; Mendeley Data requires authentication).

CQ100: 100 images (.ppm), 8 categories, CC BY-SA 4.0.
Paper: Celebi & Perez-Delgado, J. Electronic Imaging 32(3), 033019, 2023,
  doi:10.1117/1.JEI.32.3.033019.
Dataset DOI: 10.17632/vw5ys9hfxw.4

Automated download is not available (Mendeley Data requires browser-based
authentication). This script exits with instructions instead.

After downloading, extract the archive to ``experiments/corpora/cq100/`` so
that ``run_cq100.py`` and ``run_geometry.py`` discover the images. The zip
contains a nested ``CQ100/`` folder; the images must sit directly in the
corpus directory, so flatten it if needed.

Run from the repository root::

    python experiments/cq100_quality/fetch_cq100.py
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "corpora" / "cq100"


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        images = (list(OUT.glob("*.ppm")) + list(OUT.glob("*.png"))
                  + list(OUT.glob("*.jpg")))
        print(f"CQ100 already present: {len(images)} images in {OUT}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    print(
        "CQ100 cannot be downloaded automatically.\n"
        "\n"
        "  1. Visit https://data.mendeley.com/datasets/vw5ys9hfxw\n"
        "  2. Click 'Download All' (requires free Mendeley account)\n"
        "  3. Extract the archive into:\n"
        f"     {OUT.resolve()}\n"
        "\n"
        "After extraction, the directory should contain 100 .ppm images.\n"
        "The zip nests them under a CQ100/ folder; move the images up so they\n"
        "sit directly in the directory above. Then re-run this script to\n"
        "confirm, followed by:\n"
        "\n"
        "  python experiments/cq100_quality/run_cq100.py\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
