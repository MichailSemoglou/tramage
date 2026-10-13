"""Tramage: palette-structure-aware error diffusion dithering.

Standard vector error diffusion propagates the full quantization error.
Tramage instead reshapes that error along the palette's own geometry in
Oklab before propagating it, so error is passed through where the palette
has room for it and damped where it does not. See the project README and
docs/for-designers.md for the design case, and docs/mathematical-appendix.md
for the proof that shaping is inert on error the palette cannot represent.
"""

from .colour import (
    cielab_to_srgb,
    in_gamut,
    oklab_to_srgb,
    oklab_to_xyz,
    srgb_to_cielab,
    srgb_to_oklab,
    srgb_to_xyz,
)
from .diffusion import DEFAULT_WEIGHTS, diffuse
from .dv_filter import DVFilters, dv_diffuse, dv_filters
from .kb_filter import KBFilters, kb_diffuse, kb_filters
from .metrics import (
    blurred_tone_error,
    lowfreq_energy_fraction,
    noise_std,
    scielab_de,
    spectral_anisotropy,
    systematic_bias,
)
from .palette import (
    EigenspectrumReport,
    describe_palette,
    eigenspectrum_report,
    extract_palette,
)
from .shaping import (
    DEFAULT_EPS,
    global_shaping_matrix,
    per_cell_shaping_matrices,
    shaping_matrix,
)
from .swatches import load_ase, load_gpl, load_hex_list, load_swatch

__version__ = "0.1.0"

__all__ = [
    "srgb_to_oklab",
    "oklab_to_srgb",
    "srgb_to_cielab",
    "cielab_to_srgb",
    "srgb_to_xyz",
    "oklab_to_xyz",
    "in_gamut",
    "diffuse",
    "DEFAULT_WEIGHTS",
    "shaping_matrix",
    "global_shaping_matrix",
    "per_cell_shaping_matrices",
    "DEFAULT_EPS",
    "extract_palette",
    "eigenspectrum_report",
    "EigenspectrumReport",
    "describe_palette",
    "load_ase",
    "load_gpl",
    "load_hex_list",
    "load_swatch",
    "kb_filters",
    "kb_diffuse",
    "KBFilters",
    "dv_filters",
    "dv_diffuse",
    "DVFilters",
    "scielab_de",
    "blurred_tone_error",
    "lowfreq_energy_fraction",
    "noise_std",
    "spectral_anisotropy",
    "systematic_bias",
]
