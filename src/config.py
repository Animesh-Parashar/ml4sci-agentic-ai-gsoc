from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Default physical constants / parameter ranges

# Cosmology defaults (matching DeepLenseSim)
DEFAULT_H0 = 70        # Hubble constant [km/s/Mpc]
DEFAULT_OM0 = 0.3      # Matter density parameter
DEFAULT_OB0 = 0.05     # Baryon density parameter

# Halo defaults
DEFAULT_HALO_MASS = 1e12       # Solar masses
DEFAULT_Z_HALO = 0.5           # Halo redshift
DEFAULT_Z_SOURCE = 1.0         # Source galaxy redshift

# Axion defaults
DEFAULT_AXION_MASS_LOG_MIN = -24   # log10(mass/eV) minimum
DEFAULT_AXION_MASS_LOG_MAX = -22   # log10(mass/eV) maximum
DEFAULT_VORTEX_MASS = 3e10         # Solar masses

# Model-specific configs
MODEL_I_RESOLUTION = (150, 150)
MODEL_I_PIXEL_SCALE = 0.05     # arcsec/px
MODEL_I_INSTRUMENT = "Gaussian PSF (FWHM=0.087 arcsec)"

MODEL_II_RESOLUTION = (64, 64)
MODEL_II_PIXEL_SCALE = 0.101   # Euclid VIS arcsec/px
MODEL_II_INSTRUMENT = "Euclid VIS (6-year coadd)"

# Generation limits
MAX_IMAGES_PER_REQUEST = 100


# Agent Dependencies (injected via RunContex

@dataclass
class AgentDependencies:
    """
    Shared state injected into every tool call via Pydantic AI's RunContext.
    """
    output_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "outputs"
        )
    )

    def ensure_output_dir(self) -> Path:
        """Create output directory if it doesn't exist and return Path."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
