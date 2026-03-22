from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# pydantic-ai's GoogleProvider requires GOOGLE_API_KEY internally.
# Allow users to set either name -- whichever is present wins.
if "GOOGLE_API_KEY" not in os.environ and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]



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

MODEL_III_RESOLUTION = (64, 64)       # simple_sim_2() always produces 64x64
MODEL_III_PIXEL_SCALE = 0.05          # arcsec/px (HST ACS/WFC F814W-like)
MODEL_III_INSTRUMENT = "HST ACS/WFC F814W (simulated: 64x64, Gaussian PSF FWHM=0.067\")"

# Generation limits
MAX_IMAGES_PER_REQUEST = 100

# Supported LLM models shown in the frontend dropdown.
# Each entry: (model_id_string, display_label)
SUPPORTED_MODELS: list[tuple[str, str]] = [
    ("google-gla:gemini-2.0-flash",       "Gemini 2.0 Flash"),
    ("google-gla:gemini-2.0-flash-lite",  "Gemini 2.0 Flash Lite"),
    ("google-gla:gemini-flash-latest",    "Gemini Flash (latest)"),
    ("openai:gpt-4o-mini",               "OpenAI GPT-4o Mini"),
    ("openai:gpt-4o",                    "OpenAI GPT-4o"),
    ("ollama:custom",                    "Ollama (local)"),
]


# API Key Rotation

class KeyRotator:
    """
    Manages a pool of API keys for a given provider, rotating on quota exhaustion.

    Keys are loaded from numbered environment variables:
        GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...
        OPENAI_API_KEY, OPENAI_API_KEY_2, OPENAI_API_KEY_3, ...

    The un-numbered var is always tried first (index 0).
    """

    _ENV_PREFIX: dict[str, str] = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    def __init__(self, provider: str) -> None:
        """
        Args:
            provider: "gemini" or "openai".
        """
        self.provider = provider.lower()
        self._keys: list[str] = self._load_keys()
        self._index: int = 0

    def _load_keys(self) -> list[str]:
        """Collect all keys for this provider from environment variables."""
        base = self._ENV_PREFIX.get(self.provider, "")
        if not base:
            return []

        keys: list[str] = []
        # Un-numbered base key
        if v := os.environ.get(base):
            keys.append(v)
        # Numbered extras: _2, _3, _4, ...
        for n in range(2, 20):
            if v := os.environ.get(f"{base}_{n}"):
                keys.append(v)
        return keys

    @property
    def total_keys(self) -> int:
        """Total number of keys loaded for this provider."""
        return len(self._keys)

    @property
    def current_index(self) -> int:
        """Zero-based index of the currently active key."""
        return self._index

    def current_key(self) -> str | None:
        """Return the currently active key, or None if the pool is empty."""
        if not self._keys:
            return None
        return self._keys[self._index]

    def rotate(self) -> bool:
        """
        Advance to the next key in the pool.

        Returns:
            True if a new key is now active, False if all keys are exhausted.
        """
        if self._index + 1 < len(self._keys):
            self._index += 1
            return True
        return False

    def inject(self, override: str | None = None) -> None:
        """
        Write the active key (or the override) into os.environ so that
        pydantic-ai picks it up when creating an LLM client.

        Args:
            override: If provided, use this key instead of the pool key.
                      Does NOT advance the pool index.
        """
        base = self._ENV_PREFIX.get(self.provider, "")
        if not base:
            return
        key = override if override else self.current_key()
        if key:
            os.environ[base] = key

    def status_label(self) -> str:
        """Human-readable label describing the current key source."""
        if not self._keys:
            return "No key loaded"
        if self._index == 0:
            return f"Using env key 1 of {self.total_keys}"
        return f"Using env key {self._index + 1} of {self.total_keys} (rotated)"


# Agent Dependencies (injected via RunContext)

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
