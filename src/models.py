from __future__ import annotations

import numpy as np
from enum import Enum
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field, model_validator


# Enums
class SubstructureType(str, Enum):
    """Dark matter substructure types supported by DeepLenseSim."""
    NO_SUB = "no_sub"
    AXION = "axion"
    CDM = "cdm"


class ModelConfig(str, Enum):
    """Model configurations corresponding to DeepLenseSim Model folders."""
    MODEL_I = "Model_I"    # 150x150, Gaussian PSF, simple_sim()
    MODEL_II = "Model_II"  # 64x64, Euclid SimAPI, simple_sim_2()


# Input: Simulation Parameters

class SimulationParameters(BaseModel):
    """
    Validated simulation parameters extracted from user prompt.
    Covers cosmology, halo configuration, dark matter substructure,
    and model-specific settings.
    """

    model_config_type: ModelConfig = Field(
        description="Which DeepLenseSim Model configuration to use. "
                    "Model_I = 150x150 px, Gaussian PSF. "
                    "Model_II = 64x64 px, Euclid instrument."
    )
    substructure_type: SubstructureType = Field(
        description="Type of dark matter substructure: "
                    "'no_sub' (no substructure), "
                    "'axion' (vortex substructure), "
                    "'cdm' (point-mass subhalos)."
    )
    num_images: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of lensing images to generate (1-100)."
    )
    halo_mass: float = Field(
        default=1e12,
        gt=0,
        description="Main dark matter halo mass in solar masses. "
                    "Typical: 1e12 M_sun."
    )
    z_halo: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Redshift of the dark matter halo (lens). "
                    "Must be less than source redshift."
    )
    z_source: float = Field(
        default=1.0,
        ge=0.1,
        le=5.0,
        description="Redshift of the source (background) galaxy. "
                    "Must be greater than halo redshift."
    )
    axion_mass: Optional[float] = Field(
        default=None,
        description="Axion particle mass in eV. Required for 'axion' substructure. "
                    "Typical range: 1e-24 to 1e-22 eV. "
                    "If not specified for axion sims, a random mass in this range is drawn."
    )
    vortex_mass: float = Field(
        default=3e10,
        gt=0,
        description="Total vortex mass in solar masses for axion simulations. "
                    "Typical: 3e10 M_sun."
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Optional random seed for reproducibility."
    )

    # Allow arbitrary types for numpy operations inside validators
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_physics(self) -> "SimulationParameters":
        """Ensure physical consistency of parameters."""
        # Source must be behind the lens
        if self.z_source <= self.z_halo:
            raise ValueError(
                f"Source redshift ({self.z_source}) must be greater than "
                f"halo redshift ({self.z_halo})."
            )
        # Auto-assign axion mass if needed
        if self.substructure_type == SubstructureType.AXION and self.axion_mass is None:
            self.axion_mass = float(10 ** np.random.uniform(-24, -22))
        return self


# Output: Per-Image Metadata

class ImageMetadata(BaseModel):
    """Metadata for a single generated lensing image."""

    image_index: int = Field(description="Index of this image in the batch.")
    filename: str = Field(description="Filename of the saved .npy file.")
    png_filename: Optional[str] = Field(
        default=None,
        description="Filename of the rendered .png preview."
    )
    substructure_type: SubstructureType
    model_config_type: ModelConfig
    resolution: tuple[int, int] = Field(
        description="Image resolution in pixels (height, width)."
    )
    halo_mass: float
    z_halo: float
    z_source: float
    axion_mass: Optional[float] = None
    pixel_scale_arcsec: float = Field(
        description="Pixel scale in arcseconds per pixel."
    )
    instrument: str = Field(
        description="Instrument/PSF configuration used."
    )
    generation_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of image generation."
    )


# Output: Full Simulation Result

class SimulationResult(BaseModel):
    """Structured output from a complete simulation run."""

    success: bool = Field(description="Whether the simulation completed successfully.")
    message: str = Field(description="Human-readable status message.")
    parameters_used: SimulationParameters = Field(
        description="The final resolved parameters used for generation."
    )
    images: list[ImageMetadata] = Field(
        default_factory=list,
        description="List of metadata for each generated image."
    )
    total_images_generated: int = Field(
        default=0,
        description="Total number of images successfully generated."
    )
    output_directory: str = Field(
        description="Absolute path to the output directory."
    )
    generation_time_seconds: float = Field(
        default=0.0,
        description="Total wall-clock time for the simulation batch."
    )
