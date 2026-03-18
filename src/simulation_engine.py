from __future__ import annotations

import logging
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime
from collections.abc import Callable
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless generation
import matplotlib.pyplot as plt

from .models import (
    SimulationParameters,
    SimulationResult,
    ImageMetadata,
    SubstructureType,
    ModelConfig,
)
from .config import (
    MODEL_I_RESOLUTION,
    MODEL_I_PIXEL_SCALE,
    MODEL_I_INSTRUMENT,
    MODEL_II_RESOLUTION,
    MODEL_II_PIXEL_SCALE,
    MODEL_II_INSTRUMENT,
)

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Wraps DeepLenseSim's ``DeepLens`` class to generate strong gravitational
    lensing images with structured metadata.

    Supports Model_I (150x150, Gaussian PSF) and Model_II (64x64, Euclid).
    """

    # Single-image generation (synchronous, CPU-bound)

    @staticmethod
    def _run_single_simulation(
        params: SimulationParameters,
        index: int,
        output_dir: Path,
    ) -> ImageMetadata:
        """
        Generate one lensing image using DeepLenseSim.

        Args:
            params: Validated simulation parameters.
            index: Image index within the batch.
            output_dir: Directory to save output files.

        Returns:
            ImageMetadata for the generated image.

        Raises:
            RuntimeError: If the simulation fails.
        """
        # Import here to keep the module importable even without lenstronomy
        from deeplense.lens import DeepLens

        # Set seed for this image if requested
        if params.random_seed is not None:
            np.random.seed(params.random_seed + index)

        try:
            # 1. Create lens with cosmology
            lens = DeepLens(
                axion_mass=params.axion_mass,
                H0=70,
                Om0=0.3,
                Ob0=0.05,
                z_halo=params.z_halo,
                z_gal=params.z_source,
            )

            # 2. Main halo (SIE + shear)
            lens.make_single_halo(params.halo_mass)

            # 3. Substructure
            if params.substructure_type == SubstructureType.NO_SUB:
                lens.make_no_sub()
            elif params.substructure_type == SubstructureType.AXION:
                lens.make_vortex(params.vortex_mass)
            elif params.substructure_type == SubstructureType.CDM:
                lens.make_old_cdm()

            # 4. Source light & simulation (model-dependent)
            if params.model_config_type == ModelConfig.MODEL_I:
                lens.make_source_light()
                lens.simple_sim()
                resolution = MODEL_I_RESOLUTION
                pixel_scale = MODEL_I_PIXEL_SCALE
                instrument = MODEL_I_INSTRUMENT
            else:  # MODEL_II
                lens.set_instrument("euclid")
                lens.make_source_light_mag()
                lens.simple_sim_2()
                resolution = MODEL_II_RESOLUTION
                pixel_scale = MODEL_II_PIXEL_SCALE
                instrument = MODEL_II_INSTRUMENT

            # 5. Extract the generated image
            image_array = lens.image_real

            # 6. Save .npy
            sub_label = params.substructure_type.value
            model_label = params.model_config_type.value
            npy_name = f"{sub_label}_{model_label}_{index:04d}.npy"
            npy_path = output_dir / npy_name
            np.save(str(npy_path), image_array)

            # 7. Save .png preview
            png_name = f"{sub_label}_{model_label}_{index:04d}.png"
            png_path = output_dir / png_name
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(np.sqrt(np.clip(image_array, 0, None)), cmap="inferno")
            ax.set_title(f"{sub_label} | {model_label}", fontsize=10)
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(str(png_path), dpi=100, bbox_inches="tight")
            plt.close(fig)

            logger.info("Generated image %d: %s", index, npy_name)

            return ImageMetadata(
                image_index=index,
                filename=npy_name,
                png_filename=png_name,
                substructure_type=params.substructure_type,
                model_config_type=params.model_config_type,
                resolution=resolution,
                halo_mass=params.halo_mass,
                z_halo=params.z_halo,
                z_source=params.z_source,
                axion_mass=params.axion_mass,
                pixel_scale_arcsec=pixel_scale,
                instrument=instrument,
                generation_timestamp=datetime.now().isoformat(),
            )

        except Exception as exc:
            logger.error("Simulation failed for image %d: %s", index, exc)
            raise RuntimeError(
                f"Simulation failed for image {index}: {exc}"
            ) from exc

    # Batch generation (async wrapper)
    @classmethod
    async def generate_batch(
        cls,
        params: SimulationParameters,
        output_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SimulationResult:
        """
        Generate a batch of lensing images asynchronously.

        Runs each simulation in a thread pool to avoid blocking the
        event loop (lenstronomy is CPU-bound).

        Args:
            params: Validated simulation parameters.
            output_dir: Output directory.
            progress_callback: Optional callable(current, total) for progress.

        Returns:
            SimulationResult with metadata for all generated images.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        images: list[ImageMetadata] = []
        errors: list[str] = []

        for i in range(params.num_images):
            if progress_callback:
                progress_callback(i + 1, params.num_images)
            try:
                metadata = await asyncio.to_thread(
                    cls._run_single_simulation,
                    params,
                    i,
                    output_dir,
                )
                images.append(metadata)
            except RuntimeError as exc:
                errors.append(str(exc))
                logger.warning("Skipping image %d due to error: %s", i, exc)

        elapsed = time.time() - start_time

        # Build result
        if errors:
            message = (
                f"Generated {len(images)}/{params.num_images} images "
                f"({len(errors)} failed). Errors: {'; '.join(errors[:3])}"
            )
            success = len(images) > 0
        else:
            message = (
                f"Successfully generated {len(images)} "
                f"{params.substructure_type.value} lensing image(s) "
                f"using {params.model_config_type.value} configuration."
            )
            success = True

        return SimulationResult(
            success=success,
            message=message,
            parameters_used=params,
            images=images,
            total_images_generated=len(images),
            output_directory=str(output_dir),
            generation_time_seconds=round(elapsed, 2),
        )
