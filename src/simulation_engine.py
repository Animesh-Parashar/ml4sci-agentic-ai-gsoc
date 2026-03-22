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
    MODEL_III_RESOLUTION,
    MODEL_III_PIXEL_SCALE,
    MODEL_III_INSTRUMENT,
)

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Wraps DeepLenseSim's ``DeepLens`` class to generate strong gravitational
    lensing images with structured metadata.

    Supports Model_I (150x150, Gaussian PSF), Model_II (64x64, Euclid),
    and Model_III (HST, magnitude-based source, old CDM sampler).
    """

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
        # import matplotlib inside the worker thread, after fork
        # Importing at module level forces Agg on any caller (notebooks, etc.)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from deeplense.lens import DeepLens

        if params.random_seed is not None:
            np.random.seed(params.random_seed + index)

        try:
            # Only pass axion_mass when it is not None 
            # DeepLens.__init__ may not accept None for axion_mass, which would
            # cause a crash for no_sub and cdm simulations.
            lens_kwargs: dict = dict(
                H0=70,
                Om0=0.3,
                Ob0=0.05,
                z_halo=params.z_halo,
                z_gal=params.z_source,
            )
            if params.axion_mass is not None:
                lens_kwargs["axion_mass"] = params.axion_mass

            lens = DeepLens(**lens_kwargs)

            lens.make_single_halo(params.halo_mass)

            if params.substructure_type == SubstructureType.NO_SUB:
                lens.make_no_sub()
            elif params.substructure_type == SubstructureType.AXION:
                lens.make_vortex(params.vortex_mass)
            elif params.substructure_type == SubstructureType.CDM:
                lens.make_old_cdm()

            if params.model_config_type == ModelConfig.MODEL_I:
                lens.make_source_light()
                lens.simple_sim()
                resolution = MODEL_I_RESOLUTION
                pixel_scale = MODEL_I_PIXEL_SCALE
                instrument = MODEL_I_INSTRUMENT
            elif params.model_config_type == ModelConfig.MODEL_II:
                lens.set_instrument("euclid")
                lens.make_source_light_mag()
                lens.simple_sim_2()
                resolution = MODEL_II_RESOLUTION
                pixel_scale = MODEL_II_PIXEL_SCALE
                instrument = MODEL_II_INSTRUMENT
            else:  # MODEL_III -- HST ACS/WFC (F814W-like)
                # set_instrument('hst') is an unfinished stub in DeepLenseSim:
                # it falls through to else:pass and never sets kwargs_single_band.
                # We implement the missing HST instrument config directly here using
                # realistic ACS/WFC F814W-band parameters so simple_sim_2() works.
                lens.kwargs_single_band = {
                    "pixel_scale":          MODEL_III_PIXEL_SCALE,   # 0.05 arcsec/px
                    "exposure_time":        2028.0,                   # seconds (typical HST)
                    "num_exposures":        1,
                    "sky_brightness":       22.4,                     # AB mag/arcsec^2 (space)
                    "magnitude_zero_point": 25.0,                     # AB zero-point (F814W)
                    "read_noise":           4.0,                      # electrons/pixel
                    "ccd_gain":             2.0,                      # electrons/ADU
                    "psf_type":             "GAUSSIAN",
                    "seeing":               0.067,                    # FWHM arcsec (HST PSF)
                }
                lens.make_source_light_mag()
                lens.simple_sim_2()
                resolution = MODEL_III_RESOLUTION
                pixel_scale = MODEL_III_PIXEL_SCALE
                instrument = MODEL_III_INSTRUMENT

            # Defensive image array extraction
            # DeepLenseSim uses `image_real` after simple_sim() 
            image_array = getattr(lens, "image_real", None)
            if image_array is None:
                image_array = getattr(lens, "image", None)
            if image_array is None:
                raise RuntimeError(
                    "DeepLenseSim did not produce an image array. "
                    "Expected attribute 'image_real' or 'image' on DeepLens object. "
                    "Check your DeepLenseSim installation version."
                )
            image_array = np.asarray(image_array, dtype=np.float64)

            sub_label = params.substructure_type.value
            model_label = params.model_config_type.value
            npy_name = f"{sub_label}_{model_label}_{index:04d}.npy"
            npy_path = output_dir / npy_name
            np.save(str(npy_path), image_array)

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

    @classmethod
    async def generate_batch(
        cls,
        params: SimulationParameters,
        output_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SimulationResult:
        """
        Generate a batch of lensing images asynchronously.

        Runs each simulation in a thread pool executor so the event loop
        stays responsive while lenstronomy (CPU-bound) runs in a worker.
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
                f"using {params.model_config_type.value}."
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