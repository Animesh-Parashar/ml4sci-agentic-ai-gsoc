from __future__ import annotations

import sys
import textwrap
from typing import Optional

from pydantic_ai import RunContext

from .config import (
    AgentDependencies,
    MODEL_I_RESOLUTION,
    MODEL_I_PIXEL_SCALE,
    MODEL_I_INSTRUMENT,
    MODEL_II_RESOLUTION,
    MODEL_II_PIXEL_SCALE,
    MODEL_II_INSTRUMENT,
    MAX_IMAGES_PER_REQUEST,
)
from .models import (
    SimulationParameters,
    SimulationResult,
    SubstructureType,
    ModelConfig,
)
from .simulation_engine import SimulationEngine


# Tool 1: Generate lensing images


async def generate_lensing_images(
    ctx: RunContext[AgentDependencies],
    model_config_type: str,
    substructure_type: str,
    num_images: int = 1,
    halo_mass: float = 1e12,
    z_halo: float = 0.5,
    z_source: float = 1.0,
    axion_mass: Optional[float] = None,
    vortex_mass: float = 3e10,
    random_seed: Optional[int] = None,
) -> str:
    """Generate strong gravitational lensing images using DeepLenseSim.

    Call this tool ONLY after you have confirmed all parameters with the user.

    Args:
        ctx: Runtime context with agent dependencies.
        model_config_type: Model configuration — "Model_I" (150x150 px, Gaussian PSF)
            or "Model_II" (64x64 px, Euclid instrument).
        substructure_type: Dark matter substructure — "no_sub", "axion", or "cdm".
        num_images: Number of images to generate (1-100).
        halo_mass: Main halo mass in solar masses (default: 1e12).
        z_halo: Halo (lens) redshift (default: 0.5).
        z_source: Source galaxy redshift (default: 1.0). Must exceed z_halo.
        axion_mass: Axion particle mass in eV (required for axion substructure).
            If omitted for axion sims, a random value in [1e-24, 1e-22] is used.
        vortex_mass: Total vortex mass in solar masses (default: 3e10).
        random_seed: Optional random seed for reproducibility.

    Returns:
        A formatted string summarising the simulation results and metadata.
    """
    # Build validated parameters
    params = SimulationParameters(
        model_config_type=ModelConfig(model_config_type),
        substructure_type=SubstructureType(substructure_type),
        num_images=num_images,
        halo_mass=halo_mass,
        z_halo=z_halo,
        z_source=z_source,
        axion_mass=axion_mass,
        vortex_mass=vortex_mass,
        random_seed=random_seed,
    )

    # Resolve output directory
    output_path = ctx.deps.ensure_output_dir()

    # Progress callback for real-time terminal feedback
    def _progress(current: int, total: int) -> None:
        bar_len = 20
        filled = int(bar_len * current / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  ⏳ Generating image {current}/{total} [{bar}]", end="", flush=True)

    # Run simulation
    result: SimulationResult = await SimulationEngine.generate_batch(
        params, output_path, progress_callback=_progress
    )
    print()  # newline after progress bar

    # Format response for the agent
    lines = [
        f"**Simulation {'Completed' if result.success else 'Failed'}**",
        f"- {result.message}",
        f"- Images generated: {result.total_images_generated}",
        f"- Output directory: `{result.output_directory}`",
        f"- Generation time: {result.generation_time_seconds}s",
        "",
        "**Parameters Used:**",
        f"- Model: {params.model_config_type.value}",
        f"- Substructure: {params.substructure_type.value}",
        f"- Halo mass: {params.halo_mass:.2e} M☉",
        f"- z_halo: {params.z_halo}, z_source: {params.z_source}",
    ]
    if params.axion_mass is not None:
        lines.append(f"- Axion mass: {params.axion_mass:.2e} eV")

    if result.images:
        lines.append("")
        lines.append("**Generated Files:**")
        for img in result.images:
            lines.append(
                f"  - `{img.filename}` ({img.resolution[0]}×{img.resolution[1]} px) "
                f"| PNG: `{img.png_filename}`"
            )

    return "\n".join(lines)


# Tool 2: Get model information

async def get_model_info(
    ctx: RunContext[AgentDependencies],
    model_name: Optional[str] = None,
) -> str:
    """Return information about available DeepLenseSim model configurations.

    Use this tool when the user asks about differences between models,
    what configurations are available, or needs help choosing.

    Args:
        ctx: Runtime context.
        model_name: Optional specific model to query ("Model_I" or "Model_II").
            If omitted, returns info about all available models.

    Returns:
        Formatted string with model details.
    """
    models_info = {
        "Model_I": {
            "description": (
                "Closest to the original DeepLense papers. "
                "Uses a simple Gaussian PSF with manual noise generation."
            ),
            "resolution": f"{MODEL_I_RESOLUTION[0]}×{MODEL_I_RESOLUTION[1]} px",
            "pixel_scale": f"{MODEL_I_PIXEL_SCALE} arcsec/px",
            "instrument": MODEL_I_INSTRUMENT,
            "source_galaxy": "Sersic ellipse (amplitude-based)",
            "simulation_method": "simple_sim() — manual ImageModel + Poisson/background noise",
            "dm_classes": "no_sub, axion (vortex), CDM (point-mass subhalos)",
        },
        "Model_II": {
            "description": (
                "Approximates a Euclid survey. Uses lenstronomy's SimAPI "
                "with pre-defined Euclid observation characteristics."
            ),
            "resolution": f"{MODEL_II_RESOLUTION[0]}×{MODEL_II_RESOLUTION[1]} px",
            "pixel_scale": f"{MODEL_II_PIXEL_SCALE} arcsec/px",
            "instrument": MODEL_II_INSTRUMENT,
            "source_galaxy": "Sersic ellipse (magnitude-based)",
            "simulation_method": "simple_sim_2() — SimAPI with Euclid band config",
            "dm_classes": "no_sub, axion (vortex), CDM (point-mass subhalos)",
        },
    }

    if model_name and model_name in models_info:
        info = models_info[model_name]
        lines = [f"## {model_name}", ""]
        for key, val in info.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {val}")
        return "\n".join(lines)

    # Return all
    lines = ["## Available Model Configurations", ""]
    for name, info in models_info.items():
        lines.append(f"### {name}")
        for key, val in info.items():
            lines.append(f"- **{key.replace('_', ' ').title()}**: {val}")
        lines.append("")
    return "\n".join(lines)


# Tool 3: Validate parameters (pre-flight check)

async def validate_parameters(
    ctx: RunContext[AgentDependencies],
    model_config_type: str,
    substructure_type: str,
    num_images: int = 1,
    halo_mass: float = 1e12,
    z_halo: float = 0.5,
    z_source: float = 1.0,
    axion_mass: Optional[float] = None,
    vortex_mass: float = 3e10,
) -> str:
    """Validate simulation parameters before execution.

    Use this tool to check parameters for physics consistency and
    warn about edge cases BEFORE calling generate_lensing_images.

    Args:
        ctx: Runtime context.
        model_config_type: "Model_I" or "Model_II".
        substructure_type: "no_sub", "axion", or "cdm".
        num_images: Number of images (1-100).
        halo_mass: Halo mass in solar masses.
        z_halo: Halo redshift.
        z_source: Source redshift.
        axion_mass: Axion mass in eV (for axion sims).
        vortex_mass: Vortex mass in solar masses.

    Returns:
        Validation report with any warnings or errors.
    """
    warnings: list[str] = []
    errors: list[str] = []

    # -- Substructure checks --
    try:
        sub = SubstructureType(substructure_type)
    except ValueError:
        errors.append(
            f"Invalid substructure_type '{substructure_type}'. "
            f"Must be one of: {[e.value for e in SubstructureType]}"
        )
        sub = None

    try:
        model = ModelConfig(model_config_type)
    except ValueError:
        errors.append(
            f"Invalid model_config_type '{model_config_type}'. "
            f"Must be one of: {[e.value for e in ModelConfig]}"
        )
        model = None

    # -- Physics checks --
    if z_source <= z_halo:
        errors.append(
            f"Source redshift ({z_source}) must be > halo redshift ({z_halo})."
        )

    if halo_mass < 1e8:
        warnings.append(
            f"Halo mass {halo_mass:.2e} M☉ is unusually low. "
            "Typical strong lensing halos are ≥ 1e10 M☉."
        )
    elif halo_mass > 1e15:
        warnings.append(
            f"Halo mass {halo_mass:.2e} M☉ is extremely high (galaxy cluster scale). "
            "Simulation may produce unrealistic results."
        )

    if sub == SubstructureType.AXION:
        if axion_mass is not None:
            if axion_mass < 1e-26 or axion_mass > 1e-18:
                warnings.append(
                    f"Axion mass {axion_mass:.2e} eV is outside the typical range "
                    "[1e-24, 1e-22]. Results may be physically unusual."
                )
        else:
            warnings.append(
                "No axion_mass specified — a random value in [1e-24, 1e-22] eV "
                "will be drawn automatically."
            )

    elif sub is not None and sub != SubstructureType.AXION and axion_mass is not None:
        warnings.append(
            "axion_mass was specified but substructure_type is not 'axion'. "
            "The axion_mass parameter will be ignored."
        )

    if num_images > 50:
        warnings.append(
            f"Generating {num_images} images may take significant time "
            f"(~{num_images * 3}–{num_images * 10}s depending on hardware)."
        )

    # -- Build report --
    lines = ["## Parameter Validation Report", ""]
    if errors:
        lines.append("### ❌ Errors (must fix)")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")
    if warnings:
        lines.append("### ⚠️ Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    if not errors and not warnings:
        lines.append("✅ All parameters are valid. Ready to generate!")

    if not errors:
        lines.append("✅ Parameters pass validation. You may proceed with generation.")
    else:
        lines.append("❌ Please fix the errors above before generating.")

    return "\n".join(lines)
