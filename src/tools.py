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
    MODEL_III_RESOLUTION,
    MODEL_III_PIXEL_SCALE,
    MODEL_III_INSTRUMENT,
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
        model_config_type: Model configuration:
            "Model_I"   (150x150 px, Gaussian PSF),
            "Model_II"  (64x64 px, Euclid VIS), or
            "Model_III" (HST instrument, magnitude-based source, old CDM sampler).
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
            "resolution": f"{MODEL_I_RESOLUTION[0]}x{MODEL_I_RESOLUTION[1]} px",
            "pixel_scale": f"{MODEL_I_PIXEL_SCALE} arcsec/px",
            "instrument": MODEL_I_INSTRUMENT,
            "source_galaxy": "Sersic ellipse (amplitude-based)",
            "simulation_method": "simple_sim() -- manual ImageModel + Poisson/background noise",
            "dm_classes": "no_sub, axion (vortex), CDM (point-mass subhalos)",
        },
        "Model_II": {
            "description": (
                "Approximates a Euclid survey. Uses lenstronomy's SimAPI "
                "with pre-defined Euclid observation characteristics."
            ),
            "resolution": f"{MODEL_II_RESOLUTION[0]}x{MODEL_II_RESOLUTION[1]} px",
            "pixel_scale": f"{MODEL_II_PIXEL_SCALE} arcsec/px",
            "instrument": MODEL_II_INSTRUMENT,
            "source_galaxy": "Sersic ellipse (magnitude-based)",
            "simulation_method": "simple_sim_2() -- SimAPI with Euclid band config",
            "dm_classes": "no_sub, axion (vortex), CDM (point-mass subhalos)",
        },
        "Model_III": {
            "description": (
                "Same 150x150 Gaussian PSF setup as Model_I, but uses magnitude-based source light "
                "(like Model_II) and the older CDM subhalo sampler (make_old_cdm) instead of the "
                "pyHalo-based CDM sampler. The HST instrument path is not yet implemented in DeepLenseSim."
            ),
            "resolution": f"{MODEL_III_RESOLUTION[0]}x{MODEL_III_RESOLUTION[1]} px",
            "pixel_scale": f"{MODEL_III_PIXEL_SCALE} arcsec/px",
            "instrument": MODEL_III_INSTRUMENT,
            "source_galaxy": "Sersic ellipse (magnitude-based)",
            "simulation_method": "simple_sim() -- Gaussian PSF, magnitude-based source, old CDM sampler",
            "dm_classes": "no_sub, axion (vortex), CDM (old point-mass subhalos via make_old_cdm)",
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
        model_config_type: "Model_I", "Model_II", or "Model_III".
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


# Tool 4: Suggest parameters


# Scenario registry: each entry has suggested values and per-field justifications.
_SCENARIO_REGISTRY: dict[str, dict] = {
    "galaxy_scale": {
        "label": "Galaxy-Scale Lens",
        "description": (
            "Typical galaxy-scale strong gravitational lens. Reproduces the "
            "original DeepLense training distribution most closely. Best "
            "starting point for most experiments."
        ),
        "substructure_note": "Works with all three substructure types.",
        "suggestions": {
            "model_config_type": ("Model_I", "150x150 px matches original DeepLense angular resolution and PSF"),
            "halo_mass":         ("1e12 M_sun", "Milky Way-scale halo, peak strong-lensing cross-section"),
            "z_halo":            ("0.5", "Canonical lens redshift for ground-based optical surveys"),
            "z_source":          ("1.5", "Bright Lyman-break galaxy / LRG source population peak"),
            "num_images":        ("5", "Good starting point for visual inspection"),
            "random_seed":       ("None", "Leave unset for varied independent realizations"),
        },
    },
    "cluster_scale": {
        "label": "Galaxy-Cluster-Scale Lens",
        "description": (
            "Galaxy cluster acting as the primary lens. Produces dramatic "
            "arcs and multiple images. Halo masses are 2-3 orders of "
            "magnitude above galaxy-scale lenses."
        ),
        "substructure_note": (
            "CDM subhalos are most physically motivated at cluster scales. "
            "Axion substructure at this mass scale requires very light axion masses."
        ),
        "suggestions": {
            "model_config_type": ("Model_I", "Higher angular extent of cluster arcs benefits from 150x150 px"),
            "halo_mass":         ("5e14 M_sun", "Massive cluster scale; strong lensing requires M > 1e14 M_sun"),
            "z_halo":            ("0.3", "Low-redshift clusters dominate flux-limited cluster samples"),
            "z_source":          ("2.0", "High-redshift background galaxy behind nearby cluster"),
            "num_images":        ("3", "Cluster sims are slower; start with a small batch"),
            "random_seed":       ("42", "Set a seed to reproduce the specific cluster realization"),
        },
    },
    "high_redshift": {
        "label": "High-Redshift Source",
        "description": (
            "Source galaxy at high redshift (z > 2) behind a moderate-redshift "
            "lens. Probes earlier cosmic epochs and produces stronger lensing "
            "magnification due to the long source-lens angular diameter distance."
        ),
        "substructure_note": "All substructure types are valid.",
        "suggestions": {
            "model_config_type": ("Model_II", "Euclid instrument is designed for high-z survey science"),
            "halo_mass":         ("2e12 M_sun", "Slightly above MW-mass to ensure strong lensing at this geometry"),
            "z_halo":            ("0.7", "Intermediate lens redshift maximises lensing efficiency for z_source ~ 3"),
            "z_source":          ("3.0", "Lyman-break galaxy population at cosmic noon"),
            "num_images":        ("5", "Balanced starting batch"),
            "random_seed":       ("None", "Unset for varied source morphologies"),
        },
    },
    "low_mass_axion": {
        "label": "Ultra-Light Axion (Fuzzy DM)",
        "description": (
            "Ultra-light fuzzy dark matter axion with a very low particle mass. "
            "The de Broglie wavelength is large (~kpc), producing extended, "
            "coherent vortex structures clearly distinguishable from CDM subhalos."
        ),
        "substructure_note": "Requires substructure_type = axion.",
        "suggestions": {
            "model_config_type": ("Model_I", "Higher resolution better resolves extended vortex structures"),
            "halo_mass":         ("1e12 M_sun", "Galaxy-scale halo"),
            "z_halo":            ("0.5", "Standard lens redshift"),
            "z_source":          ("1.5", "Standard source redshift"),
            "axion_mass":        ("1e-24 eV", "Lower end of fuzzy DM window; largest vortex cores (~kpc)"),
            "vortex_mass":       ("3e10 M_sun", "Default total vortex mass"),
            "num_images":        ("10", "Generate several to sample vortex pattern variety"),
            "random_seed":       ("None", "Unset for varied vortex configurations"),
        },
    },
    "high_mass_axion": {
        "label": "Heavy Axion (Near-CDM Regime)",
        "description": (
            "Heavier axion particle mass. The de Broglie wavelength is short "
            "(sub-pc), so vortex cores are compact and the convergence map begins "
            "to resemble CDM point-mass subhalos. Useful for studying the "
            "axion-CDM transition regime."
        ),
        "substructure_note": "Requires substructure_type = axion.",
        "suggestions": {
            "model_config_type": ("Model_I", "150x150 px needed to resolve compact vortex cores"),
            "halo_mass":         ("1e12 M_sun", "Galaxy-scale halo"),
            "z_halo":            ("0.5", "Standard lens redshift"),
            "z_source":          ("1.5", "Standard source redshift"),
            "axion_mass":        ("1e-22 eV", "Upper end of fuzzy DM window; smallest vortex cores"),
            "vortex_mass":       ("3e10 M_sun", "Default total vortex mass"),
            "num_images":        ("10", "Generate several to sample compact vortex variety"),
            "random_seed":       ("None", "Unset for varied realizations"),
        },
    },
    "statistical_study": {
        "label": "Statistical Ensemble Study",
        "description": (
            "Large batch of images for ML training, statistical analysis, or "
            "sensitivity studies. Designed to generate enough samples for "
            "meaningful classification or power-spectrum measurements."
        ),
        "substructure_note": (
            "Run separate batches for each substructure type to build a balanced "
            "training set across no_sub, axion, and cdm classes."
        ),
        "suggestions": {
            "model_config_type": ("Model_I", "Matches the original DeepLense dataset format for compatibility"),
            "halo_mass":         ("1e12 M_sun", "Fix mass for a controlled comparison"),
            "z_halo":            ("0.5", "Fix lens redshift for a controlled comparison"),
            "z_source":          ("1.5", "Fix source redshift for a controlled comparison"),
            "num_images":        ("50", "Minimum for basic statistical analysis; use 100 for ML training"),
            "random_seed":       ("0", "Fix seed for the first batch; increment per-batch for variety"),
        },
    },
}


async def suggest_parameters(
    ctx: RunContext[AgentDependencies],
    scenario: Optional[str] = None,
    substructure_type: Optional[str] = None,
    model_config_type: Optional[str] = None,
) -> str:
    """Suggest physically motivated simulation parameters for a given scientific scenario.

    Use this tool when the user asks for recommendations, is unsure what values to
    use, or wants to explore a specific astrophysical configuration. This tool
    contains physics-based heuristics and does NOT call any external service.

    Args:
        ctx: Runtime context.
        scenario: Scientific scenario keyword. One of:
            "galaxy_scale"      - Typical galaxy-scale strong lens (default DeepLense setup)
            "cluster_scale"     - Galaxy cluster as lens (high halo mass)
            "high_redshift"     - High-redshift source behind moderate-redshift lens
            "low_mass_axion"    - Ultra-light fuzzy DM axion (large vortex cores)
            "high_mass_axion"   - Heavy axion near the CDM regime (compact cores)
            "statistical_study" - Large batch for ML training or statistical analysis
            If omitted, returns the full scenario menu so the user can choose.
        substructure_type: Already-chosen substructure type, if known ("no_sub",
            "axion", "cdm"). Used to append substructure-specific advice.
        model_config_type: Already-chosen model config, if known ("Model_I",
            "Model_II"). If provided, the model suggestion is omitted from output.

    Returns:
        Formatted Markdown suggestion report with parameter values and justifications.
    """
    # No scenario specified: return the full menu
    if scenario is None:
        lines = [
            "## Available Simulation Scenarios",
            "",
            "Choose a scenario to get tailored parameter suggestions:",
            "",
        ]
        for key, info in _SCENARIO_REGISTRY.items():
            lines.append(f"- **`{key}`** - {info['label']}: {info['description'].split('.')[0]}.")
        lines.append("")
        lines.append(
            "Ask me to suggest parameters for any of the above, "
            "e.g. \"suggest parameters for cluster_scale\"."
        )
        return "\n".join(lines)

    # Normalize key (case-insensitive, spaces or hyphens to underscores)
    scenario_key = scenario.strip().lower().replace(" ", "_").replace("-", "_")
    if scenario_key not in _SCENARIO_REGISTRY:
        close_matches = [k for k in _SCENARIO_REGISTRY if scenario_key in k or k in scenario_key]
        lines = [f"## Unknown Scenario: `{scenario}`", "", "Available scenarios:"]
        for key, info in _SCENARIO_REGISTRY.items():
            lines.append(f"- `{key}` - {info['label']}")
        if close_matches:
            lines.append(f"\nDid you mean: `{close_matches[0]}`?")
        return "\n".join(lines)

    info = _SCENARIO_REGISTRY[scenario_key]
    suggestions = dict(info["suggestions"])  # shallow copy

    # Drop model suggestion if user already chose one
    if model_config_type and "model_config_type" in suggestions:
        del suggestions["model_config_type"]

    # Build report
    lines = [
        f"## Suggested Parameters: {info['label']}",
        "",
        f"**Scenario:** {info['description']}",
        "",
    ]

    if substructure_type:
        lines.append(f"**Substructure note ({substructure_type}):** {info['substructure_note']}")
    else:
        lines.append(f"**Substructure note:** {info['substructure_note']}")
    lines.append("")

    # Parameter table
    lines.append("| Parameter | Suggested Value | Justification |")
    lines.append("|-----------|----------------|---------------|")
    for param, (value, justification) in suggestions.items():
        lines.append(f"| `{param}` | {value} | {justification} |")
    lines.append("")

    # Ready-to-use copy-paste line
    ready_parts = [f"{param}={value}" for param, (value, _) in suggestions.items()]
    lines.append("**Ready to use:**")
    lines.append("`" + ", ".join(ready_parts) + "`")

    return "\n".join(lines)
