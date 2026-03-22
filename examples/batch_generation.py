from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SimulationParameters, SubstructureType, ModelConfig
from src.simulation_engine import SimulationEngine


async def batch_generate():
    """Generate images for all substructure types using both Model_I and Model_II."""

    output_base = Path("./outputs/batch")

    # Define all combinations to generate
    configs = [
        # (model, substructure, num_images, extra_kwargs)
        (ModelConfig.MODEL_I, SubstructureType.NO_SUB, 2, {}),
        (ModelConfig.MODEL_I, SubstructureType.AXION, 2, {"axion_mass": 1e-23}),
        (ModelConfig.MODEL_I, SubstructureType.CDM, 2, {}),
        (ModelConfig.MODEL_II, SubstructureType.NO_SUB, 2, {}),
        (ModelConfig.MODEL_II, SubstructureType.AXION, 2, {"axion_mass": 5e-23}),
        (ModelConfig.MODEL_II, SubstructureType.CDM, 2, {}),
    ]

    for model, sub, n, extra in configs:
        print(f"\n{'='*60}")
        print(f"Generating: {model.value} / {sub.value} / {n} images")
        print(f"{'='*60}")

        params = SimulationParameters(
            model_config_type=model,
            substructure_type=sub,
            num_images=n,
            random_seed=42,
            **extra,
        )

        output_dir = output_base / model.value / sub.value
        result = await SimulationEngine.generate_batch(params, output_dir)

        print(f"Status: {'✓' if result.success else '✗'}")
        print(f"Message: {result.message}")
        print(f"Images: {result.total_images_generated}")
        print(f"Time: {result.generation_time_seconds}s")
        print(f"Output: {result.output_directory}")
        for img in result.images:
            print(f"- {img.filename} ({img.resolution[0]}x{img.resolution[1]})")

    print("Batch generation complete!")


if __name__ == "__main__":
    asyncio.run(batch_generate())
