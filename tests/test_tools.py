import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.config import AgentDependencies
from src.models import (
    SimulationParameters,
    SimulationResult,
    ImageMetadata,
    SubstructureType,
    ModelConfig,
)
from src.tools import generate_lensing_images, get_model_info, validate_parameters


# Helpers
def make_ctx(output_dir="/tmp/test_output"):
    """Create a mock RunContext with AgentDependencies."""
    ctx = MagicMock()
    ctx.deps = AgentDependencies(output_dir=output_dir)
    return ctx


def make_mock_result(n_images=1):
    """Create a mock SimulationResult."""
    params = SimulationParameters(
        model_config_type=ModelConfig.MODEL_I,
        substructure_type=SubstructureType.NO_SUB,
    )
    images = [
        ImageMetadata(
            image_index=i,
            filename=f"no_sub_Model_I_{i:04d}.npy",
            png_filename=f"no_sub_Model_I_{i:04d}.png",
            substructure_type=SubstructureType.NO_SUB,
            model_config_type=ModelConfig.MODEL_I,
            resolution=(150, 150),
            halo_mass=1e12,
            z_halo=0.5,
            z_source=1.0,
            pixel_scale_arcsec=0.05,
            instrument="Gaussian PSF",
        )
        for i in range(n_images)
    ]
    return SimulationResult(
        success=True,
        message=f"Generated {n_images} images",
        parameters_used=params,
        images=images,
        total_images_generated=n_images,
        output_directory="/tmp/test_output",
        generation_time_seconds=1.5,
    )


# get_model_info
class TestGetModelInfo:
    @pytest.mark.asyncio
    async def test_all_models(self):
        ctx = make_ctx()
        result = await get_model_info(ctx)
        assert "Model_I" in result
        assert "Model_II" in result
        assert "150x150" in result
        assert "64x64" in result

    @pytest.mark.asyncio
    async def test_specific_model(self):
        ctx = make_ctx()
        result = await get_model_info(ctx, model_name="Model_I")
        assert "Model_I" in result
        assert "150x150" in result

    @pytest.mark.asyncio
    async def test_unknown_model(self):
        ctx = make_ctx()
        result = await get_model_info(ctx, model_name="Model_X")
        # Should return all models when queried name isn't found
        assert "Model_I" in result
        assert "Model_II" in result



# validate_parameters
class TestValidateParameters:
    @pytest.mark.asyncio
    async def test_valid_params(self):
        ctx = make_ctx()
        result = await validate_parameters(
            ctx,
            model_config_type="Model_I",
            substructure_type="no_sub",
        )
        assert "✅" in result

    @pytest.mark.asyncio
    async def test_invalid_redshift(self):
        ctx = make_ctx()
        result = await validate_parameters(
            ctx,
            model_config_type="Model_I",
            substructure_type="no_sub",
            z_halo=1.5,
            z_source=0.5,
        )
        assert "❌" in result
        assert "Source redshift" in result

    @pytest.mark.asyncio
    async def test_invalid_substructure(self):
        ctx = make_ctx()
        result = await validate_parameters(
            ctx,
            model_config_type="Model_I",
            substructure_type="wdm",
        )
        assert "❌" in result

    @pytest.mark.asyncio
    async def test_low_halo_mass_warning(self):
        ctx = make_ctx()
        result = await validate_parameters(
            ctx,
            model_config_type="Model_I",
            substructure_type="no_sub",
            halo_mass=1e5,
        )
        assert "⚠️" in result
        assert "unusually low" in result

    @pytest.mark.asyncio
    async def test_high_halo_mass_warning(self):
        ctx = make_ctx()
        result = await validate_parameters(
            ctx,
            model_config_type="Model_I",
            substructure_type="no_sub",
            halo_mass=1e16,
        )
        assert "⚠️" in result

    @pytest.mark.asyncio
    async def test_axion_without_mass(self):
        ctx = make_ctx()
        result = await validate_parameters(
            ctx,
            model_config_type="Model_I",
            substructure_type="axion",
        )
        assert "⚠️" in result
        assert "random value" in result

    @pytest.mark.asyncio
    async def test_axion_mass_on_non_axion(self):
        ctx = make_ctx()
        result = await validate_parameters(
            ctx,
            model_config_type="Model_I",
            substructure_type="cdm",
            axion_mass=1e-23,
        )
        assert "⚠️" in result
        assert "ignored" in result

    @pytest.mark.asyncio
    async def test_large_batch_warning(self):
        ctx = make_ctx()
        result = await validate_parameters(
            ctx,
            model_config_type="Model_I",
            substructure_type="no_sub",
            num_images=60,
        )
        assert "⚠️" in result
        assert "significant time" in result

# generate_lensing_images (with mocked engine)
class TestGenerateLensingImages:
    @pytest.mark.asyncio
    @patch("src.tools.SimulationEngine.generate_batch")
    async def test_successful_generation(self, mock_batch):
        mock_batch.return_value = make_mock_result(3)
        ctx = make_ctx()
        result = await generate_lensing_images(
            ctx,
            model_config_type="Model_I",
            substructure_type="no_sub",
            num_images=3,
        )
        assert "Completed" in result
        assert "3" in result
        mock_batch.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.tools.SimulationEngine.generate_batch")
    async def test_axion_generation(self, mock_batch):
        mock_batch.return_value = make_mock_result(1)
        ctx = make_ctx()
        result = await generate_lensing_images(
            ctx,
            model_config_type="Model_II",
            substructure_type="axion",
            axion_mass=1e-23,
        )
        assert "Completed" in result
        mock_batch.assert_called_once()


# suggest_parameters
class TestSuggestParameters:
    @pytest.mark.asyncio
    async def test_no_args_returns_menu(self):
        """Calling with no arguments should return the full scenario menu."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx)
        assert "galaxy_scale" in result
        assert "cluster_scale" in result
        assert "high_redshift" in result
        assert "low_mass_axion" in result
        assert "high_mass_axion" in result
        assert "statistical_study" in result

    @pytest.mark.asyncio
    async def test_galaxy_scale(self):
        """galaxy_scale scenario should return standard galaxy-scale values."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx, scenario="galaxy_scale")
        assert "Galaxy-Scale" in result
        assert "1e12 M_sun" in result
        assert "0.5" in result  # z_halo

    @pytest.mark.asyncio
    async def test_cluster_scale_high_mass(self):
        """cluster_scale scenario should suggest halo mass >= 1e14."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx, scenario="cluster_scale")
        assert "Cluster" in result
        assert "5e14 M_sun" in result

    @pytest.mark.asyncio
    async def test_high_redshift_source(self):
        """high_redshift scenario should suggest z_source >= 2."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx, scenario="high_redshift")
        assert "3.0" in result  # z_source

    @pytest.mark.asyncio
    async def test_low_mass_axion(self):
        """low_mass_axion should suggest 1e-24 eV axion mass."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx, scenario="low_mass_axion")
        assert "1e-24 eV" in result

    @pytest.mark.asyncio
    async def test_high_mass_axion(self):
        """high_mass_axion should suggest 1e-22 eV axion mass."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx, scenario="high_mass_axion")
        assert "1e-22 eV" in result

    @pytest.mark.asyncio
    async def test_statistical_study_num_images(self):
        """statistical_study should suggest num_images >= 20."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx, scenario="statistical_study")
        # The suggestion is "50"
        assert "50" in result

    @pytest.mark.asyncio
    async def test_unknown_scenario_fallback(self):
        """Unknown scenario should return available scenarios without crashing."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx, scenario="wdm_dark_matter")
        assert "Unknown Scenario" in result
        assert "galaxy_scale" in result

    @pytest.mark.asyncio
    async def test_scenario_case_insensitive(self):
        """Scenario matching should be case-insensitive."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(ctx, scenario="Galaxy_Scale")
        assert "Galaxy-Scale" in result

    @pytest.mark.asyncio
    async def test_model_already_chosen_omits_model_suggestion(self):
        """If model_config_type is already chosen, model row should be absent."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(
            ctx, scenario="galaxy_scale", model_config_type="Model_I"
        )
        # model_config_type row removed; other params still present
        assert "1e12 M_sun" in result
        # The model_config_type row should not appear in the table
        assert "| `model_config_type`" not in result

    @pytest.mark.asyncio
    async def test_substructure_note_appears(self):
        """Substructure note should reference the provided substructure type."""
        ctx = make_ctx()
        from src.tools import suggest_parameters
        result = await suggest_parameters(
            ctx, scenario="galaxy_scale", substructure_type="cdm"
        )
        assert "cdm" in result

