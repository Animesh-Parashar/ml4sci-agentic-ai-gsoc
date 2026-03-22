import pytest
import numpy as np

from src.models import (
    SimulationParameters,
    SimulationResult,
    ImageMetadata,
    SubstructureType,
    ModelConfig,
)



# SimulationParameters
class TestSimulationParameters:
    """Test SimulationParameters validation and physics constraints."""

    def test_valid_no_sub_params(self):
        """Standard no_sub parameters should pass validation."""
        params = SimulationParameters(
            model_config_type=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.NO_SUB,
            num_images=5,
        )
        assert params.num_images == 5
        assert params.halo_mass == 1e12
        assert params.z_halo == 0.5
        assert params.z_source == 1.0
        assert params.axion_mass is None

    def test_valid_axion_params_explicit(self):
        """Axion with explicit mass should retain the given mass."""
        params = SimulationParameters(
            model_config_type=ModelConfig.MODEL_II,
            substructure_type=SubstructureType.AXION,
            axion_mass=1e-23,
        )
        assert params.axion_mass == 1e-23

    def test_axion_auto_mass(self):
        """Axion without explicit mass should auto-generate one."""
        params = SimulationParameters(
            model_config_type=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.AXION,
        )
        assert params.axion_mass is not None
        assert 1e-24 <= params.axion_mass <= 1e-22

    def test_cdm_params(self):
        """CDM substructure should have no axion_mass."""
        params = SimulationParameters(
            model_config_type=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.CDM,
        )
        assert params.axion_mass is None

    def test_invalid_redshift_order(self):
        """Source redshift <= halo redshift should raise ValueError."""
        with pytest.raises(ValueError, match="Source redshift"):
            SimulationParameters(
                model_config_type=ModelConfig.MODEL_I,
                substructure_type=SubstructureType.NO_SUB,
                z_halo=1.0,
                z_source=0.5,
            )

    def test_equal_redshifts(self):
        """Equal redshifts should also raise ValueError."""
        with pytest.raises(ValueError, match="Source redshift"):
            SimulationParameters(
                model_config_type=ModelConfig.MODEL_I,
                substructure_type=SubstructureType.NO_SUB,
                z_halo=0.5,
                z_source=0.5,
            )

    def test_num_images_bounds(self):
        """num_images must be between 1 and 100."""
        with pytest.raises(Exception):
            SimulationParameters(
                model_config_type=ModelConfig.MODEL_I,
                substructure_type=SubstructureType.NO_SUB,
                num_images=0,
            )
        with pytest.raises(Exception):
            SimulationParameters(
                model_config_type=ModelConfig.MODEL_I,
                substructure_type=SubstructureType.NO_SUB,
                num_images=101,
            )

    def test_invalid_model_config(self):
        """Invalid model config string should raise."""
        with pytest.raises(ValueError):
            SimulationParameters(
                model_config_type="Model_X",
                substructure_type=SubstructureType.NO_SUB,
            )

    def test_invalid_substructure(self):
        """Invalid substructure string should raise."""
        with pytest.raises(ValueError):
            SimulationParameters(
                model_config_type=ModelConfig.MODEL_I,
                substructure_type="wdm",
            )

    def test_random_seed(self):
        """Random seed should be stored correctly."""
        params = SimulationParameters(
            model_config_type=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.NO_SUB,
            random_seed=42,
        )
        assert params.random_seed == 42

    def test_custom_halo_mass(self):
        """Custom halo mass should be preserved."""
        params = SimulationParameters(
            model_config_type=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.CDM,
            halo_mass=5e11,
        )
        assert params.halo_mass == 5e11


# ImageMetadata
class TestImageMetadata:
    def test_create_metadata(self):
        meta = ImageMetadata(
            image_index=0,
            filename="test.npy",
            substructure_type=SubstructureType.NO_SUB,
            model_config_type=ModelConfig.MODEL_I,
            resolution=(150, 150),
            halo_mass=1e12,
            z_halo=0.5,
            z_source=1.0,
            pixel_scale_arcsec=0.05,
            instrument="Gaussian PSF",
        )
        assert meta.resolution == (150, 150)
        assert meta.png_filename is None
        assert meta.axion_mass is None

# SimulationResult
class TestSimulationResult:
    def test_successful_result(self):
        params = SimulationParameters(
            model_config_type=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.NO_SUB,
        )
        result = SimulationResult(
            success=True,
            message="OK",
            parameters_used=params,
            total_images_generated=1,
            output_directory="/tmp/test",
        )
        assert result.success
        assert result.images == []
        assert result.generation_time_seconds == 0.0

    def test_failed_result(self):
        params = SimulationParameters(
            model_config_type=ModelConfig.MODEL_I,
            substructure_type=SubstructureType.NO_SUB,
        )
        result = SimulationResult(
            success=False,
            message="Error occurred",
            parameters_used=params,
            total_images_generated=0,
            output_directory="/tmp/test",
        )
        assert not result.success


# Enum Tests
class TestEnums:
    def test_substructure_values(self):
        assert SubstructureType.NO_SUB.value == "no_sub"
        assert SubstructureType.AXION.value == "axion"
        assert SubstructureType.CDM.value == "cdm"

    def test_model_config_values(self):
        assert ModelConfig.MODEL_I.value == "Model_I"
        assert ModelConfig.MODEL_II.value == "Model_II"

    def test_substructure_from_string(self):
        assert SubstructureType("no_sub") == SubstructureType.NO_SUB
        assert SubstructureType("axion") == SubstructureType.AXION
        assert SubstructureType("cdm") == SubstructureType.CDM
