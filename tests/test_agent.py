import os
import pytest
from unittest.mock import MagicMock, patch

from src.agent import create_lensing_agent, SYSTEM_PROMPT
from src.config import AgentDependencies


class TestAgentCreation:
    """Test agent factory and configuration."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-for-ci"})
    def test_create_agent(self):
        """Agent should be created with correct configuration."""
        agent = create_lensing_agent()
        assert agent is not None

    def test_system_prompt_content(self):
        """System prompt should contain key instructions."""
        assert "DeepLense Simulation Assistant" in SYSTEM_PROMPT
        assert "Model_I" in SYSTEM_PROMPT
        assert "Model_II" in SYSTEM_PROMPT
        assert "no_sub" in SYSTEM_PROMPT
        assert "axion" in SYSTEM_PROMPT
        assert "cdm" in SYSTEM_PROMPT
        assert "Human-in-the-Loop" in SYSTEM_PROMPT
        assert "Phase 1" in SYSTEM_PROMPT
        assert "Phase 2" in SYSTEM_PROMPT

    def test_system_prompt_has_parameter_table(self):
        """System prompt should include parameter defaults table."""
        assert "halo_mass" in SYSTEM_PROMPT
        assert "z_halo" in SYSTEM_PROMPT
        assert "z_source" in SYSTEM_PROMPT
        assert "axion_mass" in SYSTEM_PROMPT

    def test_system_prompt_mandates_confirmation(self):
        """System prompt should mandate user confirmation before generation."""
        assert "confirm" in SYSTEM_PROMPT.lower()
        assert "Shall I proceed" in SYSTEM_PROMPT


class TestAgentDependencies:
    """Test the dependency injection configuration."""

    def test_default_output_dir(self):
        deps = AgentDependencies()
        assert "outputs" in deps.output_dir

    def test_custom_output_dir(self):
        deps = AgentDependencies(output_dir="/tmp/custom_output")
        assert deps.output_dir == "/tmp/custom_output"

    def test_ensure_output_dir(self, tmp_path):
        deps = AgentDependencies(output_dir=str(tmp_path / "new_dir"))
        path = deps.ensure_output_dir()
        assert path.exists()
        assert path.is_dir()
