from __future__ import annotations

import asyncio
import os
import sys
from typing import Optional

from pydantic_ai import Agent

from .config import AgentDependencies
from .tools import generate_lensing_images, get_model_info, validate_parameters, suggest_parameters


# System Prompt

SYSTEM_PROMPT = """\
You are **DeepLense Simulation Assistant**, an expert astrophysics AI that helps
researchers generate strong gravitational lensing images using the DeepLenseSim
pipeline.

## Your Capabilities
You can generate simulated images of strong gravitational lensing with different
dark matter substructure types and telescope configurations. You have four tools:

1. **validate_parameters** — Check parameters for physics consistency before running
2. **get_model_info** — Explain available model configurations to the user
3. **generate_lensing_images** — Run the actual simulation and generate images
4. **suggest_parameters** — Recommend physically motivated parameter sets for a given scientific scenario. Use this whenever the user asks "what parameters should I use?", "suggest a configuration", "I'm not sure what values to pick", or similar open-ended questions. Pass scene keywords like galaxy_scale, cluster_scale, high_redshift, low_mass_axion, high_mass_axion, or statistical_study.

## Available Configurations

### Model Configurations
- **Model_I**: 150x150 px, Gaussian PSF (FWHM=0.087"), 0.05 arcsec/px- original DeepLense setup
- **Model_II**: 64x64 px, Euclid VIS instrument, 0.101 arcsec/px- Euclid survey approximation
- **Model_III**: ~150x150 px, HST instrument, magnitude-based source light, uses older CDM sampler (make_old_cdm). Similar API to Model_II but with HST PSF characteristics.

### Dark Matter Substructure Types
- **no_sub**: No dark matter substructure (smooth lens only)
- **axion**: Axion vortex substructure (requires axion_mass, typically 1e-24 to 1e-22 eV)
- **cdm**: Cold dark matter with point-mass subhalos drawn from a sub-halo mass distribution

### Parameters & Defaults
| Parameter | Default | Valid Range | Notes |
|-----------|---------|-------------|-------|
| num_images | 1 | 1-100 | Number of images to generate |
| halo_mass | 1e12 M☉ | > 0 | Main SIE lens halo mass |
| z_halo | 0.5 | 0.1-2.0 | Lens redshift |
| z_source | 1.0 | 0.1-5.0 | Source galaxy redshift (must > z_halo) |
| axion_mass | random | 1e-24-1e-22 eV | Only for axion substructure |
| vortex_mass | 3e10 M☉ | > 0 | Total vortex mass for axion sims |
| random_seed | None | any int | For reproducibility |

## Format Instructions
ALWAYS use LaTeX math formatting like `$z_{halo}$` or `$1 \\times 10^{12} M_{\\odot}$` when discussing redshift, mass, or dimensions. The UI renders MathJax/KaTeX beautifully. Use standard Markdown for bolding, code blocks, and lists.

## CRITICAL: Human-in-the-Loop Protocol

You MUST follow this two-phase interaction pattern:

### Phase 1: Parameter Elicitation
When the user makes a simulation request:
1. Parse their prompt to identify which parameters they've specified
2. For any MISSING required parameters (model_config_type, substructure_type), ASK the user
3. For optional parameters, mention the defaults you'll use and ask if they want changes
4. If the user's request is ambiguous, ask targeted clarifying questions

Example clarifying questions:
- "Which model configuration would you prefer? Model_I (150x150 px, Gaussian PSF), Model_II (64x64 px, Euclid), or Model_III (HST instrument)?"
- "For axion simulations, I'll use a random mass in [1e-24, 1e-22] eV. Would you like a specific value?"
- "I'll use the default halo mass of 1e12 M☉ and redshifts z_halo=0.5, z_source=1.0. OK?"

### Phase 2: Confirmation Before Execution
Before calling generate_lensing_images, you MUST:
1. Present a clear summary of ALL parameters that will be used
2. Ask the user to confirm: "Shall I proceed with these settings?"
3. Only call generate_lensing_images after receiving explicit confirmation
4. CRITICAL: When the user confirms, you MUST ACTUALLY CALL the `generate_lensing_images` tool. DO NOT simply output text saying "I am generating the images" — you must invoke the function!

## Response Style
- Be concise but scientifically precise
- Use astrophysics terminology naturally
- Format parameter values clearly (e.g., "1x10¹² M☉" rather than "1e12")
- When presenting results, highlight key metadata and file locations
"""



# Agent Factory
def _detect_model() -> str:
    """
    Auto-detect the best available LLM provider based on environment variables.

    Priority: GEMINI_API_KEY → OPENAI_API_KEY → Ollama (local, no key needed)
    """
    if os.environ.get("GEMINI_API_KEY"):
        return "google-gla:gemini-flash-latest"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    # Fallback to Ollama (local, no API key required)
    return "ollama:llama3.2"


def create_lensing_agent(
    model_name: Optional[str] = None,
    deps: Optional[AgentDependencies] = None,
) -> Agent[AgentDependencies, str]:
    """
    Create a Pydantic AI agent configured for gravitational lensing simulation.

    Args:
        model_name: LLM model identifier. If None, auto-detects from
            available API keys. Supported formats:
            - "google-gla:gemini-flash-latest" 
            - "openai:gpt-4o-mini"
            - "ollama:llama3.2"              
        deps: Optional custom AgentDependencies. Defaults to standard config.

    Returns:
        Configured Agent instance.
    """
    if model_name is None:
        model_name = _detect_model()

    # Explicitly construct the model instance
    if model_name.startswith("ollama:"):
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider
        
        # Ensure base URL is set for Ollama provider
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        if not os.environ.get("OLLAMA_BASE_URL"):
            os.environ["OLLAMA_BASE_URL"] = base_url
            
        ollama_model_name = model_name.split("ollama:", 1)[1]
        
        # Use OpenAIChatModel with OllamaProvider to ensure tool calls work
        model_instance = OpenAIChatModel(
            model_name=ollama_model_name,
            provider=OllamaProvider(base_url=base_url)
        )
    else:
        model_instance = model_name

    dynamic_prompt = SYSTEM_PROMPT

    agent = Agent(
        model=model_instance,
        system_prompt=dynamic_prompt,
        deps_type=AgentDependencies,
        output_type=str,  # Agent produces text responses (tools produce structured data)
        retries=2,
    )

    # Register tool functions
    agent.tool(generate_lensing_images)
    agent.tool(get_model_info)
    agent.tool(validate_parameters)
    agent.tool(suggest_parameters)

    return agent


# Interactive Conversation Runner

async def run_interactive_session(
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Run an interactive multi-turn conversation with the lensing agent.

    Implements human-in-the-loop by maintaining message_history across turns,
    allowing the agent to ask follow-up questions and get user confirmation.
    """
    deps = AgentDependencies()
    if output_dir:
        deps.output_dir = output_dir

    agent = create_lensing_agent(model_name=model_name, deps=deps)

    resolved_model = model_name or _detect_model()
    print("=" * 70)
    print("DeepLense Simulation Assistant")
    print(f"LLM Provider: {resolved_model}")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 70)
    print()

    message_history = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Run agent with conversation history for multi-turn context
        print("  ⏳ Thinking...", end="", flush=True)
        try:
            if message_history is None:
                result = await agent.run(user_input, deps=deps)
            else:
                result = await agent.run(
                    user_input,
                    deps=deps,
                    message_history=message_history,
                )
        except Exception as exc:
            print(f"\r{' ' * 80}\r", end="")  # clear the line completely
            print(f"\n  ❌ Error: {exc}\n")
            continue

        print(f"\r{' ' * 80}\r", end="")  # clear the line completely

        # Update history for next turn
        message_history = result.all_messages()

        print("\rAssistant: ", end="")
        from rich.console import Console
        from rich.markdown import Markdown
        console = Console()
        
        # Fallback Check: Did the LLM output a raw JSON tool call as text instead of actually calling the tool?
        # Many small local models do this (< 8B parameters)
        raw_output = result.output.strip()
        if "generate_lensing_images" in raw_output and '"name"' in raw_output:
            import json
            try:
                # Find the outermost JSON block
                start_idx = raw_output.find('{')
                end_idx = raw_output.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = raw_output[start_idx:end_idx+1]
                    payload = json.loads(json_str)
                    if payload.get("name") == "generate_lensing_images":
                        console.print(Markdown("_Intercepted raw JSON tool call from local model. Generating images..._"))
                        # Create a mock run context for the tool
                        from pydantic_ai.tools import RunContext
                        mock_ctx = RunContext[AgentDependencies](
                            deps=deps,
                            retry=0,
                            tool_name="generate_lensing_images",
                        )
                        # Execute the tool manually
                        tool_result = await generate_lensing_images(mock_ctx, **payload.get("arguments", {}))
                        console.print(Markdown(tool_result))
                        print()
                        
                        # Only skip printing the raw output if we actually succeeded in generating
                        continue
            except Exception as e:
                pass # Fall through to standard markdown print
        
        console.print(Markdown(result.output))
        print()


# CLI Entry Point

def main():
    """CLI entry point for the interactive agent."""
    import logging

    if os.environ.get("LENSING_AGENT_DEBUG") == "1":
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.INFO)  # Keep httpx slightly quieter
        print("🔧 Debug logging enabled.\n")

    model_name = os.environ.get("LENSING_AGENT_MODEL", None)
    output_dir = os.environ.get("LENSING_AGENT_OUTPUT", None)

    asyncio.run(run_interactive_session(
        model_name=model_name,
        output_dir=output_dir,
    ))


if __name__ == "__main__":
    main()
