# Agentic Workflow for DeepLenseSim - Pydantic AI

An agentic workflow using **Pydantic AI** that wraps the [DeepLenseSim](https://github.com/mwt5345/DeepLenseSim) simulation pipeline to generate strong gravitational lensing images through **natural language interaction**.

## Features

- **Natural language interface** - describe simulations in plain English
- **Human-in-the-loop** - agent asks follow-up questions before generating
- **Structured output** - typed Pydantic models for all parameters and results
- **Two model configurations** - Model_I (150×150 Gaussian PSF) and Model_II (64×64 Euclid)
- **Three dark matter classes** - no substructure, axion (vortex), CDM (point-mass)
- **Validated parameters** - physics constraints enforced via Pydantic validators

## Architecture

```
User Prompt → Pydantic AI Agent → Clarify Parameters (Human-in-the-Loop)
                                ↓
                        Tool Functions → SimulationEngine → DeepLenseSim (DeepLens class)
                                ↓
                        Structured Output (SimulationResult + ImageMetadata)
```

### Project Structure

```
├── src/
│   ├── agent.py               # Agent definition + system prompt + interactive runner
│   ├── models.py              # Pydantic models (params, results, enums)
│   ├── tools.py               # Tool functions (generate, info, validate)
│   ├── simulation_engine.py   # DeepLenseSim wrapper
│   └── config.py              # Dependencies and defaults
├── examples/
│   ├── demo_conversation.py   # Multi-turn conversation demo
│   └── batch_generation.py    # Programmatic batch usage
├── tests/
│   ├── test_models.py         # Model validation tests
│   ├── test_tools.py          # Tool function tests
│   └── test_agent.py          # Agent integration tests
├── DeepLenseSim/              # Cloned DeepLenseSim repository
├── outputs/                   # Generated images
└── requirements.txt
```

## Installation

### 1. Install pyHalo from source

```bash
git clone https://github.com/dangilman/pyHalo.git
cd pyHalo
python setup.py develop
cd ..
```

### 2. Install DeepLenseSim

```bash
cd DeepLenseSim
python setup.py install
cd ..
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up an LLM provider (pick ONE)

The agent auto-detects your provider based on environment variables. **No paid API key required!**

#### Option A: Google Gemini (Free - recommended)

1. Get a free API key at [ai.google.dev](https://ai.google.dev/)
2. Set the environment variable:

```bash
export GEMINI_API_KEY="your-gemini-key"
```

#### Option B: Ollama (100% local - no API key needed)

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Pull a model:

```bash
ollama pull llama3.2
```

3. That's it! The agent will detect Ollama automatically if no API keys are set.

#### Option C: OpenAI (paid)

```bash
export OPENAI_API_KEY="sk-..."
```

#### Override: You can also set a specific model explicitly:

```bash
export LENSING_AGENT_MODEL="google-gla:gemini-2.0-flash"
# or
export LENSING_AGENT_MODEL="ollama:llama3.2"
# or
export LENSING_AGENT_MODEL="openai:gpt-4o-mini"
```

## Usage

### Interactive Agent

To start the interactive chat session, run the agent module from the root directory:

```bash
python -m src.agent
```

#### Selecting a Specific Model
By default, the agent will attempt to detect the best available model based on your environment variables (`GEMINI_API_KEY`, etc.). However, you can force the agent to use a specific model by setting the `LENSING_AGENT_MODEL` environment variable.

For Google Gemini (requires `GEMINI_API_KEY`):
```bash
LENSING_AGENT_MODEL="google-gla:gemini-2.0-flash" python -m src.agent
```

For a local Ollama model (no API key required, must have run `ollama pull <model>` first):
```bash
LENSING_AGENT_MODEL="ollama:qwen2.5-coder:7b" python -m src.agent
```

#### What to Expect
This launches an interactive terminal session where you can type natural language requests. The agent will parse your request, ask clarifying questions if parameters are missing, and generate images:

```text
You: Generate 5 CDM lensing images using Model_I
Assistant: I'd like to confirm your simulation parameters...
You: Yes, use defaults for everything
Assistant: **Simulation Completed** - 5 images generated...
```

### Demo Conversation

```bash
python -m examples.demo_conversation
```

### Batch Generation (No LLM)

```bash
python -m examples.batch_generation
```

### Running Tests

```bash
pytest tests/ -v
```

## Supported Configurations

| Property | Model_I | Model_II |
|---|---|---|
| Resolution | 150×150 px | 64×64 px |
| Pixel Scale | 0.05 arcsec/px | 0.101 arcsec/px |
| PSF | Gaussian (FWHM=0.087") | Euclid VIS |
| Source Light | Sersic (amplitude) | Sersic (magnitude) |
| Sim Method | `simple_sim()` | `simple_sim_2()` + SimAPI |

### Dark Matter Substructure Types

- **no_sub** - Smooth lens, no dark matter substructure
- **axion** - Axion vortex substructure (mass range: 10⁻²⁴–10⁻²² eV)
- **cdm** - Cold dark matter with point-mass subhalos

## Input Verification and Human-in-the-Loop Flow

The agent translates natural language simulation requests into verified, physically accurate parameter sets, employing a strict multi-phase interaction to keep the human in control:

1. **Phase 1 - Natural Language Parsing**: The Pydantic AI agent parses the user's NLP prompt to extract requested simulation parameters (masses, redshifts, model configurations, and dark matter substructures).
2. **Phase 2 - Physics Checking & Validation**: The parameters are passed through strict Pydantic model validators. The agent uses the `validate_parameters` tool as a "pre-flight check", evaluating constraints such as ensuring the source redshift (`z_source`) is strictly greater than the halo redshift (`z_halo`), and flagging physically unrealistic mass values. 
3. **Phase 3 - Elicitation**: If any required parameters are missing or anomalous, the agent asks targeted follow-up questions to request clarification or corrections from the user.
4. **Phase 4 - Confirmation**: Before any computationally expensive image generation begins, the agent explicitly presents the fully resolved and validated parameter set, requesting user approval.

## Agent Tools

To achieve this workflow, our agentic system is equipped with three highly specialized tools that enable dynamic introspection, validation, and execution:

### 1. Model Info Tool (`get_model_info`)
This tool acts as the agent's internal knowledge base about DeepLenseSim. When a user asks about the differences between models or needs help choosing one, the agent queries this tool to retrieve real-time specs on resolutions, pixel scales, instruments, and PSF types (e.g., Gaussian vs. Euclid SimAPI) for `Model_I` and `Model_II`.

### 2. Validation Pre-flight Tool (`validate_parameters`)
Before executing simulations, the agent tests the current parameter state through this tool. It checks for:
- Invalid model or substructure types.
- Physics violations (e.g., source galaxy placed in front of the lens).
- Edge-case warnings, like extremely low/high dark matter halo masses (cluster-scales vs. galaxy-scales) or out-of-range Axion masses.

The tool outputs a structured report with strict errors and warnings that the agent then relays to the user to fix.

### 3. Simulation Execution Tool (`generate_lensing_images`)
Invoked only upon explicit user confirmation, this tool acts as the bridge to DeepLenseSim's engine. It accepts strongly-typed inputs (e.g., halo mass, substructure type, redshift data), directs DeepLenseSim to spawn batch image generation, and returns a comprehensive status string with metadata and generated image paths.

## License

MIT (follows DeepLenseSim)
