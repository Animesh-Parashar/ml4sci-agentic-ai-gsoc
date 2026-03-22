# Tests

Unit tests for the DeepLense agent. **No API key or DeepLenseSim installation required** — the simulation engine is mocked throughout.

## Run

```bash
# From the repo root
pytest tests/ -v
```

Expected: **48 tests pass** in ~1 second.

## Test Files

### `test_models.py`
Tests the Pydantic data models in isolation.

| Test Class | What it checks |
|---|---|
| `TestSimulationParameters` | Valid construction, physics constraint (`z_source > z_halo`), auto-draw of `axion_mass`, out-of-range rejections, enum validation |
| `TestImageMetadata` | Field defaults, optional fields |
| `TestSimulationResult` | Success/failure result construction |
| `TestEnums` | `SubstructureType` and `ModelConfig` enum values |

Example — the redshift ordering constraint:
```python
# This raises ValidationError -- source must be behind the lens
SimulationParameters(model_config_type="Model_I", substructure_type="cdm",
                     z_halo=1.5, z_source=0.8)
```

### `test_tools.py`
Tests the four agent tool functions. `SimulationEngine.generate_batch` is patched with `AsyncMock` so no real simulations run.

| Test Class | What it checks |
|---|---|
| `TestGetModelInfo` | Returns correct specs for each model; handles unknown model names |
| `TestValidateParameters` | Catches invalid enums, bad redshifts, extreme masses, missing axion mass; warns on large batches |
| `TestGenerateLensingImages` | Calls the engine with correct params; formats the result string correctly |
| `TestSuggestParameters` | All 6 scenario keywords; no-argument menu; unknown scenario fallback |

### `test_agent.py`
Structural tests for the agent factory and system prompt.

| Test | What it checks |
|---|---|
| `test_create_agent` | `create_lensing_agent()` returns a valid agent (dummy API key injected via `patch.dict`) |
| `test_system_prompt_content` | Prompt contains HITL protocol phrases and all model/substructure names |
| `test_system_prompt_mandates_confirmation` | `"Shall I proceed"` is present -- the confirmation gate is enforced in the prompt |
| `TestAgentDependencies` | Output directory creation, custom path override |
