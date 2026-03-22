# Examples

Runnable demonstration scripts. Unlike the tests, **these need real dependencies**: a configured LLM API key and DeepLenseSim installed.

---

## `demo_conversation.py`

Runs a scripted 3-turn conversation through the real agent, demonstrating the human-in-the-loop flow end-to-end.

**Requires:** Gemini, OpenAI, or Ollama configured (see root README).

```bash
# From the repo root
export GEMINI_API_KEY="your-key"
python -m examples.demo_conversation
```

**What happens:**

| Turn | User says | Agent does |
|------|-----------|-----------|
| 1 | "Generate some axion lensing images" | Asks clarifying questions (model, count, axion mass) |
| 2 | "Use Model_I, 3 images, axion mass 1e-23 eV, default redshifts" | Validates params, presents summary, asks for confirmation |
| 3 | "Yes, go ahead!" | Calls `generate_lensing_images`, saves `.npy` + `.png` to `outputs/demo/` |

Each turn passes `message_history=result.all_messages()` so the agent has full conversation context.

---

## `batch_generation.py`

Bypasses the LLM agent and calls `SimulationEngine.generate_batch()` directly. Useful for scripted dataset generation without any conversational overhead.

**Requires:** DeepLenseSim installed. No API key needed.

```bash
# From the repo root
python -m examples.batch_generation
```

**What it generates:** 2 images for every combination of model × substructure type:

| Model | Substructure | Output dir |
|-------|-------------|-----------|
| Model_I | no_sub | `outputs/batch/Model_I/no_sub/` |
| Model_I | axion | `outputs/batch/Model_I/axion/` |
| Model_I | cdm | `outputs/batch/Model_I/cdm/` |
| Model_II | no_sub | `outputs/batch/Model_II/no_sub/` |
| Model_II | axion | `outputs/batch/Model_II/axion/` |
| Model_II | cdm | `outputs/batch/Model_II/cdm/` |

All runs use `random_seed=42` for reproducibility. Each output is a `.npy` (raw float64 array) and `.png` (sqrt-scaled, inferno colormap).
