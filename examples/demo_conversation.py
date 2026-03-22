"""
Demo: Multi-turn conversation with the DeepLense Simulation Agent.
Shows the human-in-the-loop flow:
  1. User makes a request
  2. Agent asks clarifying questions
  3. User provides answers
  4. Agent confirms parameters
  5. User approves → agent generates images

Usage:
    export GEMINI_API_KEY="your-key"
    python -m examples.demo_conversation
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import create_lensing_agent
from src.config import AgentDependencies


async def demo():
    """Run a scripted multi-turn conversation demonstrating the agent."""

    deps = AgentDependencies(output_dir="./outputs/demo")
    agent = create_lensing_agent(deps=deps)

    print("=" * 70)
    print("  DeepLense Agent — Demo Conversation")
    print("=" * 70)

    #Turn 1: User makes a vague request
    print("\n[Turn 1] User: Generate some axion lensing images\n")
    result = await agent.run(
        "Generate some axion lensing images",
        deps=deps,
    )
    print(f"Assistant: {result.output}\n")

    #Turn 2: User provides clarification
    print("[Turn 2] User: Use Model_I, 3 images, axion mass around 1e-23\n")
    result = await agent.run(
        "Use Model_I, 3 images, axion mass around 1e-23 eV. "
        "Use default redshifts and halo mass.",
        deps=deps,
        message_history=result.all_messages(),
    )
    print(f"Assistant: {result.output}\n")

    #Turn 3: User confirms
    print("[Turn 3] User: Yes, go ahead!\n")
    result = await agent.run(
        "Yes, go ahead and generate them!",
        deps=deps,
        message_history=result.all_messages(),
    )
    print(f"Assistant: {result.output}\n")

    print("  Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demo())
