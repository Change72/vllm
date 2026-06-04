# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable diverse prompt set for the two-engine evaluation.

16 prompts built from 16 distinct topical sentences, each repeated
enough times to produce ~800-1000 input tokens. Distinct opening words
guarantee no shared prefix, so vLLM's prefix cache treats them as
fully independent — i.e. every block hash is unique per prompt and
the target's plan-driven path is exercised in earnest rather than
shortcut by within-prompt prefix matching.
"""

from __future__ import annotations


_BASE_SENTENCES = [
    "The history of the Roman Empire spans more than a thousand years of "
    "political intrigue, military conquest, cultural exchange, and gradual "
    "decline that reshaped the Mediterranean world forever.",

    "Photosynthesis is the elegant biochemical process by which plants and "
    "certain microorganisms convert solar energy, water, and atmospheric "
    "carbon dioxide into glucose and breathable oxygen.",

    "Modern Python programming embraces idioms such as list comprehensions, "
    "context managers, and asynchronous coroutines that make the language "
    "expressive while remaining accessible to newcomers.",

    "Tokyo's bustling neighborhoods blend ultra-modern skyscrapers and neon "
    "advertising with centuries-old shrines, narrow alleyways, and quiet "
    "tea houses that preserve traditional Japanese aesthetics.",

    "Quantum mechanics introduces the strange notion that particles exist "
    "in superposition until measured, fundamentally challenging classical "
    "intuitions about causality, locality, and determinism in physics.",

    "The Great Barrier Reef harbors an astonishing diversity of marine "
    "life, including over fifteen hundred species of fish, hundreds of "
    "varieties of coral, and many threatened sea turtles, dugongs, sharks.",

    "Classical music composers like Bach, Mozart, and Beethoven established "
    "structural principles such as sonata form, counterpoint, and motivic "
    "development that continue to influence contemporary songwriters.",

    "Climate change driven by human greenhouse gas emissions threatens to "
    "destabilize agricultural yields, displace coastal communities, and "
    "accelerate the extinction of species already pressured by habitat loss.",

    "Neural network training relies on gradient descent variants that "
    "iteratively minimize a loss function by computing partial derivatives "
    "across millions of trainable parameters through backpropagation.",

    "Mediterranean cuisine emphasizes olive oil, fresh vegetables, legumes, "
    "whole grains, and moderate amounts of fish and poultry, contributing "
    "to lower rates of cardiovascular disease and longer life expectancy.",

    "Shakespeare's tragedies explore timeless themes of ambition, jealousy, "
    "revenge, and self-deception through unforgettable characters like "
    "Hamlet, Macbeth, Othello, and King Lear, whose flaws drive their fall.",

    "Renewable energy infrastructure including solar farms, offshore wind "
    "turbines, and grid-scale battery storage is being deployed worldwide "
    "to displace fossil fuel generation and reduce planetary emissions.",

    "Cellular respiration in mitochondria extracts chemical energy from "
    "glucose through glycolysis, the citric acid cycle, and oxidative "
    "phosphorylation, ultimately producing adenosine triphosphate as fuel.",

    "Self-driving cars combine high-resolution lidar, stereoscopic cameras, "
    "millimeter-wave radar, and inertial measurement units feeding into "
    "perception and planning stacks that must operate reliably in real time.",

    "Cryptocurrencies leverage cryptographic hash functions and "
    "consensus protocols such as proof-of-work or proof-of-stake to "
    "achieve trustless agreement on a distributed ledger of transactions.",

    "Honeybee colonies coordinate foraging trips through the celebrated "
    "waggle dance, encoding distance and direction relative to the sun in "
    "the geometry and duration of the choreographed body movements.",
]


def build_prompts(repeat: int = 50) -> list[str]:
    """Return 16 long prompts, each ``repeat`` copies of a distinct base
    sentence. ``repeat=50`` yields roughly 800-1100 tokens per prompt
    for Qwen3 BPE tokenisation."""
    return [(s + " ") * repeat for s in _BASE_SENTENCES]
