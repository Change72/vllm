"""Build a ShareGPT-derived prompt sample that exercises cross-instance KV
reuse: every prompt has the same long system-prefix (multi-block at
block_size=16) followed by a varied user question.

Defaults assume the host layout used for the TP=2 e2e doc:
   ShareGPT V3 unfiltered cleaned split at /raid/fly/datasets/
   output file at /raid/fly/kvp2p/kvp2p_sharegpt_sample.json (visible inside
   the container as /work/kvp2p_sharegpt_sample.json when /raid/fly/kvp2p
   is mounted to /work).

Override SRC / DST via env vars when running on a different machine:
   SHAREGPT_SRC=/path/to/your/sharegpt.json
   SHAREGPT_DST=/some/path/kvp2p_sharegpt_sample.json
"""
import json
import os
import random

SRC = os.environ.get(
    "SHAREGPT_SRC",
    "/raid/fly/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
)
DST = os.environ.get(
    "SHAREGPT_DST",
    "/raid/fly/kvp2p/kvp2p_sharegpt_sample.json",
)
N_PROMPTS = int(os.environ.get("SHAREGPT_N_PROMPTS", "32"))
SEED = int(os.environ.get("SHAREGPT_SEED", "20260625"))

random.seed(SEED)

with open(SRC) as f:
    data = json.load(f)

# Pick conversations whose first human turn is between 500 and 1200 chars
# so we get prompts with real prefix structure but the request fits in 8k
# context. Discard refusals / very short.
candidates = []
for row in data:
    convs = row.get("conversations", [])
    if not convs or convs[0].get("from") != "human":
        continue
    txt = convs[0].get("value", "")
    if 600 <= len(txt) <= 1400 and "\n" in txt:
        candidates.append(txt.strip())

random.shuffle(candidates)
chosen_qs = candidates[:N_PROMPTS]

SHARED_PREFIX = (
    "You are a careful senior engineer assisting users with technical "
    "questions. For each question, first briefly restate the user's "
    "question in your own words, then answer it in 2-3 short paragraphs. "
    "Be specific: cite concrete tools, libraries, configuration knobs, "
    "or commands wherever relevant. If the question is ambiguous, pick "
    "the most likely interpretation and proceed; do not stall by asking "
    "for clarification. If you do not know an answer with certainty, "
    "say so explicitly rather than inventing details. Keep the tone "
    "neutral and professional throughout, and finish with a one-line "
    "summary labelled `Summary:`."
)

items = []
for i, q in enumerate(chosen_qs):
    items.append({
        "idx": i,
        "shared_prefix": SHARED_PREFIX,
        "user_question": q,
        # Full prompt the test sends to the model.
        "prompt": f"{SHARED_PREFIX}\n\nUser: {q}\n\nAssistant:",
    })

os.makedirs(os.path.dirname(DST), exist_ok=True)
with open(DST, "w") as f:
    json.dump({
        "seed": SEED,
        "n": len(items),
        "shared_prefix_chars": len(SHARED_PREFIX),
        "items": items,
    }, f, indent=2)

print(f"wrote {len(items)} items to {DST}")
print(f"shared_prefix len = {len(SHARED_PREFIX)} chars")
print(f"first user_question len = {len(items[0]['user_question'])} chars")
print(f"first full prompt len = {len(items[0]['prompt'])} chars")
