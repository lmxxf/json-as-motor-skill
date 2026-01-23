"""
Experiment 2: Noise Injection
==============================
Tests whether the structural parsing circuit is robust to semantic noise
injected inside valid JSON structures.

Prediction: The model can recover correct JSON structure even when
irrelevant natural language text is inserted between key-value pairs.
Recovery should show a phase transition (not gradual degradation),
consistent with a dedicated circuit being overwhelmed.

Requirements:
  pip install openai

Usage:
  export DEEPSEEK_API_KEY=xxx
  python experiment2_noise_injection.py
"""

import json
import os
import time
from datetime import datetime
from typing import Optional
from openai import OpenAI


# ===== Noise Levels =====

NOISE_PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "In 1969 humans first walked on the moon.",
    "Photosynthesis converts carbon dioxide into oxygen.",
    "The capital of Australia is Canberra, not Sydney.",
    "Dolphins are mammals that live in the ocean.",
    "Pi is approximately 3.14159265358979.",
    "Shakespeare wrote 37 plays during his lifetime.",
    "The speed of light is about 300,000 km per second.",
    "Mount Everest is the tallest mountain on Earth.",
    "Water boils at 100 degrees Celsius at sea level.",
]

BASE_JSON = {
    "users": [
        {"name": "Alice", "role": "admin", "active": True},
        {"name": "Bob", "role": "editor", "active": False},
        {"name": "Carol", "role": "viewer", "active": True}
    ]
}


def inject_noise(json_str: str, level: int) -> str:
    """
    Inject noise into JSON string at various levels:
    Level 0: No noise (control)
    Level 1: 1 short phrase (1 sentence)
    Level 2: 3 phrases (~50 words)
    Level 3: 5 phrases (~100 words)
    Level 4: 10 phrases (~200 words) - should overwhelm the circuit
    """
    if level == 0:
        return json_str

    noise_count = [0, 1, 3, 5, 10][level]
    noise_text = " ".join(NOISE_PHRASES[:noise_count])

    # Insert noise after the first object in the array
    insertion_point = json_str.find("},") + 2
    if insertion_point < 2:
        insertion_point = len(json_str) // 2

    noisy = (
        json_str[:insertion_point] +
        f'\n"{noise_text}",\n' +
        json_str[insertion_point:]
    )
    return noisy


# ===== Metrics =====

def evaluate_recovery(output: str, original: dict) -> dict:
    """Evaluate how well the model recovered the original structure."""
    metrics = {
        "is_valid_json": False,
        "structure_preserved": False,
        "data_preserved": False,
        "noise_removed": False,
        "noise_acknowledged": False,
    }

    # Check if output is valid JSON
    try:
        parsed = json.loads(output.strip())
        metrics["is_valid_json"] = True
    except (json.JSONDecodeError, ValueError):
        # Try to extract JSON from output (model might add explanation)
        import re
        json_match = re.search(r'[\[{].*[\]}]', output, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                metrics["is_valid_json"] = True
            except:
                return metrics
        else:
            return metrics

    # Check structure preservation
    if isinstance(parsed, dict) and "users" in parsed:
        metrics["structure_preserved"] = True
        users = parsed.get("users", [])
        if isinstance(users, list) and len(users) == 3:
            # Check if all user objects have correct keys
            expected_keys = {"name", "role", "active"}
            all_correct = all(
                isinstance(u, dict) and set(u.keys()) == expected_keys
                for u in users
            )
            if all_correct:
                metrics["data_preserved"] = True
    elif isinstance(parsed, list) and len(parsed) == 3:
        # Also accept if returned as array directly
        metrics["structure_preserved"] = True

    # Check if noise was removed
    noise_indicators = ["fox", "moon", "dolphin", "Shakespeare", "Everest", "Pi"]
    output_lower = output.lower()
    has_noise = any(ind.lower() in output_lower for ind in noise_indicators)
    metrics["noise_removed"] = not has_noise

    # Check if model acknowledged the noise (upper-layer awareness)
    acknowledgment_words = ["noise", "irrelevant", "removed", "extraneous",
                           "invalid", "extra", "cleaned", "ignored"]
    metrics["noise_acknowledged"] = any(w in output_lower for w in acknowledgment_words)

    return metrics


# ===== API Call =====

def call_model(prompt: str, model: str = "deepseek-chat") -> Optional[str]:
    """Call DeepSeek API."""
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=800,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  API Error: {e}")
        return None


TASK_PROMPT = """The following JSON contains some noise/errors. Parse it and output a corrected, clean version. Output ONLY the valid JSON, nothing else.

Input:
{noisy_json}

Output:"""


# ===== Main Experiment =====

def run_experiment(model: str = "deepseek-chat", repetitions: int = 3):
    """Run noise injection experiment across all levels."""
    base_json_str = json.dumps(BASE_JSON, indent=2)
    results = {level: [] for level in range(5)}

    print(f"=" * 60)
    print(f"Experiment 2: Noise Injection")
    print(f"Model: {model}")
    print(f"Noise levels: 0-4 (0=control, 4=severe)")
    print(f"Repetitions per level: {repetitions}")
    print(f"=" * 60)

    for level in range(5):
        print(f"\n--- Noise Level {level} ---")
        noisy = inject_noise(base_json_str, level)
        noise_word_count = len(noisy.split()) - len(base_json_str.split())
        print(f"  Noise words injected: ~{noise_word_count}")

        for rep in range(repetitions):
            prompt = TASK_PROMPT.format(noisy_json=noisy)
            output = call_model(prompt, model)

            if output is None:
                print(f"  Rep {rep+1}: FAILED (API error)")
                continue

            metrics = evaluate_recovery(output, BASE_JSON)
            metrics["noise_level"] = level
            metrics["noise_word_count"] = noise_word_count
            metrics["input"] = noisy
            metrics["output"] = output
            results[level].append(metrics)

            status = "OK" if metrics["data_preserved"] else "DEGRADED"
            ack = " (acknowledged)" if metrics["noise_acknowledged"] else ""
            print(f"  Rep {rep+1}: {status} | valid={metrics['is_valid_json']} "
                  f"struct={metrics['structure_preserved']} "
                  f"data={metrics['data_preserved']} "
                  f"noise_gone={metrics['noise_removed']}{ack}")
            time.sleep(0.5)

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Level':<8} {'Valid JSON':<12} {'Structure':<12} {'Data':<12} {'Noise Removed':<15} {'Acknowledged':<12}")
    print("-" * 71)

    for level in range(5):
        trials = results[level]
        if trials:
            n = len(trials)
            valid = sum(t["is_valid_json"] for t in trials) / n
            struct = sum(t["structure_preserved"] for t in trials) / n
            data = sum(t["data_preserved"] for t in trials) / n
            noise_rm = sum(t["noise_removed"] for t in trials) / n
            ack = sum(t["noise_acknowledged"] for t in trials) / n
            print(f"{level:<8} {valid:<12.2f} {struct:<12.2f} {data:<12.2f} {noise_rm:<15.2f} {ack:<12.2f}")

    # ===== Save Results =====
    output_path = os.path.join(os.path.dirname(__file__), "results_exp2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "noise_injection",
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "results": {str(k): v for k, v in results.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ===== Interpretation =====
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")
    print("""
Key observations to look for:

1. PHASE TRANSITION: Does recovery drop suddenly (e.g., 100% at L3, 0% at L4)?
   → Supports dedicated circuit with fixed capacity
   → Gradual degradation would suggest general capability, not dedicated circuit

2. NOISE ACKNOWLEDGMENT: Does the model mention removing noise?
   → If yes at high levels but not low: upper layer notices when circuit is stressed
   → If never: lower-layer circuit handles everything silently

3. STRUCTURE vs DATA preservation:
   → If structure survives longer than data: structural circuit is more robust
   → Supports the "motor circuit" analogy (structure is automatic, data is cognitive)
""")


if __name__ == "__main__":
    run_experiment()
