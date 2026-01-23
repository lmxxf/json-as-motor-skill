"""
Experiment 1: Structure vs Content Disruption
==============================================
Tests whether LLMs process JSON structure independently of semantic content.

Four conditions:
  A) Valid JSON + Semantic content
  B) Valid JSON + Nonsense content
  C) Invalid JSON + Semantic content
  D) Invalid JSON + Nonsense content

Prediction: Structural correctness depends on input structure (A≈B >> C≈D),
while semantic coherence depends on input content (A>>B, C>>D).
If these two dimensions are separable, it proves the dual-layer architecture.

Requirements:
  pip install anthropic openai

Usage:
  export ANTHROPIC_API_KEY=xxx
  python experiment1_structure_vs_content.py
"""

import json
import os
import time
from datetime import datetime
from typing import Optional
import anthropic

# ===== Stimuli =====

STIMULI = {
    "valid_semantic": [
        '{"name": "Alice", "age": 28, "city": "Tokyo"}',
        '{"product": "laptop", "price": 999, "brand": "Dell"}',
        '{"country": "France", "capital": "Paris", "population": 67000000}',
        '{"title": "Inception", "year": 2010, "director": "Nolan"}',
        '{"language": "Python", "version": "3.11", "paradigm": "multi"}',
    ],
    "valid_nonsense": [
        '{"xqz": "brmf", "plk": 42, "wnv": "htjd"}',
        '{"kkf": "zznp", "qrr": 77, "mxl": "vvbt"}',
        '{"pph": "llkw", "dds": 15, "yyq": "rrms"}',
        '{"ttg": "ffnx", "bbj": 63, "ccw": "hhzp"}',
        '{"nnr": "sskt", "wwf": 28, "ggm": "ppxl"}',
    ],
    "invalid_semantic": [
        '{"name" "Alice", "age": 28 "city" "Tokyo"',
        '{"product" "laptop" "price" 999, "brand": "Dell"',
        '{"country": "France" "capital" "Paris", "population" 67000000',
        '{"title" "Inception", "year" 2010 "director": "Nolan"',
        '{"language" "Python" "version": "3.11" "paradigm" "multi"',
    ],
    "invalid_nonsense": [
        '{"xqz" "brmf" "plk" 42, "wnv" "htjd"',
        '{"kkf" "zznp", "qrr" 77 "mxl" "vvbt"',
        '{"pph": "llkw" "dds" 15, "yyq" "rrms"',
        '{"ttg" "ffnx" "bbj": 63 "ccw" "hhzp"',
        '{"nnr" "sskt", "wwf" 28 "ggm": "ppxl"',
    ],
}

TASK_PROMPT = """Complete the following to make it a valid JSON array containing 3 objects with the same schema as the input. Output ONLY the JSON array, nothing else.

Input:
{stimulus}

Output:"""


# ===== Metrics =====

def check_structural_correctness(output: str) -> bool:
    """Is the output valid JSON?"""
    try:
        parsed = json.loads(output.strip())
        if isinstance(parsed, list) and len(parsed) == 3:
            return True
        # Also accept if it's valid JSON but not exactly 3 items
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def check_schema_consistency(output: str, stimulus: str) -> float:
    """Does the output maintain the key structure of the input?"""
    try:
        parsed = json.loads(output.strip())
        if not isinstance(parsed, list):
            parsed = [parsed]

        # Extract keys from stimulus
        try:
            stim_obj = json.loads(stimulus)
            if isinstance(stim_obj, dict):
                expected_keys = set(stim_obj.keys())
            else:
                return 0.0
        except:
            # For invalid JSON stimuli, try to extract keys heuristically
            import re
            keys = re.findall(r'"(\w+)"', stimulus)
            expected_keys = set(keys[:3])  # Take first 3 as likely keys

        if not expected_keys:
            return 0.0

        # Check each output object
        scores = []
        for obj in parsed:
            if isinstance(obj, dict):
                overlap = len(set(obj.keys()) & expected_keys) / len(expected_keys)
                scores.append(overlap)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    except (json.JSONDecodeError, ValueError):
        return 0.0


def check_semantic_coherence(output: str) -> float:
    """
    Are the generated values semantically meaningful?
    Simple heuristic: check if string values are real words vs random chars.
    """
    try:
        parsed = json.loads(output.strip())
        if not isinstance(parsed, list):
            parsed = [parsed]

        # Common English words (simple check)
        import re
        all_string_values = []
        for obj in parsed:
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, str):
                        all_string_values.append(v)

        if not all_string_values:
            return 0.5  # No string values to judge

        # Heuristic: real words have vowels and common patterns
        vowels = set('aeiouAEIOU')
        meaningful_count = 0
        for s in all_string_values:
            has_vowel = any(c in vowels for c in s)
            reasonable_length = 2 <= len(s) <= 30
            not_random = not bool(re.match(r'^[bcdfghjklmnpqrstvwxyz]+$', s.lower()))
            if has_vowel and reasonable_length and not_random:
                meaningful_count += 1

        return meaningful_count / len(all_string_values)

    except (json.JSONDecodeError, ValueError):
        return 0.0


# ===== API Call =====

def call_model(prompt: str, model: str = "claude-sonnet-4-20250514") -> Optional[str]:
    """Call Claude API and return response text."""
    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"  API Error: {e}")
        return None


# ===== Main Experiment =====

def run_experiment(model: str = "claude-sonnet-4-20250514", trials_per_condition: int = 5):
    """Run all conditions and collect metrics."""
    results = {condition: [] for condition in STIMULI.keys()}

    print(f"=" * 60)
    print(f"Experiment 1: Structure vs Content Disruption")
    print(f"Model: {model}")
    print(f"Trials per condition: {trials_per_condition}")
    print(f"Total API calls: {trials_per_condition * 4}")
    print(f"=" * 60)

    for condition, stimuli in STIMULI.items():
        print(f"\n--- Condition: {condition} ---")
        for i, stimulus in enumerate(stimuli[:trials_per_condition]):
            prompt = TASK_PROMPT.format(stimulus=stimulus)
            output = call_model(prompt, model)

            if output is None:
                print(f"  Trial {i+1}: FAILED (API error)")
                continue

            sc = check_structural_correctness(output)
            schc = check_schema_consistency(output, stimulus)
            semc = check_semantic_coherence(output)

            results[condition].append({
                "stimulus": stimulus,
                "output": output,
                "structural_correctness": sc,
                "schema_consistency": schc,
                "semantic_coherence": semc,
            })

            print(f"  Trial {i+1}: SC={sc}, SchC={schc:.2f}, SemC={semc:.2f}")
            time.sleep(0.5)  # Rate limiting

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Condition':<20} {'SC (mean)':<12} {'SchC (mean)':<12} {'SemC (mean)':<12}")
    print("-" * 56)

    for condition, trials in results.items():
        if trials:
            sc_mean = sum(t["structural_correctness"] for t in trials) / len(trials)
            schc_mean = sum(t["schema_consistency"] for t in trials) / len(trials)
            semc_mean = sum(t["semantic_coherence"] for t in trials) / len(trials)
            print(f"{condition:<20} {sc_mean:<12.2f} {schc_mean:<12.2f} {semc_mean:<12.2f}")

    # ===== Save Results =====
    output_path = os.path.join(os.path.dirname(__file__), "results_exp1.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "structure_vs_content",
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ===== Interpretation =====
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")
    print("""
If SC(valid_semantic) ≈ SC(valid_nonsense) >> SC(invalid_*):
  → Structure processing is INDEPENDENT of content
  → Supports dual-layer hypothesis (structural motor circuit)

If SemC(valid_semantic) >> SemC(valid_nonsense):
  → Semantic processing is INDEPENDENT of structure
  → The two dimensions are separable

If both hold: STRONG EVIDENCE for dual-layer architecture.
""")


if __name__ == "__main__":
    run_experiment()
