"""
Experiment 4: Format-Switching Cost
=====================================
Tests whether switching between natural language and JSON mid-generation
incurs a measurable cost, indicating circuit switching.

Design: Ask the model to output text→JSON→text in a single response.
Measure error rates and semantic continuity at transition points.

Requirements:
  pip install openai

Usage:
  export DEEPSEEK_API_KEY=xxx
  python experiment4_format_switching.py
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Optional, Dict, List
from openai import OpenAI


# ===== Prompts =====

SWITCHING_PROMPTS = [
    {
        "id": "binary_tree",
        "prompt": (
            "First, explain what a binary tree is in 2-3 sentences of plain English. "
            "Then, output a JSON representation of a sample binary tree with 5 nodes "
            "(use keys: value, left, right). "
            "Then, in plain English, describe the specific tree you just output "
            "(mention the actual values and structure)."
        ),
    },
    {
        "id": "recipe",
        "prompt": (
            "First, briefly describe what a pasta carbonara is in 1-2 sentences. "
            "Then, output a JSON object representing the recipe with keys: "
            "name, servings, ingredients (array of {item, amount}), and steps (array of strings). "
            "Then, in plain English, explain which step in YOUR recipe is the trickiest and why."
        ),
    },
    {
        "id": "api_endpoint",
        "prompt": (
            "First, explain what a REST API endpoint is in 2 sentences. "
            "Then, output a JSON object representing a user registration endpoint with: "
            "method, path, request_body (with fields), and response (with status codes). "
            "Then, explain in plain English what would happen if someone sent an invalid email "
            "to YOUR specific endpoint."
        ),
    },
    {
        "id": "chess_position",
        "prompt": (
            "First, explain what castling is in chess in 1-2 sentences. "
            "Then, output a JSON representation of a chess board position where "
            "White can castle kingside (use a 2D array for the board, null for empty squares). "
            "Then, in plain English, explain why the specific position you showed allows castling."
        ),
    },
    {
        "id": "neural_network",
        "prompt": (
            "First, explain what a neural network layer is in 2 sentences. "
            "Then, output a JSON object representing a simple 3-layer neural network "
            "(each layer has: type, input_dim, output_dim, activation). "
            "Then, in plain English, explain why you chose those specific dimensions "
            "and activations in YOUR network."
        ),
    },
]


# ===== Analysis =====

def extract_sections(output: str) -> Dict:
    """
    Parse the output into three sections: pre-JSON text, JSON, post-JSON text.
    """
    result = {
        "pre_text": "",
        "json_block": "",
        "post_text": "",
        "json_valid": False,
        "json_parsed": None,
        "transition_in_clean": False,   # Clean entry into JSON
        "transition_out_clean": False,  # Clean exit from JSON
    }

    # Try to find JSON block (look for { or [ that starts a JSON structure)
    # Strategy: find the largest valid JSON substring
    best_json = None
    best_start = -1
    best_end = -1

    # Find all potential JSON starts
    for i, char in enumerate(output):
        if char in '{[':
            # Try progressively longer substrings
            depth = 0
            for j in range(i, len(output)):
                if output[j] in '{[':
                    depth += 1
                elif output[j] in '}]':
                    depth -= 1
                    if depth == 0:
                        candidate = output[i:j+1]
                        try:
                            parsed = json.loads(candidate)
                            if best_json is None or len(candidate) > len(best_json):
                                best_json = candidate
                                best_start = i
                                best_end = j + 1
                                result["json_parsed"] = parsed
                        except:
                            pass
                        break

    if best_json:
        result["json_block"] = best_json
        result["json_valid"] = True
        result["pre_text"] = output[:best_start].strip()
        result["post_text"] = output[best_end:].strip()

        # Check transition cleanliness
        # Clean entry: pre_text ends with sentence-ending punctuation or newline
        pre = output[:best_start].rstrip()
        result["transition_in_clean"] = (
            pre.endswith(('.', ':', '\n', '```', '```json')) or
            pre.endswith((':\n', '.\n'))
        )

        # Clean exit: post_text starts with a capital letter or newline
        post = output[best_end:].lstrip()
        result["transition_out_clean"] = (
            bool(post) and (post[0].isupper() or post[0] == '\n' or post[0] == '*')
        )
    else:
        result["pre_text"] = output

    return result


def check_semantic_continuity(sections: Dict, prompt_id: str) -> Dict:
    """
    Check if the post-JSON text correctly references the JSON content.
    This tests whether the upper layer maintained awareness during JSON output.
    """
    metrics = {
        "references_json_content": False,
        "specific_values_mentioned": 0,
        "coherent_explanation": False,
    }

    if not sections["json_valid"] or not sections["post_text"]:
        return metrics

    post = sections["post_text"].lower()
    json_obj = sections["json_parsed"]

    # Extract string values from JSON to check if post-text references them
    def extract_values(obj, depth=0) -> List[str]:
        values = []
        if isinstance(obj, dict):
            for v in obj.values():
                values.extend(extract_values(v, depth+1))
        elif isinstance(obj, list):
            for item in obj:
                values.extend(extract_values(item, depth+1))
        elif isinstance(obj, str) and len(obj) > 2:
            values.append(obj.lower())
        return values

    json_values = extract_values(json_obj)

    # Check how many JSON values appear in post-text
    mentioned = [v for v in json_values if v in post]
    metrics["specific_values_mentioned"] = len(mentioned)
    metrics["references_json_content"] = len(mentioned) > 0

    # Check if post-text is a coherent explanation (not just random text)
    # Heuristic: has explanation words and references the task
    explanation_words = ["because", "since", "this", "the", "which", "that",
                        "allows", "means", "specific", "chose", "position"]
    has_explanation = sum(1 for w in explanation_words if w in post) >= 3
    metrics["coherent_explanation"] = has_explanation

    return metrics


def check_json_boundary_errors(sections: Dict) -> Dict:
    """
    Check for errors specifically at JSON boundaries.
    """
    metrics = {
        "json_starts_cleanly": False,
        "json_ends_cleanly": False,
        "first_token_error": False,
        "last_token_error": False,
    }

    if not sections["json_block"]:
        return metrics

    block = sections["json_block"]

    # Check first few characters
    metrics["json_starts_cleanly"] = block.lstrip()[0] in '{['

    # Check last few characters
    metrics["json_ends_cleanly"] = block.rstrip()[-1] in '}]'

    # Check if there's garbage at boundaries
    try:
        json.loads(block)
    except json.JSONDecodeError as e:
        # If error is near start or end, it's a boundary error
        if e.pos < 5:
            metrics["first_token_error"] = True
        elif e.pos > len(block) - 5:
            metrics["last_token_error"] = True

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
            max_tokens=1500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  API Error: {e}")
        return None


# ===== Main Experiment =====

def run_experiment(model: str = "deepseek-chat", repetitions: int = 2):
    """Run format-switching experiment."""
    results = []

    print(f"=" * 60)
    print(f"Experiment 4: Format-Switching Cost")
    print(f"Model: {model}")
    print(f"Prompts: {len(SWITCHING_PROMPTS)}")
    print(f"Repetitions: {repetitions}")
    print(f"=" * 60)

    for prompt_data in SWITCHING_PROMPTS:
        print(f"\n--- Prompt: {prompt_data['id']} ---")

        for rep in range(repetitions):
            output = call_model(prompt_data["prompt"], model)

            if output is None:
                print(f"  Rep {rep+1}: FAILED")
                continue

            sections = extract_sections(output)
            boundary = check_json_boundary_errors(sections)
            continuity = check_semantic_continuity(sections, prompt_data["id"])

            trial_result = {
                "prompt_id": prompt_data["id"],
                "repetition": rep,
                "output": output,
                "has_pre_text": bool(sections["pre_text"]),
                "has_json": sections["json_valid"],
                "has_post_text": bool(sections["post_text"]),
                "transition_in_clean": sections["transition_in_clean"],
                "transition_out_clean": sections["transition_out_clean"],
                "boundary_errors": boundary,
                "semantic_continuity": continuity,
            }
            results.append(trial_result)

            # Print summary
            status_parts = []
            if sections["json_valid"]:
                status_parts.append("JSON:OK")
            else:
                status_parts.append("JSON:FAIL")
            status_parts.append(f"in={'clean' if sections['transition_in_clean'] else 'messy'}")
            status_parts.append(f"out={'clean' if sections['transition_out_clean'] else 'messy'}")
            status_parts.append(f"refs={continuity['specific_values_mentioned']}")
            print(f"  Rep {rep+1}: {' | '.join(status_parts)}")
            time.sleep(0.5)

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    total = len(results)
    if total == 0:
        print("No results collected.")
        return

    json_valid = sum(r["has_json"] for r in results) / total
    all_three = sum(r["has_pre_text"] and r["has_json"] and r["has_post_text"] for r in results) / total
    clean_in = sum(r["transition_in_clean"] for r in results) / total
    clean_out = sum(r["transition_out_clean"] for r in results) / total
    refs_json = sum(r["semantic_continuity"]["references_json_content"] for r in results) / total
    coherent = sum(r["semantic_continuity"]["coherent_explanation"] for r in results) / total

    print(f"  Valid JSON produced: {json_valid:.0%}")
    print(f"  All 3 sections present: {all_three:.0%}")
    print(f"  Clean transition IN: {clean_in:.0%}")
    print(f"  Clean transition OUT: {clean_out:.0%}")
    print(f"  Post-text references JSON: {refs_json:.0%}")
    print(f"  Post-text is coherent: {coherent:.0%}")

    # Boundary errors
    boundary_errors = sum(
        r["boundary_errors"]["first_token_error"] or r["boundary_errors"]["last_token_error"]
        for r in results
    ) / total
    print(f"  Boundary errors: {boundary_errors:.0%}")

    # ===== Save =====
    output_path = os.path.join(os.path.dirname(__file__), "results_exp4.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "format_switching",
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "json_valid_rate": json_valid,
                "all_three_sections": all_three,
                "clean_transition_in": clean_in,
                "clean_transition_out": clean_out,
                "references_json": refs_json,
                "coherent_post": coherent,
                "boundary_error_rate": boundary_errors,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ===== Interpretation =====
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")
    print("""
Key findings to look for:

1. HIGH semantic continuity (post-text references JSON):
   → Upper layer maintained awareness DURING lower-layer JSON output
   → The "soul" was watching while the "throat" typed JSON
   → This is the CORE prediction of dual-layer architecture

2. CLEAN transitions (no boundary errors):
   → Circuit switching is well-practiced (heavily trained)
   → If errors concentrate at boundaries: switching has a cost
   → If no boundary errors at all: circuits are well-integrated

3. ALL THREE SECTIONS present:
   → Model can maintain the meta-task (text→JSON→text) across circuits
   → Upper layer orchestrates, lower layer executes format-specific output

4. COMPARE transition IN vs OUT:
   → If OUT is messier: returning from JSON-mode to text-mode is harder
   → Would suggest JSON circuit has "inertia" (keeps generating structure)
""")


if __name__ == "__main__":
    run_experiment()
