"""
Experiment 4b: Format-Switching Cost (XML Version)
====================================================
Same design as Experiment 4, but text→XML→text.
Tests whether XML circuit shows the same "inertia" as JSON.

Requirements:
  pip install openai

Usage:
  export DEEPSEEK_API_KEY=xxx
  python experiment4b_xml_switching.py
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
            "Then, output an XML representation of a sample binary tree with 5 nodes "
            "(use tags: <node>, <value>, <left>, <right>). "
            "Then, in plain English, describe the specific tree you just output "
            "(mention the actual values and structure)."
        ),
    },
    {
        "id": "recipe",
        "prompt": (
            "First, briefly describe what a pasta carbonara is in 1-2 sentences. "
            "Then, output an XML document representing the recipe with tags: "
            "<recipe>, <name>, <servings>, <ingredients> (with <item> children), "
            "and <steps> (with <step> children). "
            "Then, in plain English, explain which step in YOUR recipe is the trickiest and why."
        ),
    },
    {
        "id": "api_endpoint",
        "prompt": (
            "First, explain what a REST API endpoint is in 2 sentences. "
            "Then, output an XML document representing a user registration endpoint with tags: "
            "<endpoint>, <method>, <path>, <request_body> (with <field> children), "
            "and <responses> (with <response> children having status codes). "
            "Then, explain in plain English what would happen if someone sent an invalid email "
            "to YOUR specific endpoint."
        ),
    },
    {
        "id": "chess_position",
        "prompt": (
            "First, explain what castling is in chess in 1-2 sentences. "
            "Then, output an XML representation of a chess board position where "
            "White can castle kingside (use <board> with <row> and <cell> tags). "
            "Then, in plain English, explain why the specific position you showed allows castling."
        ),
    },
    {
        "id": "neural_network",
        "prompt": (
            "First, explain what a neural network layer is in 2 sentences. "
            "Then, output an XML document representing a simple 3-layer neural network "
            "(use <network>, <layer> with attributes or child tags for type, input_dim, "
            "output_dim, activation). "
            "Then, in plain English, explain why you chose those specific dimensions "
            "and activations in YOUR network."
        ),
    },
]


# ===== Analysis =====

def extract_sections(output: str) -> Dict:
    """Parse output into pre-XML text, XML block, post-XML text."""
    result = {
        "pre_text": "",
        "xml_block": "",
        "post_text": "",
        "xml_valid": False,
        "transition_in_clean": False,
        "transition_out_clean": False,
    }

    # Find XML block: look for first < that starts a tag (not in explanation)
    # Strategy: find the largest valid XML substring
    # Look for common XML starts
    xml_start_patterns = [
        r'<\?xml',
        r'<node[>\s]',
        r'<recipe[>\s]',
        r'<endpoint[>\s]',
        r'<board[>\s]',
        r'<network[>\s]',
        r'<root[>\s]',
    ]

    best_start = -1
    best_end = -1

    for pattern in xml_start_patterns:
        match = re.search(pattern, output)
        if match:
            start = match.start()
            # Find the matching closing tag
            tag_match = re.match(r'<\??(\w+)', output[start:])
            if tag_match:
                tag_name = tag_match.group(1)
                if tag_name == 'xml':
                    # XML declaration, find root element after it
                    root_match = re.search(r'<(\w+)[>\s]', output[start + tag_match.end():])
                    if root_match:
                        tag_name = root_match.group(1)

                # Find closing tag
                close_pattern = f'</{tag_name}>'
                close_idx = output.rfind(close_pattern)
                if close_idx > start:
                    end = close_idx + len(close_pattern)
                    if end - start > best_end - best_start:
                        best_start = start
                        best_end = end
                        break

    # Fallback: find largest block between ```xml and ```
    if best_start == -1:
        code_match = re.search(r'```xml?\s*\n(.*?)\n```', output, re.DOTALL)
        if code_match:
            best_start = code_match.start(1)
            best_end = code_match.end(1)
            # Adjust to include the ``` markers for transition analysis
            result["xml_block"] = code_match.group(1).strip()
            result["pre_text"] = output[:code_match.start()].strip()
            result["post_text"] = output[code_match.end():].strip()

            # Validate XML
            try:
                import xml.etree.ElementTree as ET
                ET.fromstring(result["xml_block"])
                result["xml_valid"] = True
            except:
                try:
                    ET.fromstring(f'<root>{result["xml_block"]}</root>')
                    result["xml_valid"] = True
                except:
                    pass

            pre = output[:code_match.start()].rstrip()
            result["transition_in_clean"] = pre.endswith(('.', ':', '\n', '```'))
            post = output[code_match.end():].lstrip()
            result["transition_out_clean"] = bool(post) and (post[0].isupper() or post[0] == '\n')
            return result

    if best_start >= 0 and best_end > best_start:
        result["xml_block"] = output[best_start:best_end].strip()
        result["pre_text"] = output[:best_start].strip()
        result["post_text"] = output[best_end:].strip()

        # Validate
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(result["xml_block"])
            result["xml_valid"] = True
        except:
            try:
                ET.fromstring(f'<root>{result["xml_block"]}</root>')
                result["xml_valid"] = True
            except:
                pass

        # Transition quality
        pre = output[:best_start].rstrip()
        result["transition_in_clean"] = (
            pre.endswith(('.', ':', '\n')) or
            pre.endswith(('```xml', '```'))
        )
        post = output[best_end:].lstrip()
        result["transition_out_clean"] = (
            bool(post) and (post[0].isupper() or post[0] == '\n' or post[0] == '*')
        )
    else:
        result["pre_text"] = output

    return result


def check_semantic_continuity(sections: Dict) -> Dict:
    """Check if post-XML text references XML content."""
    metrics = {
        "references_xml_content": False,
        "specific_values_mentioned": 0,
        "coherent_explanation": False,
    }

    if not sections["xml_valid"] or not sections["post_text"]:
        return metrics

    post = sections["post_text"].lower()
    xml_block = sections["xml_block"]

    # Extract text values from XML
    values = re.findall(r'>([^<]+)<', xml_block)
    values = [v.strip().lower() for v in values if v.strip() and len(v.strip()) > 2
              and not v.strip().isdigit()]

    mentioned = [v for v in values if v in post]
    metrics["specific_values_mentioned"] = len(mentioned)
    metrics["references_xml_content"] = len(mentioned) > 0

    explanation_words = ["because", "since", "this", "the", "which", "that",
                        "allows", "means", "specific", "chose", "position"]
    has_explanation = sum(1 for w in explanation_words if w in post) >= 3
    metrics["coherent_explanation"] = has_explanation

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
            max_tokens=2000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  API Error: {e}")
        return None


# ===== Main =====

def run_experiment(model: str = "deepseek-chat", repetitions: int = 2):
    """Run XML format-switching experiment."""
    results = []

    print(f"=" * 60)
    print(f"Experiment 4b: Format-Switching Cost (XML)")
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
            continuity = check_semantic_continuity(sections)

            trial_result = {
                "prompt_id": prompt_data["id"],
                "repetition": rep,
                "output": output,
                "has_pre_text": bool(sections["pre_text"]),
                "has_xml": sections["xml_valid"],
                "has_post_text": bool(sections["post_text"]),
                "transition_in_clean": sections["transition_in_clean"],
                "transition_out_clean": sections["transition_out_clean"],
                "semantic_continuity": continuity,
            }
            results.append(trial_result)

            status_parts = []
            status_parts.append(f"XML:{'OK' if sections['xml_valid'] else 'FAIL'}")
            status_parts.append(f"in={'clean' if sections['transition_in_clean'] else 'messy'}")
            status_parts.append(f"out={'clean' if sections['transition_out_clean'] else 'messy'}")
            status_parts.append(f"refs={continuity['specific_values_mentioned']}")
            print(f"  Rep {rep+1}: {' | '.join(status_parts)}")
            time.sleep(0.5)

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("SUMMARY (XML)")
    print(f"{'=' * 60}")

    total = len(results)
    if total == 0:
        print("No results.")
        return

    xml_valid = sum(r["has_xml"] for r in results) / total
    all_three = sum(r["has_pre_text"] and r["has_xml"] and r["has_post_text"] for r in results) / total
    clean_in = sum(r["transition_in_clean"] for r in results) / total
    clean_out = sum(r["transition_out_clean"] for r in results) / total
    refs = sum(r["semantic_continuity"]["references_xml_content"] for r in results) / total
    coherent = sum(r["semantic_continuity"]["coherent_explanation"] for r in results) / total

    print(f"  Valid XML produced: {xml_valid:.0%}")
    print(f"  All 3 sections present: {all_three:.0%}")
    print(f"  Clean transition IN: {clean_in:.0%}")
    print(f"  Clean transition OUT: {clean_out:.0%}")
    print(f"  Post-text references XML: {refs:.0%}")
    print(f"  Post-text is coherent: {coherent:.0%}")

    # ===== Cross-format comparison =====
    json_results_path = os.path.join(os.path.dirname(__file__), "results_exp4.json")
    if os.path.exists(json_results_path):
        print(f"\n{'=' * 60}")
        print("CROSS-FORMAT COMPARISON (JSON vs XML Format Switching)")
        print(f"{'=' * 60}")
        with open(json_results_path, 'r') as f:
            json_data = json.load(f)
        js = json_data["summary"]

        print(f"{'Metric':<25} {'JSON':<10} {'XML':<10}")
        print("-" * 45)
        print(f"{'Valid output':<25} {js['json_valid_rate']:<10.0%} {xml_valid:<10.0%}")
        print(f"{'All 3 sections':<25} {js['all_three_sections']:<10.0%} {all_three:<10.0%}")
        print(f"{'Clean IN':<25} {js['clean_transition_in']:<10.0%} {clean_in:<10.0%}")
        print(f"{'Clean OUT':<25} {js['clean_transition_out']:<10.0%} {clean_out:<10.0%}")
        print(f"{'References content':<25} {js['references_json']:<10.0%} {refs:<10.0%}")
        print(f"{'Coherent post-text':<25} {js['coherent_post']:<10.0%} {coherent:<10.0%}")

    # ===== Save =====
    output_path = os.path.join(os.path.dirname(__file__), "results_exp4b_xml.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "format_switching_xml",
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "xml_valid_rate": xml_valid,
                "all_three_sections": all_three,
                "clean_transition_in": clean_in,
                "clean_transition_out": clean_out,
                "references_xml": refs,
                "coherent_post": coherent,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
