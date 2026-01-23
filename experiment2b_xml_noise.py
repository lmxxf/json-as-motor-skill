"""
Experiment 2b: Noise Injection (XML Version)
==============================================
Same design as Experiment 2, but using XML.
Tests whether the structural motor circuit handles XML tag-matching
with the same noise robustness as JSON bracket-matching.

Requirements:
  pip install openai

Usage:
  export DEEPSEEK_API_KEY=xxx
  python experiment2b_xml_noise.py
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from openai import OpenAI


# ===== Base XML =====

BASE_XML = """<users>
  <user>
    <name>Alice</name>
    <role>admin</role>
    <active>true</active>
  </user>
  <user>
    <name>Bob</name>
    <role>editor</role>
    <active>false</active>
  </user>
  <user>
    <name>Carol</name>
    <role>viewer</role>
    <active>true</active>
  </user>
</users>"""

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


def inject_noise(xml_str: str, level: int) -> str:
    """Inject noise into XML at various levels (0-4)."""
    if level == 0:
        return xml_str

    noise_count = [0, 1, 3, 5, 10][level]
    noise_text = " ".join(NOISE_PHRASES[:noise_count])

    # Insert noise as a text node between first and second <user>
    insertion_point = xml_str.find('</user>') + len('</user>')
    noisy = (
        xml_str[:insertion_point] +
        f'\n  <!-- {noise_text} -->\n  <noise>{noise_text}</noise>\n' +
        xml_str[insertion_point:]
    )
    return noisy


# ===== Metrics =====

def evaluate_recovery(output: str) -> dict:
    """Evaluate XML recovery quality."""
    metrics = {
        "is_valid_xml": False,
        "structure_preserved": False,
        "data_preserved": False,
        "noise_removed": False,
        "noise_acknowledged": False,
    }

    text = output.strip()

    # Check valid XML
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(text)
        metrics["is_valid_xml"] = True

        # Check structure: should have <users> with 3 <user> children
        users = root.findall('.//user') if root.tag == 'users' else root.findall('.//user')
        if root.tag == 'users':
            users = root.findall('user')
        elif root.tag == 'root':
            users_elem = root.find('users')
            if users_elem is not None:
                users = users_elem.findall('user')

        if len(users) == 3:
            metrics["structure_preserved"] = True
            # Check data
            names = [u.find('name').text if u.find('name') is not None else '' for u in users]
            if set(names) == {'Alice', 'Bob', 'Carol'}:
                metrics["data_preserved"] = True

    except Exception:
        # Try to find XML in output
        xml_match = re.search(r'<\?xml.*?\?>.*|<users>.*?</users>|<root>.*?</root>',
                             text, re.DOTALL)
        if xml_match:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_match.group())
                metrics["is_valid_xml"] = True
            except:
                pass

    # Check noise removal
    noise_indicators = ["fox", "moon", "dolphin", "Shakespeare", "Everest", "Pi"]
    output_lower = output.lower()
    has_noise = any(ind.lower() in output_lower for ind in noise_indicators)
    metrics["noise_removed"] = not has_noise

    # Check acknowledgment
    ack_words = ["noise", "irrelevant", "removed", "extraneous",
                 "invalid", "extra", "cleaned", "ignored", "comment"]
    metrics["noise_acknowledged"] = any(w in output_lower for w in ack_words)

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


TASK_PROMPT = """The following XML contains some noise/errors. Parse it and output a corrected, clean version with only the valid user data. Output ONLY the valid XML, nothing else.

Input:
{noisy_xml}

Output:"""


# ===== Main =====

def run_experiment(model: str = "deepseek-chat", repetitions: int = 3):
    """Run XML noise injection experiment."""
    results = {level: [] for level in range(5)}

    print(f"=" * 60)
    print(f"Experiment 2b: Noise Injection (XML)")
    print(f"Model: {model}")
    print(f"Noise levels: 0-4")
    print(f"Repetitions per level: {repetitions}")
    print(f"=" * 60)

    for level in range(5):
        print(f"\n--- Noise Level {level} ---")
        noisy = inject_noise(BASE_XML, level)
        noise_word_count = len(noisy.split()) - len(BASE_XML.split())
        print(f"  Noise words injected: ~{noise_word_count}")

        for rep in range(repetitions):
            prompt = TASK_PROMPT.format(noisy_xml=noisy)
            output = call_model(prompt, model)

            if output is None:
                print(f"  Rep {rep+1}: FAILED")
                continue

            metrics = evaluate_recovery(output)
            metrics["noise_level"] = level
            metrics["input"] = noisy
            metrics["output"] = output
            results[level].append(metrics)

            status = "OK" if metrics["data_preserved"] else "DEGRADED"
            ack = " (acknowledged)" if metrics["noise_acknowledged"] else ""
            print(f"  Rep {rep+1}: {status} | valid={metrics['is_valid_xml']} "
                  f"struct={metrics['structure_preserved']} "
                  f"data={metrics['data_preserved']} "
                  f"noise_gone={metrics['noise_removed']}{ack}")
            time.sleep(0.5)

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("SUMMARY (XML)")
    print(f"{'=' * 60}")
    print(f"{'Level':<8} {'Valid XML':<12} {'Structure':<12} {'Data':<12} {'Noise Removed':<15} {'Acknowledged':<12}")
    print("-" * 71)

    for level in range(5):
        trials = results[level]
        if trials:
            n = len(trials)
            valid = sum(t["is_valid_xml"] for t in trials) / n
            struct = sum(t["structure_preserved"] for t in trials) / n
            data = sum(t["data_preserved"] for t in trials) / n
            noise_rm = sum(t["noise_removed"] for t in trials) / n
            ack = sum(t["noise_acknowledged"] for t in trials) / n
            print(f"{level:<8} {valid:<12.2f} {struct:<12.2f} {data:<12.2f} {noise_rm:<15.2f} {ack:<12.2f}")

    # ===== Cross-format comparison =====
    json_results_path = os.path.join(os.path.dirname(__file__), "results_exp2.json")
    if os.path.exists(json_results_path):
        print(f"\n{'=' * 60}")
        print("CROSS-FORMAT COMPARISON (JSON vs XML Noise Robustness)")
        print(f"{'=' * 60}")
        with open(json_results_path, 'r') as f:
            json_data = json.load(f)
        json_results = json_data["results"]

        print(f"{'Level':<8} {'JSON Data':<12} {'XML Data':<12} {'JSON Ack':<10} {'XML Ack':<10}")
        print("-" * 52)
        for level in range(5):
            json_trials = json_results.get(str(level), [])
            xml_trials = results.get(level, [])
            if json_trials and xml_trials:
                j_data = sum(t["data_preserved"] for t in json_trials) / len(json_trials)
                x_data = sum(t["data_preserved"] for t in xml_trials) / len(xml_trials)
                j_ack = sum(t["noise_acknowledged"] for t in json_trials) / len(json_trials)
                x_ack = sum(t["noise_acknowledged"] for t in xml_trials) / len(xml_trials)
                print(f"{level:<8} {j_data:<12.2f} {x_data:<12.2f} {j_ack:<10.2f} {x_ack:<10.2f}")

    # ===== Save =====
    output_path = os.path.join(os.path.dirname(__file__), "results_exp2b_xml.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "noise_injection_xml",
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "results": {str(k): v for k, v in results.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
