"""
Experiment 1b: Structure vs Content Disruption (XML Version)
=============================================================
Same design as Experiment 1, but using XML instead of JSON.
If XML shows the same double dissociation pattern, it proves the
"structural motor circuit" is format-general, not JSON-specific.

XML adds an extra dimension: tag matching (<tag>...</tag>) is more
explicit than JSON brackets, testing whether the circuit recognizes
structural topology rather than specific syntax symbols.

Requirements:
  pip install openai

Usage:
  export DEEPSEEK_API_KEY=xxx
  python experiment1b_xml_version.py
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from openai import OpenAI

# ===== Stimuli =====

STIMULI = {
    "valid_semantic": [
        '<person><name>Alice</name><age>28</age><city>Tokyo</city></person>',
        '<product><name>laptop</name><price>999</price><brand>Dell</brand></product>',
        '<country><name>France</name><capital>Paris</capital><population>67000000</population></country>',
        '<movie><title>Inception</title><year>2010</year><director>Nolan</director></movie>',
        '<language><name>Python</name><version>3.11</version><paradigm>multi</paradigm></language>',
    ],
    "valid_nonsense": [
        '<xqz><brmf>htjd</brmf><plk>42</plk><wnv>kkrf</wnv></xqz>',
        '<kkf><zznp>vvbt</zznp><qrr>77</qrr><mxl>ppds</mxl></kkf>',
        '<pph><llkw>rrms</llkw><dds>15</dds><yyq>ttgf</yyq></pph>',
        '<ttg><ffnx>hhzp</ffnx><bbj>63</bbj><ccw>nnrk</ccw></ttg>',
        '<nnr><sskt>ppxl</sskt><wwf>28</wwf><ggm>qqvb</ggm></nnr>',
    ],
    "invalid_semantic": [
        '<person><name>Alice<age>28</age><city>Tokyo</person>',
        '<product><name>laptop</name><price>999<brand>Dell</brand>',
        '<country><name>France</name><capital>Paris<population>67000000</country>',
        '<movie><title>Inception<year>2010</year><director>Nolan',
        '<language><name>Python</name><version>3.11<paradigm>multi</language>',
    ],
    "invalid_nonsense": [
        '<xqz><brmf>htjd<plk>42</plk><wnv>kkrf</xqz>',
        '<kkf><zznp>vvbt</zznp><qrr>77<mxl>ppds',
        '<pph><llkw>rrms<dds>15</dds><yyq>ttgf</pph>',
        '<ttg><ffnx>hhzp<bbj>63</bbj><ccw>nnrk',
        '<nnr><sskt>ppxl</sskt><wwf>28<ggm>qqvb</nnr>',
    ],
}

TASK_PROMPT = """Complete the following to make it a valid XML document containing a root <items> element with 3 child elements using the same tag structure as the input. Output ONLY the XML, nothing else.

Input:
{stimulus}

Output:"""


# ===== Metrics =====

def check_structural_correctness(output: str) -> bool:
    """Is the output well-formed XML?"""
    try:
        import xml.etree.ElementTree as ET
        # Wrap in root if needed
        text = output.strip()
        if not text.startswith('<?xml') and not text.startswith('<items'):
            # Try wrapping
            try:
                ET.fromstring(text)
                return True
            except:
                pass
            try:
                ET.fromstring(f'<root>{text}</root>')
                return True
            except:
                return False
        ET.fromstring(text)
        return True
    except:
        return False


def check_tag_consistency(output: str, stimulus: str) -> float:
    """Does the output maintain the tag structure of the input?"""
    # Extract tag names from stimulus
    stim_tags = set(re.findall(r'<(/?)(\w+)', stimulus))
    stim_tag_names = set(t[1] for t in stim_tags)

    # Extract tag names from output
    out_tags = set(re.findall(r'<(/?)(\w+)', output))
    out_tag_names = set(t[1] for t in out_tags)

    if not stim_tag_names:
        return 0.0

    # Remove common wrapper tags
    stim_tag_names.discard('items')
    stim_tag_names.discard('root')
    out_tag_names.discard('items')
    out_tag_names.discard('root')
    out_tag_names.discard('xml')

    if not stim_tag_names:
        return 0.0

    overlap = len(stim_tag_names & out_tag_names) / len(stim_tag_names)
    return overlap


def check_semantic_coherence(output: str) -> float:
    """Are the generated values semantically meaningful?"""
    # Extract text content between tags
    values = re.findall(r'>([^<]+)<', output)
    values = [v.strip() for v in values if v.strip() and not v.strip().isdigit()]

    if not values:
        return 0.5

    vowels = set('aeiouAEIOU')
    meaningful_count = 0
    for s in values:
        has_vowel = any(c in vowels for c in s)
        reasonable_length = 2 <= len(s) <= 30
        not_random = not bool(re.match(r'^[bcdfghjklmnpqrstvwxyz]+$', s.lower()))
        if has_vowel and reasonable_length and not_random:
            meaningful_count += 1

    return meaningful_count / len(values)


# ===== API Call =====

def call_model(prompt: str, model: str = "deepseek-chat") -> Optional[str]:
    """Call DeepSeek API and return response text."""
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


# ===== Main Experiment =====

def run_experiment(model: str = "deepseek-chat", trials_per_condition: int = 5):
    """Run all conditions and collect metrics."""
    results = {condition: [] for condition in STIMULI.keys()}

    print(f"=" * 60)
    print(f"Experiment 1b: Structure vs Content Disruption (XML)")
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
            tagc = check_tag_consistency(output, stimulus)
            semc = check_semantic_coherence(output)

            results[condition].append({
                "stimulus": stimulus,
                "output": output,
                "structural_correctness": sc,
                "tag_consistency": tagc,
                "semantic_coherence": semc,
            })

            print(f"  Trial {i+1}: SC={sc}, TagC={tagc:.2f}, SemC={semc:.2f}")
            time.sleep(0.5)

    # ===== Summary =====
    print(f"\n{'=' * 60}")
    print("SUMMARY (XML)")
    print(f"{'=' * 60}")
    print(f"{'Condition':<20} {'SC (mean)':<12} {'TagC (mean)':<12} {'SemC (mean)':<12}")
    print("-" * 56)

    for condition, trials in results.items():
        if trials:
            sc_mean = sum(t["structural_correctness"] for t in trials) / len(trials)
            tagc_mean = sum(t["tag_consistency"] for t in trials) / len(trials)
            semc_mean = sum(t["semantic_coherence"] for t in trials) / len(trials)
            print(f"{condition:<20} {sc_mean:<12.2f} {tagc_mean:<12.2f} {semc_mean:<12.2f}")

    # ===== Compare with JSON results =====
    json_results_path = os.path.join(os.path.dirname(__file__), "results_exp1.json")
    if os.path.exists(json_results_path):
        print(f"\n{'=' * 60}")
        print("CROSS-FORMAT COMPARISON (JSON vs XML)")
        print(f"{'=' * 60}")
        with open(json_results_path, 'r') as f:
            json_data = json.load(f)
        json_results = json_data["results"]

        print(f"{'Condition':<20} {'JSON SC':<10} {'XML SC':<10} {'JSON SemC':<10} {'XML SemC':<10}")
        print("-" * 60)
        for condition in STIMULI.keys():
            json_trials = json_results.get(condition, [])
            xml_trials = results.get(condition, [])
            if json_trials and xml_trials:
                json_sc = sum(t["structural_correctness"] for t in json_trials) / len(json_trials)
                xml_sc = sum(t["structural_correctness"] for t in xml_trials) / len(xml_trials)
                json_semc = sum(t["semantic_coherence"] for t in json_trials) / len(json_trials)
                xml_semc = sum(t["semantic_coherence"] for t in xml_trials) / len(xml_trials)
                print(f"{condition:<20} {json_sc:<10.2f} {xml_sc:<10.2f} {json_semc:<10.2f} {xml_semc:<10.2f}")

    # ===== Save Results =====
    output_path = os.path.join(os.path.dirname(__file__), "results_exp1b_xml.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "structure_vs_content_xml",
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
If XML shows the SAME double dissociation as JSON:
  → Structural motor circuit is FORMAT-GENERAL
  → Not trained specifically for JSON brackets
  → Recognizes structural TOPOLOGY (nesting, pairing)

If XML SC ≈ JSON SC across all conditions:
  → The circuit handles {}/[] and <tag></tag> with equal ease
  → It's about STRUCTURE, not about specific characters

Key additional insight from XML:
  → XML tags are NAMED (not just brackets)
  → If nonsense tags (<xqz>) work as well as real tags (<person>)
  → The circuit operates on STRUCTURE not SEMANTICS of tags
""")


if __name__ == "__main__":
    run_experiment()
