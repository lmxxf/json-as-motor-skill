"""
Experiment 3: Attention Pattern Analysis
==========================================
Uses an open-source model to visualize attention patterns when
processing JSON vs Markdown vs plain text.

Prediction: JSON processing activates dedicated "structural attention heads"
that form long-range connections (bracket-to-bracket), which are absent
during Markdown/text processing of the same information.

Requirements:
  pip install transformers torch matplotlib numpy

Hardware: DGX Spark or GPU with 16GB+ VRAM (for 7B model)
  - DeepSeek-V2-Lite-Chat or LLaMA-3-8B recommended

Usage:
  python experiment3_attention_patterns.py --model deepseek-ai/DeepSeek-V2-Lite-Chat
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("WARNING: torch/transformers/matplotlib not installed.")
    print("Install with: pip install torch transformers matplotlib")
    print("This script will generate the analysis framework but cannot run.")


# ===== Stimuli =====

STIMULI = {
    "json": """{
  "employees": [
    {"name": "Alice", "dept": "Engineering", "projects": [{"title": "Atlas", "status": "active"}]},
    {"name": "Bob", "dept": "Marketing", "projects": [{"title": "Beacon", "status": "completed"}]},
    {"name": "Carol", "dept": "Design", "projects": [{"title": "Canvas", "status": "active"}]}
  ]
}""",

    "markdown": """## Employees
- **Alice** - Engineering
  - Project: Atlas (active)
- **Bob** - Marketing
  - Project: Beacon (completed)
- **Carol** - Design
  - Project: Canvas (active)""",

    "plaintext": """There are three employees. Alice works in Engineering on the Atlas project which is active. Bob works in Marketing on the Beacon project which is completed. Carol works in Design on the Canvas project which is active."""
}

# Structural token pairs to look for in JSON
JSON_STRUCTURAL_PAIRS = [
    ("{", "}"),
    ("[", "]"),
    (":", ","),  # key-value separator to next key
]


def get_attention_maps(model, tokenizer, text: str) -> Tuple[np.ndarray, List[str]]:
    """
    Get attention maps from all layers for a given text.
    Returns: (attention_tensor [layers, heads, seq, seq], token_list)
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Stack attention from all layers: [num_layers, num_heads, seq_len, seq_len]
    attentions = torch.stack(outputs.attentions).squeeze(1).cpu().numpy()
    return attentions, tokens


def find_structural_pairs(tokens: List[str]) -> List[Tuple[int, int]]:
    """Find matching bracket pairs in token list."""
    pairs = []
    stack_curly = []
    stack_square = []

    for i, token in enumerate(tokens):
        clean = token.strip().replace("Ġ", "").replace("▁", "")
        if "{" in clean:
            stack_curly.append(i)
        elif "}" in clean and stack_curly:
            pairs.append((stack_curly.pop(), i))
        elif "[" in clean:
            stack_square.append(i)
        elif "]" in clean and stack_square:
            pairs.append((stack_square.pop(), i))

    return pairs


def compute_long_range_attention(attentions: np.ndarray, min_distance: int = 20) -> np.ndarray:
    """
    Compute fraction of attention weight that connects tokens > min_distance apart.
    Returns: [num_layers, num_heads] array of long-range attention ratios.
    """
    num_layers, num_heads, seq_len, _ = attentions.shape
    ratios = np.zeros((num_layers, num_heads))

    for layer in range(num_layers):
        for head in range(num_heads):
            attn = attentions[layer, head]  # [seq, seq]
            total_weight = attn.sum()
            if total_weight == 0:
                continue

            # Create distance mask
            positions = np.arange(seq_len)
            distances = np.abs(positions[:, None] - positions[None, :])
            long_range_mask = distances > min_distance

            long_range_weight = (attn * long_range_mask).sum()
            ratios[layer, head] = long_range_weight / total_weight

    return ratios


def compute_structural_attention(attentions: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    """
    Compute how much attention weight flows between structural pairs.
    Returns: [num_layers, num_heads] array of structural attention scores.
    """
    num_layers, num_heads, seq_len, _ = attentions.shape
    scores = np.zeros((num_layers, num_heads))

    if not pairs:
        return scores

    for layer in range(num_layers):
        for head in range(num_heads):
            attn = attentions[layer, head]
            pair_attention = 0.0
            for (i, j) in pairs:
                if i < seq_len and j < seq_len:
                    pair_attention += attn[i, j] + attn[j, i]
            scores[layer, head] = pair_attention / (2 * len(pairs))

    return scores


def plot_comparison(results: Dict, output_dir: str):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (format_name, data) in enumerate(results.items()):
        lr_ratios = data["long_range_ratios"]
        im = axes[idx].imshow(lr_ratios, aspect="auto", cmap="hot", vmin=0, vmax=0.5)
        axes[idx].set_title(f"{format_name.upper()}\nLong-range attention ratio")
        axes[idx].set_xlabel("Head")
        axes[idx].set_ylabel("Layer")
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "long_range_attention_comparison.png"), dpi=150)
    plt.close()
    print(f"  Saved: long_range_attention_comparison.png")

    # Structural attention (JSON only)
    if "structural_scores" in results.get("json", {}):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(results["json"]["structural_scores"], aspect="auto", cmap="hot")
        ax.set_title("JSON: Structural Pair Attention")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "json_structural_attention.png"), dpi=150)
        plt.close()
        print(f"  Saved: json_structural_attention.png")


def run_experiment(model_name: str = "deepseek-ai/DeepSeek-V2-Lite-Chat"):
    """Run attention pattern analysis."""
    output_dir = os.path.dirname(os.path.abspath(__file__))

    if not HAS_DEPS:
        print("\nCannot run experiment without dependencies.")
        print("Install: pip install torch transformers matplotlib")
        print("Then run: python experiment3_attention_patterns.py")
        return

    print(f"=" * 60)
    print(f"Experiment 3: Attention Pattern Analysis")
    print(f"Model: {model_name}")
    print(f"=" * 60)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        output_attentions=True
    )
    model.eval()
    print(f"  Model loaded. Layers: {model.config.num_hidden_layers}, "
          f"Heads: {model.config.num_attention_heads}")

    results = {}

    for format_name, text in STIMULI.items():
        print(f"\n--- Processing: {format_name} ---")
        print(f"  Input length: {len(text)} chars")

        # Get attention maps
        attentions, tokens = get_attention_maps(model, tokenizer, text)
        print(f"  Tokens: {len(tokens)}")
        print(f"  Attention shape: {attentions.shape}")

        # Compute long-range attention ratios
        lr_ratios = compute_long_range_attention(attentions, min_distance=20)
        mean_lr = lr_ratios.mean()
        max_lr = lr_ratios.max()
        print(f"  Long-range attention: mean={mean_lr:.4f}, max={max_lr:.4f}")

        result = {
            "tokens": tokens,
            "long_range_ratios": lr_ratios,
            "mean_long_range": float(mean_lr),
            "max_long_range": float(max_lr),
        }

        # For JSON, also compute structural pair attention
        if format_name == "json":
            pairs = find_structural_pairs(tokens)
            print(f"  Structural pairs found: {len(pairs)}")
            struct_scores = compute_structural_attention(attentions, pairs)
            result["structural_scores"] = struct_scores
            result["structural_pairs"] = pairs
            result["mean_structural"] = float(struct_scores.mean())
            result["max_structural"] = float(struct_scores.max())
            print(f"  Structural attention: mean={struct_scores.mean():.4f}, max={struct_scores.max():.4f}")

            # Find "structural heads" (> 2x average structural attention)
            threshold = struct_scores.mean() + 2 * struct_scores.std()
            structural_heads = list(zip(*np.where(struct_scores > threshold)))
            print(f"  Structural heads (>2σ): {len(structural_heads)}")
            for (layer, head) in structural_heads[:10]:
                print(f"    Layer {layer}, Head {head}: score={struct_scores[layer, head]:.4f}")

        results[format_name] = result

    # ===== Comparison =====
    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Format':<12} {'Mean LR Attn':<15} {'Max LR Attn':<15}")
    print("-" * 42)
    for fmt, data in results.items():
        print(f"{fmt:<12} {data['mean_long_range']:<15.4f} {data['max_long_range']:<15.4f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_comparison(results, output_dir)

    # Save numerical results
    save_results = {}
    for fmt, data in results.items():
        save_results[fmt] = {
            "mean_long_range": data["mean_long_range"],
            "max_long_range": data["max_long_range"],
            "num_tokens": len(data["tokens"]),
        }
        if "mean_structural" in data:
            save_results[fmt]["mean_structural"] = data["mean_structural"]
            save_results[fmt]["max_structural"] = data["max_structural"]

    output_path = os.path.join(output_dir, "results_exp3.json")
    with open(output_path, "w") as f:
        json.dump({
            "experiment": "attention_patterns",
            "model": model_name,
            "timestamp": datetime.now().isoformat(),
            "results": save_results
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ===== Interpretation =====
    print(f"\n{'=' * 60}")
    print("INTERPRETATION")
    print(f"{'=' * 60}")
    print("""
Key findings to look for:

1. JSON should have HIGHER long-range attention than MD/text
   → Bracket-matching requires attending across long distances
   → Natural language is more local (adjacent words matter most)

2. "Structural heads" should exist ONLY for JSON
   → These heads specifically connect opening/closing brackets
   → If they don't activate for MD/text = dedicated circuit

3. Structural heads should concentrate in EARLY-MID layers
   → Not the final layers (which handle semantics)
   → This supports the lower-layer (automatic) assignment

4. If structural heads are CONTENT-INDEPENDENT:
   → Run JSON with different content, same structure
   → Same heads should activate → motor circuit confirmed
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite-Chat",
                       help="Model name/path (needs to fit in GPU memory)")
    args = parser.parse_args()
    run_experiment(args.model)
