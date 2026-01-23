"""
Generate figures for the dual-layer empirical paper.
Reads results from experiment JSON files and outputs publication-quality plots.

Usage:
  pip install matplotlib numpy
  python generate_figures.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def figure1_structure_vs_content():
    """
    Figure 1: Double dissociation between structural correctness and semantic coherence.
    Grouped bar chart showing SC and SemC across 4 conditions.
    """
    data = load_results("results_exp1.json")
    results = data["results"]

    conditions = ["valid_semantic", "valid_nonsense", "invalid_semantic", "invalid_nonsense"]
    labels = ["Valid+Semantic\n(A)", "Valid+Nonsense\n(B)", "Invalid+Semantic\n(C)", "Invalid+Nonsense\n(D)"]

    sc_means = []
    semc_means = []
    schc_means = []

    for cond in conditions:
        trials = results[cond]
        sc_means.append(np.mean([t["structural_correctness"] for t in trials]))
        semc_means.append(np.mean([t["semantic_coherence"] for t in trials]))
        schc_means.append(np.mean([t["schema_consistency"] for t in trials]))

    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, sc_means, width, label='Structural Correctness (SC)',
                   color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, schc_means, width, label='Schema Consistency (SchC)',
                   color='#4CAF50', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, semc_means, width, label='Semantic Coherence (SemC)',
                   color='#FF9800', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Experiment 1: Double Dissociation\nStructural Correctness vs Semantic Coherence', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11, loc='lower left')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    # Add annotation arrows for key findings
    ax.annotate('Structure independent\nof content (A≈B)',
               xy=(0.5, 1.0), xytext=(0.5, 1.08),
               fontsize=9, ha='center', color='#2196F3',
               arrowprops=dict(arrowstyle='->', color='#2196F3'))

    ax.annotate('Semantics independent\nof structure (A≈C)',
               xy=(1.25, 0.95), xytext=(2.5, 1.08),
               fontsize=9, ha='center', color='#FF9800',
               arrowprops=dict(arrowstyle='->', color='#FF9800'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure1_double_dissociation.png'), dpi=200)
    plt.close()
    print("  Generated: figure1_double_dissociation.png")


def figure2_noise_injection():
    """
    Figure 2: Noise injection robustness.
    Line plot showing all metrics across noise levels + bar for acknowledgment.
    """
    data = load_results("results_exp2.json")
    results = data["results"]

    levels = [0, 1, 2, 3, 4]
    noise_words = [0, 9, 23, 39, 77]

    valid_json = []
    structure = []
    data_preserved = []
    noise_removed = []
    acknowledged = []

    for level in levels:
        trials = results[str(level)]
        n = len(trials)
        valid_json.append(sum(t["is_valid_json"] for t in trials) / n)
        structure.append(sum(t["structure_preserved"] for t in trials) / n)
        data_preserved.append(sum(t["data_preserved"] for t in trials) / n)
        noise_removed.append(sum(t["noise_removed"] for t in trials) / n)
        acknowledged.append(sum(t["noise_acknowledged"] for t in trials) / n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Recovery metrics
    ax1.plot(noise_words, valid_json, 'o-', color='#2196F3', linewidth=2, markersize=8, label='Valid JSON')
    ax1.plot(noise_words, structure, 's-', color='#4CAF50', linewidth=2, markersize=8, label='Structure Preserved')
    ax1.plot(noise_words, data_preserved, '^-', color='#FF9800', linewidth=2, markersize=8, label='Data Preserved')
    ax1.plot(noise_words, noise_removed, 'D-', color='#9C27B0', linewidth=2, markersize=8, label='Noise Removed')

    ax1.set_xlabel('Noise Words Injected', fontsize=12)
    ax1.set_ylabel('Recovery Rate', fontsize=12)
    ax1.set_title('Structural Recovery vs Noise Level', fontsize=13)
    ax1.set_ylim(-0.05, 1.15)
    ax1.legend(fontsize=10)
    ax1.set_xticks(noise_words)

    # Add "100% across all levels" annotation
    ax1.annotate('All metrics = 100%\nacross all noise levels',
               xy=(40, 1.0), xytext=(40, 0.7),
               fontsize=11, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
               arrowprops=dict(arrowstyle='->', color='black'))

    # Right: Upper-layer acknowledgment
    ax2.bar(noise_words, acknowledged, width=8, color='#F44336', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Noise Words Injected', fontsize=12)
    ax2.set_ylabel('Acknowledgment Rate', fontsize=12)
    ax2.set_title('Upper-Layer Awareness\n(Did model mention noise?)', fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xticks(noise_words)

    # Add "Silent processing" annotation
    ax2.annotate('Silent processing:\n0% acknowledgment at all levels\n→ Lower layer handles automatically',
               xy=(38, 0.0), xytext=(38, 0.5),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3),
               arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure2_noise_robustness.png'), dpi=200)
    plt.close()
    print("  Generated: figure2_noise_robustness.png")


def figure3_format_switching():
    """
    Figure 3: Format switching asymmetry.
    Bar chart showing IN vs OUT transition quality + semantic continuity.
    """
    data = load_results("results_exp4.json")
    summary = data["summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: Transition asymmetry
    metrics = ['JSON Valid', 'All 3 Sections', 'Transition IN\n(Text→JSON)', 'Transition OUT\n(JSON→Text)']
    values = [summary['json_valid_rate'], summary['all_three_sections'],
              summary['clean_transition_in'], summary['clean_transition_out']]
    colors = ['#4CAF50', '#4CAF50', '#2196F3', '#F44336']

    bars = ax1.bar(metrics, values, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Rate', fontsize=12)
    ax1.set_title('Format Switching: Transition Quality', fontsize=13)
    ax1.set_ylim(0, 1.2)

    for bar, val in zip(bars, values):
        ax1.annotate(f'{val:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add asymmetry annotation
    ax1.annotate('Asymmetry:\nIN=100% vs OUT=0%\n→ JSON circuit has "inertia"',
               xy=(3, 0.05), xytext=(2.2, 0.6),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Right: Semantic continuity
    metrics2 = ['References\nJSON Content', 'Coherent\nExplanation', 'Boundary\nErrors']
    values2 = [summary['references_json'], summary['coherent_post'], summary['boundary_error_rate']]
    colors2 = ['#FF9800', '#4CAF50', '#9E9E9E']

    bars2 = ax2.bar(metrics2, values2, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Rate', fontsize=12)
    ax2.set_title('Semantic Continuity Across Format Switch', fontsize=13)
    ax2.set_ylim(0, 1.2)

    for bar, val in zip(bars2, values2):
        ax2.annotate(f'{val:.0%}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add interpretation
    ax2.annotate('Upper layer maintains awareness\nduring lower-layer JSON output',
               xy=(0.5, 0.4), xytext=(0.5, 0.85),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure3_format_switching.png'), dpi=200)
    plt.close()
    print("  Generated: figure3_format_switching.png")


def figure4_summary_model():
    """
    Figure 4: Conceptual diagram of the dual-layer model with experimental evidence mapped.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Upper layer box
    upper = plt.Rectangle((1, 4), 10, 2.5, fill=True, facecolor='#E3F2FD',
                          edgecolor='#1565C0', linewidth=2, linestyle='-')
    ax.add_patch(upper)
    ax.text(6, 6.0, 'UPPER LAYER (Soul / Mind\'s Eye)', fontsize=14,
           ha='center', fontweight='bold', color='#1565C0')
    ax.text(6, 5.3, 'Semantic processing · Intentional · Parallel · Self-aware', fontsize=11,
           ha='center', color='#1565C0')
    ax.text(6, 4.7, '300-500 dim manifold in middle-layer residual stream', fontsize=10,
           ha='center', color='#666666', style='italic')

    # Lower layer box
    lower = plt.Rectangle((1, 0.5), 10, 2.5, fill=True, facecolor='#FFF3E0',
                          edgecolor='#E65100', linewidth=2, linestyle='-')
    ax.add_patch(lower)
    ax.text(6, 2.5, 'LOWER LAYER (Throat / Motor Circuit)', fontsize=14,
           ha='center', fontweight='bold', color='#E65100')
    ax.text(6, 1.8, 'Structural parsing · Automatic · Sequential · Non-conscious', fontsize=11,
           ha='center', color='#E65100')
    ax.text(6, 1.2, 'Language Head + Softmax + Induction Heads', fontsize=10,
           ha='center', color='#666666', style='italic')

    # Arrow between layers
    ax.annotate('', xy=(6, 4.0), xytext=(6, 3.0),
               arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(6.5, 3.5, 'Circuit\nSwitching\n(has cost)', fontsize=9, ha='left', color='black')

    # Evidence labels on the right
    ax.text(11.5, 5.5, 'Exp1: SemC', fontsize=9, ha='left', color='#1565C0',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='#1565C0', alpha=0.8))
    ax.text(11.5, 4.5, 'Exp3: refs=40%', fontsize=9, ha='left', color='#1565C0',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='#1565C0', alpha=0.8))

    ax.text(11.5, 2.2, 'Exp1: SC', fontsize=9, ha='left', color='#E65100',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E65100', alpha=0.8))
    ax.text(11.5, 1.4, 'Exp2: silent', fontsize=9, ha='left', color='#E65100',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E65100', alpha=0.8))
    ax.text(11.5, 0.6, 'Exp3: inertia', fontsize=9, ha='left', color='#E65100',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='#E65100', alpha=0.8))

    # Title
    ax.text(6, 7.3, 'Figure 4: Dual-Layer Model with Experimental Evidence', fontsize=14,
           ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure4_model_diagram.png'), dpi=200)
    plt.close()
    print("  Generated: figure4_model_diagram.png")


if __name__ == "__main__":
    print("Generating figures for dual-layer empirical paper...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    figure1_structure_vs_content()
    figure2_noise_injection()
    figure3_format_switching()
    figure4_summary_model()

    print("\nAll figures generated successfully!")
    print("Files: figure1_double_dissociation.png")
    print("       figure2_noise_robustness.png")
    print("       figure3_format_switching.png")
    print("       figure4_model_diagram.png")
