import matplotlib.pyplot as plt
import numpy as np

# Data from results_exp3 JSONs
models = ['Llama-3.3-70B', 'Qwen2.5-72B']
formats = ['JSON', 'Markdown', 'Plaintext']

# [model][format]
data = {
    'Llama-3.3-70B': [0.7197, 0.5036, 0.4373],
    'Qwen2.5-72B': [0.6685, 0.4406, 0.3756],
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Long-range attention by format
x = np.arange(len(formats))
width = 0.35
for i, model in enumerate(models):
    axes[0].bar(x + i*width, data[model], width, label=model)
axes[0].set_ylabel('Long-Range Attention Ratio')
axes[0].set_title('Long-Range Attention: JSON vs Markdown vs Plaintext')
axes[0].set_xticks(x + width/2)
axes[0].set_xticklabels(formats)
axes[0].legend()
axes[0].set_ylim(0, 0.85)
for i, model in enumerate(models):
    for j, v in enumerate(data[model]):
        axes[0].text(x[j] + i*width, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)

# Plot 2: Structural head stats
struct_data = {
    'Llama-3.3-70B': {'count': 149, 'pct': 2.9, 'max_score': 0.1558, 'layer_range': '0-5'},
    'Qwen2.5-72B': {'count': 86, 'pct': 1.7, 'max_score': 0.2808, 'layer_range': '3-14'},
}

categories = ['Head Count', 'Max Score (×100)']
llama_vals = [149, 0.1558 * 100]
qwen_vals = [86, 0.2808 * 100]

x2 = np.arange(len(categories))
axes[1].bar(x2, llama_vals, width, label='Llama-3.3-70B')
axes[1].bar(x2 + width, qwen_vals, width, label='Qwen2.5-72B')
axes[1].set_title('Structural Heads: Count vs Intensity')
axes[1].set_xticks(x2 + width/2)
axes[1].set_xticklabels(categories)
axes[1].legend()
axes[1].text(0, 149 + 2, '149 (2.9%)\nL0-5', ha='center', fontsize=8)
axes[1].text(0 + width, 86 + 2, '86 (1.7%)\nL3-14', ha='center', fontsize=8)
axes[1].text(1, 15.58 + 1, '0.156', ha='center', fontsize=8)
axes[1].text(1 + width, 28.08 + 1, '0.281', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('/home/lmxxf/work/ai-theorys-study/arxiv/paper65/figure5_cross_model_comparison.png', dpi=150)
print("Saved figure5_cross_model_comparison.png")
