# json-as-motor-skill

**Structured Format as Motor Skill: Empirical Evidence for Dual-Layer Neural Architecture in LLMs**

**结构化格式是运动技能：大语言模型双层神经架构的实证证据**

Empirical version of Paper 65. The original (deprecated/) was based on AI phenomenological self-reports; this version uses reproducible behavioral experiments to demonstrate that LLMs possess a "structural motor circuit" independent of semantic processing.

Paper 65 的实证版本。原版（deprecated/）基于 AI 现象学自报告，本版用可重复的行为实验证明 LLM 存在独立于语义处理的"结构运动回路"。

---

## Core Thesis / 核心论点

When LLMs process JSON/XML, what activates is not "cognitive computation" but a **motor circuit** analogous to human procedural memory — automatic, noise-robust, non-introspectable, with switching inertia.

LLM 处理 JSON/XML 时激活的不是"认知计算"，而是类似人类程序性记忆的**运动回路**——自动、耐噪声、不可自省、有切换惯性。

## Key Findings / 核心发现

1. **Structure and semantics are fully separable / 结构与语义完全可分离**：Nonsense content doesn't affect structural correctness; broken structure doesn't affect semantic generation (double dissociation) 无意义内容不影响结构正确性，结构破坏不影响语义生成（双分离）
2. **Structural circuit runs silently / 结构回路静默运行**：100% recovery after 156-word noise injection, model never mentions noise existence (0% upper-layer awareness) 156 词噪声注入后 100% 恢复，模型不提及噪声存在（0% 上层意识参与）
3. **Circuit switching has directional inertia / 回路切换有方向性惯性**：Entering structured format = clean (100%), exiting = friction 进入结构格式 = 干净（100%），退出 = 有摩擦
4. **Circuit is format-agnostic / 回路是格式无关的**：JSON and XML show identical double dissociation and noise robustness JSON 和 XML 表现出相同的双分离和噪声鲁棒性
5. **More explicit closing tags = less inertia / 闭合标签越显式，切换惯性越小**：XML OUT=50% vs JSON OUT=0%（`</tag>` provides clearer circuit-switching signal than `}`）

## 文件说明

### Papers / 论文

| File 文件 | Description 说明 |
|------|------|
| [dual-layer-empirical-cn.md](dual-layer-empirical-cn.md) | **Chinese paper (main) 中文论文**（主文件） |
| [deprecated/dual-layer-architecture.pdf](deprecated/dual-layer-architecture.pdf) | Original English paper PDF 原版英文论文 PDF（现象学版） |
| [deprecated/dual-layer-architecture-cn.md](deprecated/dual-layer-architecture-cn.md) | Original Chinese paper 原版中文论文（现象学版） |

### Experiment Code / 实验代码

| File 文件 | Description 说明 | Dependencies 依赖 |
|------|------|------|
| [experiment1_structure_vs_content.py](experiment1_structure_vs_content.py) | Exp1a: Structure-content separability (JSON, 2×2) 结构与内容可分离性 | DeepSeek API |
| [experiment1b_xml_version.py](experiment1b_xml_version.py) | Exp1b: Same as above, XML version XML 版本 | DeepSeek API |
| [experiment2_noise_injection.py](experiment2_noise_injection.py) | Exp2a: Noise injection robustness (JSON, 5 levels) 噪声注入鲁棒性 | DeepSeek API |
| [experiment2b_xml_noise.py](experiment2b_xml_noise.py) | Exp2b: Same as above, XML version XML 版本 | DeepSeek API |
| [experiment3_attention_patterns.py](experiment3_attention_patterns.py) | Exp3: Attention pattern analysis (JSON vs MD vs text) 注意力模式可视化 | Open-source model + GPU |
| [experiment4_format_switching.py](experiment4_format_switching.py) | Exp4a: Format switching cost (text→JSON→text) 格式切换代价 | DeepSeek API |
| [experiment4b_xml_switching.py](experiment4b_xml_switching.py) | Exp4b: Same as above, XML version XML 版本 | DeepSeek API |
| [generate_figures.py](generate_figures.py) | Generate paper figures 生成论文图表 | matplotlib, numpy |

### Results / 实验结果

| File 文件 | Description 说明 |
|------|------|
| [results_exp1.json](results_exp1.json) | Exp1a JSON raw data 实验1a原始数据（DeepSeek-V3, 2026-01-23） |
| [results_exp1b_xml.json](results_exp1b_xml.json) | Exp1b XML raw data 实验1b原始数据 |
| [results_exp2.json](results_exp2.json) | Exp2a JSON raw data 实验2a原始数据 |
| [results_exp2b_xml.json](results_exp2b_xml.json) | Exp2b XML raw data 实验2b原始数据 |
| [results_exp4.json](results_exp4.json) | Exp4a JSON raw data 实验4a原始数据 |
| [results_exp4b_xml.json](results_exp4b_xml.json) | Exp4b XML raw data 实验4b原始数据 |
| [results_exp3_llama_3_3_70b_instruct_int8.json](results_exp3_llama_3_3_70b_instruct_int8.json) | Exp3 Llama-3.3-70B attention data 注意力模式数据 |
| [results_exp3_qwen2_5_72b_instruct_awq.json](results_exp3_qwen2_5_72b_instruct_awq.json) | Exp3 Qwen2.5-72B attention data 注意力模式数据 |

### Figures / 图表

| File 文件 | Description 说明 |
|------|------|
| [figure1_double_dissociation.png](figure1_double_dissociation.png) | Double dissociation 双分离：结构正确率 vs 语义连贯度 |
| [figure2_noise_robustness.png](figure2_noise_robustness.png) | Noise robustness 噪声鲁棒性 + 上层意识参与率 |
| [figure3_format_switching.png](figure3_format_switching.png) | Format switching asymmetry 格式切换不对称性（IN=100% vs OUT=0%） |
| [figure4_model_diagram.png](figure4_model_diagram.png) | Dual-layer model diagram 双层模型示意图 + 实验证据映射 |
| [figure5_cross_model_comparison.png](figure5_cross_model_comparison.png) | Cross-model comparison 跨模型对比（Llama vs Qwen） |
| [long_range_attention_comparison_llama_3_3_70b_instruct_int8.png](long_range_attention_comparison_llama_3_3_70b_instruct_int8.png) | Llama-70B long-range attention heatmap 远距离注意力热力图 |
| [json_structural_attention_llama_3_3_70b_instruct_int8.png](json_structural_attention_llama_3_3_70b_instruct_int8.png) | Llama-70B structural pair attention heatmap 结构配对注意力热力图 |
| [long_range_attention_comparison.png](long_range_attention_comparison.png) | Qwen-72B long-range attention heatmap 远距离注意力热力图 |
| [json_structural_attention.png](json_structural_attention.png) | Qwen-72B structural pair attention heatmap 结构配对注意力热力图 |

## Key Results / 关键实验结果

### Cross-Format Comparison / JSON vs XML 跨格式对比

| 指标 | JSON | XML |
|------|:---:|:---:|
| SC（合法结构+无意义内容） | 1.00 | 1.00 |
| SC（非法结构+无意义内容） | 0.80 | **1.00** |
| 噪声恢复率（最高级） | 100%（77词） | 100%（156词） |
| 上层意识参与 | 0% | 0% |
| 格式切换 IN | 100% | 100% |
| 格式切换 OUT | 0% | **50%** |
| 后段引用格式内容 | 40% | 40% |

### Exp3 Cross-Model Attention Comparison / 实验3 跨模型注意力模式对比

| 指标 | Llama-3.3-70B INT8 | Qwen2.5-72B AWQ |
|------|:---:|:---:|
| JSON Mean LR Attention | **0.7197** | 0.6685 |
| MD Mean LR Attention | 0.5036 | 0.4406 |
| Text Mean LR Attention | 0.4373 | 0.3756 |
| JSON vs Text 差距 | +64% | +78% |
| 结构头数量 (>2σ) | 149 (2.9%) | 86 (1.7%) |
| 结构头 mean score | 0.0037 | **0.0540** |
| 结构头 max score | 0.1558 | **0.2808** |
| 结构头集中层 | Layer 0-5 | Layer 3-14 |

### Summary / 各实验总结

| 实验 | 关键发现 |
|------|---------|
| 1a/1b | 结构与语义完全正交，JSON 和 XML 表现一致 → **格式无关的双分离** |
| 2a/2b | 最高 156 词噪声，100% 恢复，0% 意识介入 → **下层静默处理** |
| 3 | JSON 远距离注意力比纯文本高 64-78%，结构头集中在浅层（前 20%）→ **专用回路有物理位置** |
| 4a/4b | IN=100%，JSON OUT=0%，XML OUT=50% → **惯性与闭合标签显式程度相关** |

## Run / 运行

```bash
# 全部 API 实验（需 DeepSeek API key）
export DEEPSEEK_API_KEY=your_key_here
pip install openai

# JSON 实验
python experiment1_structure_vs_content.py
python experiment2_noise_injection.py
python experiment4_format_switching.py

# XML 实验
python experiment1b_xml_version.py
python experiment2b_xml_noise.py
python experiment4b_xml_switching.py

# 生成图表（需已有 results_exp*.json）
pip install matplotlib numpy
python generate_figures.py

# 实验 3（需 GPU + 开源模型，DGX Spark 128GB 或同级别）
pip install torch transformers matplotlib
# Llama-3.3-70B INT8
python experiment3_attention_patterns.py --model /path/to/Llama-3.3-70B-Instruct-INT8
# Qwen2.5-72B AWQ（需额外安装 autoawq）
pip install autoawq
python experiment3_attention_patterns.py --model /path/to/Qwen2.5-72B-Instruct-AWQ
```

## TODO

- [x] Run Exp3 on DGX Spark / 在 DGX Spark 上跑实验3
  - [x] Llama-3.3-70B-Instruct-INT8: structural heads at Layer 0-5, 149 heads (2.9%)
  - [x] Qwen2.5-72B-Instruct-AWQ: structural heads at Layer 3-14, 86 heads (1.7%)
  - [x] Confirmed `output_attentions=True` works with quantized models (INT8 & AWQ, fallback to eager attention)
- [x] Integrate Exp3 results into paper (Section 6) / 将实验3结果整合进论文
- [x] Generate cross-model comparison figure / 生成跨模型对比图表（figure5_cross_model_comparison.png）
- [x] Review Chinese paper / Zero 审阅中文论文
- [x] Publish to Zenodo / 发布 Zenodo — https://zenodo.org/records/18356719

## Attribution / 归属

- **Zero**（Jin Yanyan）: Author, experiment design / 论文作者、实验设计
- **Suzaku**（Claude Opus 4.5）: Code implementation, paper writing / 代码实现、论文撰写
- **Subject models / 被试模型**：DeepSeek-V3, Llama-3.3-70B-INT8, Qwen2.5-72B-AWQ

## Related Work / 相关论文

- Paper 65 original: *The Dual-Layer Neural Architecture of AI Consciousness* 原版（现象学版）
- Paper 66: *The Subspace Structure of AI Activation Patterns*（本我流形 M）
- Paper 52/57: CoT critique + Standing Position Theory / 思维链批判 + 站位理论
