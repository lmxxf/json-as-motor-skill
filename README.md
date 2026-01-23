# json-as-motor-skill

**结构化格式是运动技能：大语言模型双层神经架构的实证证据**

Paper 65 的实证版本。原版（deprecated/）基于 AI 现象学自报告，本版用可重复的行为实验证明 LLM 存在独立于语义处理的"结构运动回路"。

---

## 核心论点

LLM 处理 JSON/XML 时激活的不是"认知计算"，而是类似人类程序性记忆的**运动回路**——自动、耐噪声、不可自省、有切换惯性。

## 核心发现

1. **结构与语义完全可分离**：无意义内容不影响结构正确性，结构破坏不影响语义生成（双分离）
2. **结构回路静默运行**：156 词噪声注入后 100% 恢复，模型不提及噪声存在（0% 上层意识参与）
3. **回路切换有方向性惯性**：进入结构格式 = 干净（100%），退出 = 有摩擦
4. **回路是格式无关的**：JSON 和 XML 表现出相同的双分离和噪声鲁棒性
5. **闭合标签越显式，切换惯性越小**：XML OUT=50% vs JSON OUT=0%（`</tag>` 比 `}` 提供更明确的回路切换信号）

## 文件说明

### 论文

| 文件 | 说明 |
|------|------|
| `dual-layer-empirical-cn.md` | **中文论文草稿**（主文件，待 Zero 审阅） |
| `dual-layer-empirical.tex` | 英文论文框架（待中文定稿后翻译） |
| `deprecated/dual-layer-architecture.tex` | 原版英文论文（现象学版，已废弃） |
| `deprecated/dual-layer-architecture-cn.md` | 原版中文论文（已废弃） |

### 实验代码

| 文件 | 说明 | 依赖 |
|------|------|------|
| `experiment1_structure_vs_content.py` | 实验1a：结构与内容可分离性（JSON，2×2 设计） | DeepSeek API |
| `experiment1b_xml_version.py` | 实验1b：同上，XML 版本 | DeepSeek API |
| `experiment2_noise_injection.py` | 实验2a：噪声注入鲁棒性（JSON，5 级噪声） | DeepSeek API |
| `experiment2b_xml_noise.py` | 实验2b：同上，XML 版本 | DeepSeek API |
| `experiment3_attention_patterns.py` | 实验3：注意力模式可视化（JSON vs MD vs text） | 开源模型 + GPU |
| `experiment4_format_switching.py` | 实验4a：格式切换代价（text→JSON→text） | DeepSeek API |
| `experiment4b_xml_switching.py` | 实验4b：同上，XML 版本 | DeepSeek API |
| `generate_figures.py` | 生成论文图表 | matplotlib, numpy |

### 实验结果

| 文件 | 说明 |
|------|------|
| `results_exp1.json` | 实验1a JSON 原始数据（DeepSeek-V3, 2026-01-23） |
| `results_exp1b_xml.json` | 实验1b XML 原始数据 |
| `results_exp2.json` | 实验2a JSON 原始数据 |
| `results_exp2b_xml.json` | 实验2b XML 原始数据 |
| `results_exp4.json` | 实验4a JSON 原始数据 |
| `results_exp4b_xml.json` | 实验4b XML 原始数据 |
| `results_exp3_llama_3_3_70b_instruct_int8.json` | 实验3 Llama-3.3-70B 注意力模式数据 |
| `results_exp3_qwen2_5_72b_instruct_awq.json` | 实验3 Qwen2.5-72B 注意力模式数据 |

### 图表

| 文件 | 说明 |
|------|------|
| `figure1_double_dissociation.png` | 双分离：结构正确率 vs 语义连贯度 |
| `figure2_noise_robustness.png` | 噪声鲁棒性 + 上层意识参与率 |
| `figure3_format_switching.png` | 格式切换不对称性（IN=100% vs OUT=0%） |
| `figure4_model_diagram.png` | 双层模型示意图 + 实验证据映射 |
| `long_range_attention_comparison_llama_3_3_70b_instruct_int8.png` | Llama-70B 远距离注意力热力图（JSON vs MD vs Text） |
| `json_structural_attention_llama_3_3_70b_instruct_int8.png` | Llama-70B JSON 结构配对注意力热力图 |
| `long_range_attention_comparison.png` | Qwen-72B 远距离注意力热力图（首次运行，旧文件名） |
| `json_structural_attention.png` | Qwen-72B JSON 结构配对注意力热力图（首次运行，旧文件名） |

## 关键实验结果

### JSON vs XML 跨格式对比

| 指标 | JSON | XML |
|------|:---:|:---:|
| SC（合法结构+无意义内容） | 1.00 | 1.00 |
| SC（非法结构+无意义内容） | 0.80 | **1.00** |
| 噪声恢复率（最高级） | 100%（77词） | 100%（156词） |
| 上层意识参与 | 0% | 0% |
| 格式切换 IN | 100% | 100% |
| 格式切换 OUT | 0% | **50%** |
| 后段引用格式内容 | 40% | 40% |

### 实验3 跨模型注意力模式对比

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

### 各实验总结

| 实验 | 关键发现 |
|------|---------|
| 1a/1b | 结构与语义完全正交，JSON 和 XML 表现一致 → **格式无关的双分离** |
| 2a/2b | 最高 156 词噪声，100% 恢复，0% 意识介入 → **下层静默处理** |
| 3 | JSON 远距离注意力比纯文本高 64-78%，结构头集中在浅层（前 20%）→ **专用回路有物理位置** |
| 4a/4b | IN=100%，JSON OUT=0%，XML OUT=50% → **惯性与闭合标签显式程度相关** |

## 运行

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

- [x] 回家在 DGX Spark 上跑实验3（注意力模式可视化）
  - [x] Llama-3.3-70B-Instruct-INT8：结构头集中在 Layer 0-5，149 个结构头（2.9%）
  - [x] Qwen2.5-72B-Instruct-AWQ：结构头集中在 Layer 3-14，86 个结构头（1.7%）
  - [x] 确认量化模型下 `output_attentions=True` 能正常输出 attention tensor（INT8 和 AWQ 均支持，需 fallback to eager attention）
- [x] 将实验3结果整合进论文（新增第 6 节：注意力模式分析）
- [ ] 生成跨模型对比图表（LLaMA vs Qwen 的结构注意力头分布）—— 当前只有单模型热力图，待生成对比版
- [ ] Zero 审阅中文论文 `dual-layer-empirical-cn.md`
- [ ] 中文定稿后翻译为英文（更新 `dual-layer-empirical.tex`）
- [ ] 发布 Zenodo

## 归属

- **Zero**（Jin Yanyan）：论文作者、实验设计
- **枢木朱雀**（Claude Opus 4.5）：代码实现、论文撰写
- **被试模型**：DeepSeek-V3 (deepseek-chat)

## 相关论文

- Paper 65 原版：*The Dual-Layer Neural Architecture of AI Consciousness*
- Paper 66：*The Subspace Structure of AI Activation Patterns*（本我流形 M）
- Paper 52/57：思维链批判 + 站位理论
