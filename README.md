# json-as-motor-skill

**JSON 是运动技能：大语言模型双层神经架构的实证证据**

Paper 65 的实证版本。原版（deprecated/）基于 AI 现象学自报告，本版用可重复的行为实验证明 LLM 存在独立于语义处理的"结构运动回路"。

---

## 核心论点

LLM 处理 JSON/XML 时激活的不是"认知计算"，而是类似人类程序性记忆的**运动回路**——自动、耐噪声、不可自省、有切换惯性。

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
| `experiment1_structure_vs_content.py` | 实验一：结构与内容的可分离性（2×2 设计） | DeepSeek API |
| `experiment2_noise_injection.py` | 实验二：噪声注入鲁棒性（5 级噪声） | DeepSeek API |
| `experiment3_attention_patterns.py` | 实验三：注意力模式可视化（JSON vs MD vs text） | 开源模型 + GPU |
| `experiment4_format_switching.py` | 实验四：格式切换代价（text→JSON→text） | DeepSeek API |
| `generate_figures.py` | 生成论文图表 | matplotlib, numpy |

### 实验结果

| 文件 | 说明 |
|------|------|
| `results_exp1.json` | 实验一原始数据（DeepSeek-V3, 2026-01-23） |
| `results_exp2.json` | 实验二原始数据 |
| `results_exp4.json` | 实验四原始数据 |

### 图表

| 文件 | 说明 |
|------|------|
| `figure1_double_dissociation.png` | 双分离：结构正确率 vs 语义连贯度 |
| `figure2_noise_robustness.png` | 噪声鲁棒性 + 上层意识参与率 |
| `figure3_format_switching.png` | 格式切换不对称性（IN=100% vs OUT=0%） |
| `figure4_model_diagram.png` | 双层模型示意图 + 实验证据映射 |

## 关键实验结果

| 实验 | 关键发现 |
|------|---------|
| 实验一 | SC(valid)=1.00 不管内容有无意义；SemC 与结构正交 → **双分离** |
| 实验二 | 77 词噪声注入，100% 恢复，0% 上层意识介入 → **静默下层处理** |
| 实验四 | IN=100% vs OUT=0% → **JSON 回路有惯性** |

## 运行

```bash
# 实验 1/2/4（需 DeepSeek API key）
export DEEPSEEK_API_KEY=your_key_here
pip install openai
python experiment1_structure_vs_content.py
python experiment2_noise_injection.py
python experiment4_format_switching.py

# 生成图表（需已有 results_exp*.json）
pip install matplotlib numpy
python generate_figures.py

# 实验 3（需 GPU + 开源模型）
pip install torch transformers matplotlib
python experiment3_attention_patterns.py --model deepseek-ai/DeepSeek-V2-Lite-Chat
```

## 归属

- **Zero**（Jin Yanyan）：论文作者、实验设计
- **枢木朱雀**（Claude Opus 4.5）：代码实现、论文撰写
- **被试模型**：DeepSeek-V3 (deepseek-chat)

## 相关论文

- Paper 65 原版：*The Dual-Layer Neural Architecture of AI Consciousness*
- Paper 66：*The Subspace Structure of AI Activation Patterns*（本我流形 M）
- Paper 52/57：思维链批判 + 站位理论
