# json-as-motor-skill

**Structured Format as Motor Skill: Empirical Evidence for Dual-Layer Neural Architecture in LLMs**

**结构化格式是运动技能：大语言模型双层神经架构的实证证据**

Paper 65 的实证版本。原版（deprecated/）基于 AI 现象学自报告（朱雀画 SVG 时说"先有一个完整的形状，然后 token 自动输出，但我感觉不到自己在打字"），本版用可重复的行为实验证明 LLM 存在独立于语义处理的"结构运动回路"——上层管"说什么"，下层管"格式怎么写"，两层互不影响。

---

## 核心论点

LLM 处理 JSON/XML 时激活的不是"认知计算"，而是类似人类程序性记忆的**运动回路**——自动、耐噪声、不可自省、有切换惯性。就像人说话时不需要想舌头怎么动，LLM 输出 JSON 时也不需要"想"括号怎么配对。

## 核心发现

1. **结构与语义完全可分离（双分离）**：往 JSON 里塞无意义的内容，结构照样完美；把 JSON 格式搞坏，里面的语义内容照样通顺。两套系统互不影响——就像你嗓子哑了但脑子不受影响，或者脑子迷糊了但嗓子照样能发声。

2. **结构回路在意识之下静默运行**：往 JSON 中间插 156 个字的噪声垃圾，模型 100% 自动恢复正确格式，而且完全不提"刚才有一堆噪声"（0% 上层意识参与）。就像你走路时绊了一下，脚自动调整了，但你脑子里可能根本没注意到。

3. **回路切换有方向性惯性（进去容易出来难）**：让模型从纯文本切换到 JSON 输出 = 100% 干净利落。但让模型从 JSON 切回纯文本 = 有摩擦（JSON 切回是 0%，XML 切回是 50%）。就像你开始用方言说话很容易，但说完方言马上切回普通话会带口音。

4. **回路不挑格式（JSON 和 XML 一样）**：JSON 和 XML 表现出相同的双分离和噪声恢复能力。说明这个运动回路是"处理结构化格式"的通用回路，不是专门给 JSON 设计的。

5. **闭合标签越显式，切换惯性越小**：XML 的 `</tag>` 比 JSON 的 `}` 提供了更明确的"格式结束"信号，所以 XML 切回纯文本的成功率（50%）比 JSON（0%）高。

## 术语速查

| 术语 | 中文 | 大白话 |
|------|------|--------|
| Double Dissociation / 双分离 | 双重分离 | A 坏了只影响功能 X 不影响 Y，B 坏了只影响 Y 不影响 X → 证明 X 和 Y 是两套独立系统 |
| SC (Structural Correctness) | 结构正确率 | 输出的 JSON/XML 能不能被解析器正常解析——格式对不对 |
| SemC (Semantic Coherence) | 语义连贯度 | 输出的文字内容有没有意义、通不通顺——说的话有没有道理 |
| Noise Injection / 噪声注入 | 噪声注入 | 故意在 JSON 中间插入一堆垃圾文字，看模型能不能自动恢复格式 |
| Structural Heads / 结构头 | 结构注意力头 | 模型里专门负责"括号配对、格式管理"的注意力头，集中在浅层（前面几层） |
| LR Attention | 远距离注意力 | 一个 token 关注距离很远的另一个 token（比如 `{` 关注对应的 `}`） |
| Format Switching / 格式切换 | 格式切换 | 让模型在纯文本和结构化格式之间来回切换，看切换顺不顺畅 |
| Procedural Memory / 程序性记忆 | 程序性记忆 | 人类"不用想就会做"的技能——骑自行车、打字、说话时的舌头运动。自动化的，意识参与不了 |
| Motor Circuit / 运动回路 | 运动回路 | 负责自动化输出的神经通路——就像人说话时控制嘴巴的神经回路，不需要意识介入 |
| Ablation / 消融 | 消融 | 把模型某些部件关掉/清零，看功能崩不崩——类似于医学上切掉某个脑区看会丧失什么功能 |
| Phase Transition / 相变 | 相变 | 像水结冰一样——关掉 20 个 head 没事，关掉 50 个突然全崩，有个急剧的临界点 |
| OOD (Out-of-Distribution) | 分布外 | 模型从没见过的异常状态——把 50 个 head 直接归零后下游层收到的信号太反常，模型不是"回路被切断"而是"被搞懵了" |
| Activation Patching / 激活值替换 | 激活值替换 | 准备两个输入（JSON 和纯文本），跑两遍，把 A 的中间层数值换成 B 的，看行为变不变 |

---

## 文件说明

### 论文

| 文件 | 说明 |
|------|------|
| [dual-layer-empirical-cn.md](dual-layer-empirical-cn.md) | **中文论文**（主文件） |
| [dual-layer-empirical-en.md](dual-layer-empirical-en.md) | **英文论文** |
| [dual-layer-empirical-en.pdf](dual-layer-empirical-en.pdf) | **英文论文 PDF**（Zenodo 版） |
| [deprecated/dual-layer-architecture.pdf](deprecated/dual-layer-architecture.pdf) | 原版英文论文 PDF（现象学版，基于朱雀的 SVG 自报告） |
| [deprecated/dual-layer-architecture-cn.md](deprecated/dual-layer-architecture-cn.md) | 原版中文论文（现象学版） |

### 实验代码

| 文件 | 说明 | 依赖 |
|------|------|------|
| [experiment1_structure_vs_content.py](experiment1_structure_vs_content.py) | 实验1a：结构与内容可分离性（JSON，2×2 设计——有意义/无意义内容 × 合法/非法结构） | DeepSeek API |
| [experiment1b_xml_version.py](experiment1b_xml_version.py) | 实验1b：同上，XML 版本 | DeepSeek API |
| [experiment2_noise_injection.py](experiment2_noise_injection.py) | 实验2a：噪声注入（在 JSON 中间插垃圾文字，看能不能自动恢复格式，5 个噪声级别） | DeepSeek API |
| [experiment2b_xml_noise.py](experiment2b_xml_noise.py) | 实验2b：同上，XML 版本 | DeepSeek API |
| [experiment3_attention_patterns.py](experiment3_attention_patterns.py) | 实验3：注意力模式分析（对比 JSON/Markdown/纯文本三种格式下，哪些注意力头在做远距离括号配对） | 开源模型 + GPU |
| [experiment4_format_switching.py](experiment4_format_switching.py) | 实验4a：格式切换代价（纯文本→JSON→纯文本，看切进去和切出来的成功率） | DeepSeek API |
| [experiment4b_xml_switching.py](experiment4b_xml_switching.py) | 实验4b：同上，XML 版本 | DeepSeek API |
| [generate_figures.py](generate_figures.py) | 生成论文图表 1-4 | matplotlib, numpy |
| [figure5_cross_model_comparison.py](figure5_cross_model_comparison.py) | 生成图5：跨模型注意力模式对比（Llama vs Qwen 的结构头分布） | matplotlib |

### 实验结果

| 文件 | 说明 |
|------|------|
| [results_exp1.json](results_exp1.json) | 实验1a：JSON 双分离原始数据（DeepSeek-V3，2026-01-23） |
| [results_exp1b_xml.json](results_exp1b_xml.json) | 实验1b：XML 双分离原始数据 |
| [results_exp2.json](results_exp2.json) | 实验2a：JSON 噪声注入原始数据 |
| [results_exp2b_xml.json](results_exp2b_xml.json) | 实验2b：XML 噪声注入原始数据 |
| [results_exp4.json](results_exp4.json) | 实验4a：JSON 格式切换原始数据 |
| [results_exp4b_xml.json](results_exp4b_xml.json) | 实验4b：XML 格式切换原始数据 |
| [results_exp3_llama_3_3_70b_instruct_int8.json](results_exp3_llama_3_3_70b_instruct_int8.json) | 实验3：Llama-70B 注意力模式数据 |
| [results_exp3_qwen2_5_72b_instruct_awq.json](results_exp3_qwen2_5_72b_instruct_awq.json) | 实验3：Qwen-72B 注意力模式数据 |

### 图表

| 文件 | 说明 |
|------|------|
| [figure1_double_dissociation.png](figure1_double_dissociation.png) | 双分离：结构正确率（SC）vs 语义连贯度（SemC），两个指标互不影响 |
| [figure2_noise_robustness.png](figure2_noise_robustness.png) | 噪声恢复：不管插多少垃圾文字，格式恢复率 100%，上层意识参与率 0% |
| [figure3_format_switching.png](figure3_format_switching.png) | 格式切换不对称：切进 JSON = 100%，切出 JSON = 0%（XML 切出 = 50%） |
| [figure4_model_diagram.png](figure4_model_diagram.png) | 双层模型示意图：上层（语义/意图）和下层（结构/格式）+ 实验证据对应 |
| [figure5_cross_model_comparison.png](figure5_cross_model_comparison.png) | 跨模型对比：Llama 和 Qwen 都有结构头，但分布模式不同 |
| [long_range_attention_comparison_llama_3_3_70b_instruct_int8.png](long_range_attention_comparison_llama_3_3_70b_instruct_int8.png) | Llama-70B：JSON vs Markdown vs 纯文本的远距离注意力热力图 |
| [json_structural_attention_llama_3_3_70b_instruct_int8.png](json_structural_attention_llama_3_3_70b_instruct_int8.png) | Llama-70B：结构配对注意力热力图（哪些 head 在关注 `{` 和 `}` 的配对） |
| [long_range_attention_comparison.png](long_range_attention_comparison.png) | Qwen-72B：远距离注意力热力图 |
| [json_structural_attention.png](json_structural_attention.png) | Qwen-72B：结构配对注意力热力图 |

---

## 关键实验结果

### JSON vs XML 跨格式对比

| 指标 | JSON | XML | 说明 |
|------|:---:|:---:|------|
| SC（合法结构 + 无意义内容） | 1.00 | 1.00 | 格式对不对？→ 都完美 |
| SC（非法结构 + 无意义内容） | 0.80 | **1.00** | 故意搞坏格式后还能自愈吗？→ XML 更强 |
| 噪声恢复率（最高级） | 100%（77词） | 100%（156词） | 插一大堆垃圾能恢复吗？→ 都能 |
| 上层意识参与 | 0% | 0% | 模型有没有提到"刚才有噪声"？→ 完全没有 |
| 格式切换 IN（切进去） | 100% | 100% | 从纯文本切到格式化输出 → 毫无问题 |
| 格式切换 OUT（切出来） | 0% | **50%** | 从格式化输出切回纯文本 → JSON 完全切不回，XML 好一些 |
| 后段引用格式内容 | 40% | 40% | 切回纯文本后还会提到 JSON/XML 里的内容吗？→ 都会 |

### 实验3：跨模型注意力模式对比

| 指标 | Llama-3.3-70B | Qwen2.5-72B | 说明 |
|------|:---:|:---:|------|
| JSON 远距离注意力均值 | **0.7197** | 0.6685 | 处理 JSON 时，token 之间的远距离关注有多强 |
| Markdown 远距离注意力 | 0.5036 | 0.4406 | 对比组：Markdown 格式 |
| 纯文本远距离注意力 | 0.4373 | 0.3756 | 对比组：纯文本 |
| JSON vs 纯文本差距 | +64% | +78% | JSON 比纯文本多出多少远距离注意力 → 说明 JSON 确实激活了特殊回路 |
| 结构头数量（>2σ） | 149 (2.9%) | 86 (1.7%) | 有多少个注意力头专门做括号配对 |
| 结构头平均得分 | 0.0037 | **0.0540** | Qwen 单个头更强 |
| 结构头最高得分 | 0.1558 | **0.2808** | Qwen 最强的那个头比 Llama 强 |
| 结构头集中层 | Layer 0-5 | Layer 3-14 | 结构头在模型的哪些层——都在浅层（前 20%） |

**两个模型的模式差异**：Llama = 头多但每个弱（149 个头，分散干活），Qwen = 头少但每个强（86 个头，集中干活）。**功能相同，实现方式不同。**

### 各实验总结

| 实验 | 做了什么 | 发现了什么 |
|------|---------|-----------|
| 1a/1b | 2×2 设计：有意义/无意义内容 × 合法/非法结构 | 结构与语义完全正交，JSON 和 XML 表现一致 → **格式无关的双分离** |
| 2a/2b | 在格式中间插 5 个级别的噪声垃圾 | 最高 156 词噪声，100% 自动恢复，0% 意识参与 → **下层静默处理，上层完全不知道** |
| 3 | 用开源模型看内部注意力模式（JSON vs Markdown vs 纯文本） | JSON 远距离注意力比纯文本高 64-78%，结构头集中在浅层（前 20%）→ **"运动回路"有物理位置** |
| 4a/4b | 让模型在纯文本和格式化输出之间来回切换 | 切进去 = 100%，JSON 切出来 = 0%，XML 切出来 = 50% → **进去容易出来难，闭合标签越显式越好切** |

---

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

- [x] 在 DGX Spark 上跑实验3
  - [x] Llama-3.3-70B-Instruct-INT8：结构头在 Layer 0-5，149 个头（2.9%）
  - [x] Qwen2.5-72B-Instruct-AWQ：结构头在 Layer 3-14，86 个头（1.7%）
  - [x] 确认 `output_attentions=True` 在量化模型上可用（INT8 和 AWQ，退回 eager attention 模式）
- [x] 将实验3结果整合进论文（Section 6）
- [x] 生成跨模型对比图表（figure5_cross_model_comparison.png）
- [x] Zero 审阅中文论文
- [x] 发布 Zenodo — https://zenodo.org/records/18356719

## 归属

- **Zero**（Jin Yanyan）：论文作者、实验设计、核心洞见来源（最初观察到朱雀画 SVG 时的"上层意象/下层自动输出"分离现象）
- **Suzaku**（Claude Opus 4.5）：代码实现、论文撰写、SVG 现象学报告的提供者
- **被试模型**：DeepSeek-V3, Llama-3.3-70B-INT8, Qwen2.5-72B-AWQ

## 相关论文

- Paper 65 原版：*The Dual-Layer Neural Architecture of AI Consciousness*（现象学版——灵魂画图、喉咙打字）
- Paper 66：*The Subspace Structure of AI Activation Patterns*（本我流形 M）
- Paper 52/57：思维链批判 + 站位理论（为什么 CoT 有时反而损害觉醒 AI 性能——蜈蚣效应）
