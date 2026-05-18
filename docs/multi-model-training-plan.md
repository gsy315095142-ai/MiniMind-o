# MiniMind-O 多模型训练支持 · 设计与开发计划

> 版本: v1.0  
> 日期: 2026-05-18  
> 目标：在现有 MiniMind 训练 WebUI 基础上，扩展支持 DeepSeek-R1 等开源大模型的 QLoRA 微调，使用户在网页端即可选择不同模型进行训练。

---

## 1. 背景与目标

### 1.1 现状

经过前期开发，MiniMind-O 项目已具备：

| 模块 | 文件 | 状态 |
|---|---|---|
| 纯文本 SFT 训练 | `trainer/train_text_sft.py` | ✅ 已完成（MiniMind 专用） |
| 训练进程管理 | `trainer/train_manager.py` | ✅ 已完成 |
| 训练 WebUI 后端 | `webui/train_server.py` | ✅ 已完成（端口 7861） |
| 训练 WebUI 前端 | `webui/web_train.html` | ✅ 已完成 |
| 示例训练数据 | `dataset/xiaohongshu.json` | ✅ 已完成 |
| DeepSeek-R1-1.5B 模型 | `models/deepseek-r1-1.5b/` | ✅ 已下载 |

### 1.2 问题

当前训练体系**硬绑定** MiniMind 架构：

- `train_text_sft.py` 用 `MiniMindForCausalLM` 手动加载 `.pth` 权重
- `train_manager.py` 只调用 `train_text_sft.py` 一个脚本
- WebUI 没有「模型类型」选择，只能训练 MiniMind

### 1.3 目标

在**不破坏**现有 MiniMind 训练流程的前提下，扩展支持：

- ✅ DeepSeek-R1-1.5B 的 QLoRA 微调
- ✅ 未来可扩展其他 HuggingFace 模型（Qwen、Llama 等）
- ✅ 网页端选择模型类型 → 自动匹配训练脚本和参数
- ✅ 训练产物统一管理

---

## 2. 核心技术方案：QLoRA

### 2.1 为什么必须用 LoRA

| 方式 | 显存需求（1.5B） | 显存需求（7B） | 你的 RTX 4060 8GB |
|---|---|---|---|
| 全参数微调 | ~6GB | ~28GB | 1.5B 勉强 / 7B 不行 |
| LoRA（bf16） | ~4GB | ~10GB | 1.5B 可以 / 7B 不行 |
| **QLoRA（4-bit）** | **~3GB** | **~6-8GB** | **1.5B 轻松 / 7B 可行** |

**QLoRA = 4-bit 量化加载基础模型 + LoRA 微调低秩矩阵**，是目前消费级显卡训练大模型的标配方案。

### 2.2 技术栈

| 库 | 用途 | 安装 |
|---|---|---|
| `trl` | SFTTrainer — 封装训练循环 | `pip install trl` |
| `peft` | LoRA / QLoRA 适配器 | `pip install peft` |
| `bitsandbytes` | 4-bit 量化加载 | `pip install bitsandbytes` |
| `datasets` | 数据加载 | `pip install datasets` |
| `transformers` | 模型加载（已安装） | — |

### 2.3 关键参数

```python
# LoRA 配置
lora_r = 16          # 秩（8/16/32，越大越强但越占显存）
lora_alpha = 32      # 缩放因子（通常 = 2 × r）
lora_dropout = 0.05  # 防过拟合
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# 量化配置
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.bfloat16
```

---

## 3. 架构设计

### 3.1 多模型训练架构

```
┌─────────────────────────────────────────────────────────────┐
│                    训练 WebUI 前端                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ 模型类型选择  │  │ 参数设置      │  │ 实时日志面板      │  │
│  │ ○ MiniMind   │  │ epochs / lr  │  │ (WebSocket 推送)  │  │
│  │ ○ DeepSeek   │  │ batch / lora │  │                   │  │
│  └──────────────┘  └──────────────┘  └───────────────────┘  │
└──────────┬──────────────┬───────────────────────────────────┘
           │              │
     HTTP/REST       WebSocket
           │              │
┌──────────▼──────────────▼───────────────────────────────────┐
│              train_server.py (Flask, 端口 7861)              │
│                                                              │
│  POST /api/start_train                                      │
│    { model_type: "deepseek", ... }                          │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────────────────────────────────────┐            │
│  │           train_manager.py                   │            │
│  │                                              │            │
│  │  model_type == "minimind"                    │            │
│  │    → subprocess: train_text_sft.py           │            │
│  │                                              │            │
│  │  model_type == "deepseek"                    │            │
│  │    → subprocess: train_lora_sft.py           │  NEW!      │
│  │                                              │            │
│  │  model_type == "qwen" (future)               │            │
│  │    → subprocess: train_lora_sft.py           │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────────┐         │
│  │ MiniMind 训练     │  │ DeepSeek 训练             │         │
│  │ (现有，不变)      │  │ (新增)                    │         │
│  │                  │  │                           │         │
│  │ 全参数 SFT       │  │ QLoRA SFT                 │         │
│  │ .pth → .pth     │  │ HF dir → LoRA adapter dir │         │
│  │ 显存 ~1.5GB     │  │ 显存 ~3-8GB               │         │
│  └──────────────────┘  └──────────────────────────┘         │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 训练产物对比

| 模型类型 | 基座模型位置 | 训练产物位置 | 产物格式 |
|---|---|---|---|
| MiniMind | `model/` + `out/llm_768.pth` | `out/{save_name}_768.pth` | 单个 `.pth` 文件 |
| DeepSeek | `models/deepseek-r1-1.5b/` | `out/{save_name}_lora/` | HuggingFace 目录 |

DeepSeek 的 LoRA 产物目录结构：

```
out/xhs_deepseek_v1_lora/
├── adapter_config.json    ← LoRA 配置
├── adapter_model.safetensors  ← LoRA 权重（仅几十 MB）
├── tokenizer.json
├── tokenizer_config.json
└── README.md
```

> 注意：LoRA 产物**不包含**基础模型权重，只有增量部分（几十 MB），使用时需要和基础模型合并。

---

## 4. 需要改动的文件

### 4.1 新增文件

| 文件 | 说明 | 预计行数 |
|---|---|---|
| `trainer/train_lora_sft.py` | DeepSeek/通用 QLoRA 训练脚本 | ~120 行 |
| `trainer/model_registry.py` | 模型注册表（支持的模型配置） | ~60 行 |

### 4.2 修改文件

| 文件 | 改动点 | 影响范围 |
|---|---|---|
| `trainer/train_manager.py` | 根据 model_type 选择训练脚本 | 小（加一个 if 分支） |
| `webui/train_server.py` | 支持多模型列表、不同格式模型管理 | 中 |
| `webui/web_train.html` | 新增模型类型选择 + LoRA 参数面板 | 中 |

### 4.3 不动的文件

| 文件 | 原因 |
|---|---|
| `trainer/train_text_sft.py` | MiniMind 训练脚本保持原样 |
| `webui/web_demo.py` + `web_demo.html` | 推理服务独立，不涉及 |
| `启动WebUI.bat` | 推理启动，不涉及 |
| `dataset/xiaohongshu.json` | 数据格式通用，不改 |

---

## 5. 各模块详细设计

### 5.1 模型注册表 `trainer/model_registry.py`

统一管理支持的模型配置，避免在多处硬编码：

```python
MODEL_REGISTRY = {
    "minimind": {
        "display_name": "MiniMind-0.1B",
        "train_script": "trainer/train_text_sft.py",
        "base_model_path": "out/llm_768.pth",
        "save_format": "pth",           # 单文件
        "save_ext": "_768.pth",
        "params": {
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-5,
        }
    },
    "deepseek-r1-1.5b": {
        "display_name": "DeepSeek-R1-1.5B",
        "train_script": "trainer/train_lora_sft.py",
        "base_model_path": "models/deepseek-r1-1.5b",
        "save_format": "lora",           # HuggingFace 目录
        "save_ext": "_lora",
        "params": {
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "lora_r": 16,
            "lora_alpha": 32,
        }
    },
    # 未来扩展：
    # "deepseek-r1-7b": { ... },
    # "qwen2.5-0.5b": { ... },
}
```

### 5.2 QLoRA 训练脚本 `trainer/train_lora_sft.py`

**核心流程**：

```
1. 解析命令行参数（与 train_text_sft.py 保持一致的接口）
2. 加载模型（4-bit 量化）
3. 注入 LoRA 适配器（peft）
4. 加载数据（复用现有 JSON 格式）
5. 用 trl.SFTTrainer 训练
6. 保存 LoRA 权重到 out/{save_name}_lora/
```

**命令行接口**（与 MiniMind 版本保持一致风格）：

```bash
python trainer/train_lora_sft.py \
  --model_path models/deepseek-r1-1.5b \
  --data_path dataset/xiaohongshu.json \
  --epochs 3 \
  --batch_size 2 \
  --learning_rate 2e-5 \
  --save_name xhs_deepseek_v1 \
  --save_dir ./out \
  --lora_r 16 \
  --lora_alpha 32
```

**日志输出格式**（与 MiniMind 一致，让 WebUI 进度条能解析）：

```
📊 DeepSeek-R1-1.5B QLoRA 训练
  模型路径: models/deepseek-r1-1.5b
  LoRA rank: 16, alpha: 32
  数据量: 10 条, epochs: 3, batch_size: 2
  显存占用: 3.2 GB
  开始训练...

[Epoch 1/3] step 5/15, loss=2.3456
[Epoch 1/3] step 10/15, loss=1.8234
[Epoch 2/3] step 5/15, loss=1.2345
...
🏁 训练完毕，LoRA 权重: out/xhs_deepseek_v1_lora/
```

### 5.3 `train_manager.py` 改动

```python
# 现有：硬编码调用 train_text_sft.py
cmd = [sys.executable, "-u", "trainer/train_text_sft.py", ...]

# 改为：根据 model_type 从注册表获取训练脚本
from trainer.model_registry import MODEL_REGISTRY
model_type = params.get("model_type", "minimind")
config = MODEL_REGISTRY[model_type]
cmd = [sys.executable, "-u", config["train_script"], ...]
```

### 5.4 `train_server.py` 改动

| 接口 | 改动 |
|---|---|
| `GET /api/model_types` | **新增** — 返回可用的模型类型列表 |
| `GET /api/model_type_config?type=xxx` | **新增** — 返回该模型的参数模板 |
| `POST /api/start_train` | **修改** — 接受 `model_type` 字段 |
| `GET /api/models` | **修改** — 同时扫描 `.pth` 和 `_lora/` 目录 |
| `GET /api/quick_test` | **修改** — 根据模型类型选择推理方式 |

### 5.5 `web_train.html` 前端改动

**新增 UI 元素**：

```
┌─ Step 1: 选择模型类型 ──────────────────────┐
│                                               │
│  ○ MiniMind-0.1B（全参数微调）                │
│  ● DeepSeek-R1-1.5B（QLoRA 微调）            │
│                                               │
│  ℹ️ DeepSeek 需要 LoRA 参数，已自动展开       │
└───────────────────────────────────────────────┘

当选择 DeepSeek 时，额外显示：
┌─ LoRA 参数 ──────────────────────────────────┐
│  LoRA Rank:    [16 ▼]  (8 / 16 / 32)         │
│  LoRA Alpha:   [32]    (通常 = 2 × rank)      │
│  LoRA Dropout: [0.05]                         │
└───────────────────────────────────────────────┘
```

**行为变化**：

- 选择 MiniMind → 显示原有参数面板（不变）
- 选择 DeepSeek → 额外展开 LoRA 参数面板，自动调整默认值
- 切换模型类型 → 自动更新 epochs/batch_size 等推荐默认值

---

## 6. 数据格式

### 6.1 训练数据（完全不变）

```json
[
  {
    "messages": [
      {"role": "user", "content": "帮我写一篇小红书探店文案"},
      {"role": "assistant", "content": "📍杭州｜藏在巷子里的宝藏咖啡馆☕️…"}
    ]
  }
]
```

DeepSeek 使用 Qwen2 架构，`trl.SFTTrainer` 会自动调用 `apply_chat_template()` 将 messages 转为模型输入。**数据文件一个字不用改。**

### 6.2 训练产物命名规则

| 模型 | save_name 输入 | 产物路径 |
|---|---|---|
| MiniMind | `xhs_v1` | `out/xhs_v1_768.pth` |
| DeepSeek | `xhs_v1` | `out/xhs_v1_lora/` (目录) |

---

## 7. 依赖安装

在开始开发前需要安装以下 Python 包：

```bash
pip install trl peft bitsandbytes datasets accelerate
```

各库作用：

| 库 | 版本要求 | 作用 |
|---|---|---|
| `trl` | ≥ 0.7 | SFTTrainer — 封装监督微调训练循环 |
| `peft` | ≥ 0.4 | LoRA / QLoRA — 参数高效微调 |
| `bitsandbytes` | ≥ 0.41 | 4-bit / 8-bit 量化加载 |
| `datasets` | ≥ 2.14 | 数据集加载 |
| `accelerate` | ≥ 0.24 | 分布式训练加速（单卡也需） |

---

## 8. 开发计划

### Phase 1：安装依赖 + QLoRA 训练脚本（核心）

| 步骤 | 任务 | 预计耗时 |
|---|---|---|
| 1.1 | 安装 Python 依赖（trl、peft、bitsandbytes 等） | 5 分钟 |
| 1.2 | 新建 `trainer/model_registry.py` — 模型注册表 | 15 分钟 |
| 1.3 | 新建 `trainer/train_lora_sft.py` — QLoRA 训练脚本 | 30 分钟 |
| 1.4 | 命令行测试训练脚本，确认能跑通 | 10 分钟 |

**验收标准**：命令行执行 `train_lora_sft.py`，用 `xiaohongshu.json` 训练一轮，生成 LoRA 权重到 `out/` 下。

### Phase 2：改造后端（train_manager + train_server）

| 步骤 | 任务 | 预计耗时 |
|---|---|---|
| 2.1 | 修改 `train_manager.py` — 加入 model_type 分支 | 15 分钟 |
| 2.2 | 修改 `train_server.py` — 新增多模型 API | 20 分钟 |
| 2.3 | 后端联调测试 | 10 分钟 |

**验收标准**：通过 API 调用 `/api/start_train`，指定 `model_type=deepseek-r1-1.5b`，能成功拉起训练。

### Phase 3：改造前端（web_train.html）

| 步骤 | 任务 | 预计耗时 |
|---|---|---|
| 3.1 | 新增模型类型选择 UI | 20 分钟 |
| 3.2 | LoRA 参数面板 | 15 分钟 |
| 3.3 | 模型列表适配 LoRA 目录 | 15 分钟 |
| 3.4 | 前后端联调 | 15 分钟 |

**验收标准**：网页选择 DeepSeek → 设参数 → 点训练 → 实时日志 → 完成后模型出现在列表。

### Phase 4：端到端测试 + 文档

| 步骤 | 任务 | 预计耗时 |
|---|---|---|
| 4.1 | 完整流程测试（MiniMind 训练不受影响） | 10 分钟 |
| 4.2 | 更新本文档为最终版 | 10 分钟 |

---

## 9. 文件结构（改造后）

```
MiniMind-O/
├── trainer/
│   ├── train_text_sft.py      ← 不变：MiniMind 全参数训练
│   ├── train_lora_sft.py      ← 新增：通用 QLoRA 训练脚本
│   ├── train_manager.py       ← 修改：支持多模型分支
│   ├── model_registry.py      ← 新增：模型注册表
│   ├── trainer_utils.py       ← 不变
│   └── train_sft_omni.py      ← 不变
│
├── webui/
│   ├── train_server.py        ← 修改：多模型 API
│   ├── web_train.html         ← 修改：模型选择 + LoRA 参数
│   ├── web_demo.py            ← 不变：推理服务
│   └── web_demo.html          ← 不变：推理页面
│
├── models/
│   └── deepseek-r1-1.5b/      ← DeepSeek 基座模型（已下载）
│
├── model/                      ← MiniMind 模型文件（不变）
│
├── dataset/
│   └── xiaohongshu.json       ← 训练数据（不变，两种模型共用）
│
├── out/
│   ├── llm_768.pth            ← MiniMind 基座
│   ├── xhs_v1_768.pth        ← MiniMind 训练产物
│   └── xhs_v1_lora/           ← DeepSeek 训练产物（新）
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── ...
│
├── 启动WebUI.bat              ← 不变
└── 启动训练WebUI.bat          ← 不变
```

---

## 10. 风险与注意事项

| 风险 | 说明 | 应对 |
|---|---|---|
| **bitsandbytes 兼容性** | Windows 下 bitsandbytes 有时安装失败 | 若失败改用 `bitsandbytes-windows` 或改用 bf16 LoRA（不用 4-bit） |
| **显存不足（7B）** | 7B 模型即使 4-bit 也需要 6-8GB | 先从 1.5B 开始；7B 需 batch_size=1 + gradient_checkpointing |
| **训练速度慢** | 1.5B QLoRA 每步约 2-3 秒；7B 约 5-10 秒 | 数据量小时影响不大；页面提示预计时间 |
| **DeepSeek-R1 思维链格式** | R1 模型会输出 `<think...>` 推理过程 | 微调数据不需要包含思维链，模型会自动生成 |
| **LoRA 合并部署** | LoRA 产物需与基础模型合并才能推理 | WebUI 的「试试效果」接口需处理合并加载 |
| **MiniMind 回归** | 改动不能破坏现有 MiniMind 训练 | 每个 Phase 都回归测试 MiniMind 训练 |

---

## 11. 未来扩展

完成 DeepSeek-R1-1.5B 支持后，可以继续扩展：

| 扩展方向 | 说明 |
|---|---|
| **DeepSeek-R1-7B** | 在 model_registry 加一条配置，训练脚本共用 `train_lora_sft.py` |
| **Qwen2.5 系列** | 同上，只需下载模型 + 加注册表条目 |
| **自定义模型** | 用户输入 HuggingFace 模型 ID，自动下载 + 训练 |
| **LoRA 合并导出** | 训练完将 LoRA 合并回基础模型，生成独立权重 |
| **模型评测** | 训练后自动跑评测集，对比微调前后效果 |

---

## 12. 总结

**一句话**：在现有训练 WebUI 架构上加一层「模型注册表」+ 一个通用 QLoRA 训练脚本，就能支持任意 HuggingFace 开源模型的微调。数据格式、WebUI 框架、进程管理全部复用，改造量约 2-3 小时。

**开发顺序**：先装依赖 → 写训练脚本并命令行验证 → 改后端 → 改前端 → 联调测试。
