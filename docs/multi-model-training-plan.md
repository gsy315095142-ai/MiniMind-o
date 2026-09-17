# MiniMind-O 多模型训练支持 · 设计与开发计划

> 版本: v1.2  
> 日期: 2026-05-18  
> 目标：在现有 MiniMind 训练 WebUI 基础上，扩展支持 DeepSeek-R1 等开源大模型的 QLoRA 微调，使用户在网页端即可选择不同模型进行训练。
>
> **v1.1 更新**：补充 Windows bitsandbytes 三级降级策略、trl/peft 版本锁定、进度条日志格式硬约束、`base_model` 字段冲突坑、LoRA 缓存与显存控制、quick_test LoRA 加载方式、难度档位语义差异等实操要点。
>
> **v1.1.1 修复（2026-05-18）**：用户反馈选 DeepSeek 时看不到基座信息。修复方案：`/api/model_type_config` 新增 `base_model_rel` + `base_model_size_mb` 字段；前端在「基础模型」下拉的同一栏位增加一个**只读基座展示框**（虚线边框示意只读），DeepSeek 模式下显示 `display_name + 相对路径 + 体积 + 已下载/未下载` 状态。这是 Phase 3 完成后发现的 UX 缺陷——「不能选」不等于「不该看到」。
>
> **v1.2 新增（2026-05-18）— LoRA 多阶段续训**：用户反馈「训完 A 想再训 A 怎么办」。MiniMind 那条路径靠「基础模型下拉选历史 .pth」天然支持续训，DeepSeek/HF LoRA 路径却完全没入口——每次都从原始基座 + 全新 LoRA 重头训。修复方案：
>
> 1. `trainer/train_lora_sft.py` 新增 `--resume_from_lora` 参数。命中时走 `PeftModel.from_pretrained(base, adapter, is_trainable=True)` 替代 `get_peft_model`，LoRA 结构参数（r/alpha/dropout/target_modules）由原 adapter 的 `adapter_config.json` 固定，命令行同名参数被忽略并在日志中明确标注。
> 2. `trainer/train_manager.py` 在 hf_causal_lm 分支透传 `resume_from_lora`。
> 3. `webui/train_server.py`：
>    - 新增 `GET /api/resumable_loras?type=xxx`，列出 `out/*_lora/` 中可作为续训起点的产物，并通过比较 `minimind_o_meta.json` 里的 `base_model_path` 与目标 `model_type` 的基座路径判定 `compatible`（兼容**旧产物相对路径** vs **新产物绝对路径**的混用场景）。
>    - `POST /api/start_train` 新增 `resume_from` 字段，做四重防御：路径穿越、`_lora` 后缀强制、同名冲突拦截、跨基座续训拦截。
>    - `GET /api/models` 在 LoRA 行新增 `resume_from` 字段，展示血统。
> 4. `webui/web_train.html`：在 HF 模式的基座框下方新增「续训起点」下拉；选了非空续训源后弹出黄色提示条 + 自动将 LoRA Rank/Alpha/Dropout 标灰（小标签「已被续训源固定」），但保留 Max Seq Len 可编辑（数据预处理参数与 LoRA 结构无关）；「已有模型」表格在模型名下补一行「🔁 续训自 xxx」。
> 5. `minimind_o_meta.json` 新增 `resume_from_lora` 字段，记录血统链路。
>
> **设计取舍**：采用「在已有 adapter 上继续训」而非「合并基座重训」方案——前者产物仍是小 LoRA、保留前一轮成果、训练成本最低；后者需要先 `merge_and_unload()` 再训一个全新 LoRA，适合「彻底固化阶段 1 + 独立阶段 2」场景，列入 §11 未来扩展。

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
  --lora_alpha 32 \
  [--no_4bit]   # 可选：bitsandbytes 不可用时显式降级到 bf16 LoRA
```

**日志输出格式**（**必须严格对齐 MiniMind 版**，否则 `web_train.html` 的进度条解析会失败）：

WebUI 进度条依赖的正则是 `Epoch \[(\d+)/(\d+)\] Step \[(\d+)/(\d+)\] loss=([\d.]+)`，所以训练循环的每一行 step 日志必须长这样：

```
📊 DeepSeek-R1-1.5B QLoRA 训练
  模型路径: models/deepseek-r1-1.5b
  量化模式: 4-bit NF4    (或 bf16 LoRA，按实际)
  LoRA rank: 16, alpha: 32, dropout: 0.05
  数据量: 10 条, epochs: 3, batch_size: 2
  可训练参数: 9.4M / 1543.7M (0.61%)
  显存占用: 3.2 GB
🚀 开始训练 | epochs=3 | batch=2 | lr=2e-5 | steps=15

  Epoch [1/3] Step [5/5] loss=2.3456 avg=2.4123 lr=2.0e-05 eta=1.2min
  Epoch [2/3] Step [5/5] loss=1.8234 avg=1.9012 lr=1.5e-05 eta=0.8min
  ...
✅ Epoch [3/3] 完成 | avg_loss=1.2345 | 耗时=2.4min
🏁 训练完毕，LoRA 权重: out/xhs_deepseek_v1_lora/
```

> **关键约束**：`Epoch [x/y] Step [a/b] loss=... avg=... lr=... eta=...` 这一整行格式不能变。前缀 emoji、数据集元信息行随便加，但 step 行必须长这样。

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
| `POST /api/start_train` | **修改** — 接受 `model_type` 字段；MiniMind 路径走原有 `--base_model xxx.pth` 拼接；DeepSeek 路径绕过它，改用 `--model_path models/...`（**坑：当前 `f"{base_model}_768.pth"` 是 MiniMind 写死的，DeepSeek 时必须不走这条分支**） |
| `GET /api/models` | **修改** — 同时扫描 `out/*.pth` 和 `out/*_lora/` 目录；返回 item 上加 `format: "pth" / "lora"` 字段 |
| `POST /api/delete_model` | **修改** — 支持删除 `_lora/` 目录（`shutil.rmtree`，加上安全校验防路径穿越） |
| `POST /api/quick_test` | **修改** — 根据模型类型选择推理方式：MiniMind 走原逻辑；LoRA 用 `AutoModelForCausalLM.from_pretrained(base, quantization_config=...)` + `PeftModel.from_pretrained(adapter_dir)` |

**`_TEST_CACHE` 缓存策略**（重要）：
- 现状是按 `model_name:mtime` 缓存，可以并存多个模型；
- LoRA 模型加载即使是 4-bit 量化也要占 2GB+ 显存，**多缓存一个就会撑爆 8GB 显存**；
- 改造后应限制：**LoRA 模型同时只缓存最近 1 个**，加载新 LoRA 前先清理旧的（`del model; torch.cuda.empty_cache()`）。

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
pip install "trl>=1.4.0" "peft>=0.19.0" "bitsandbytes>=0.49.0" "datasets>=4.0.0" "accelerate>=1.0.0"
```

### 7.1 当前环境实测锁定版本（2026-05-18）

| 库 | 实测版本 | Python 3.12.2 / torch 2.6.0+cu124 / transformers 4.57.6 | 备注 |
|---|---|---|---|
| `trl` | **1.4.0** | ✅ SFTConfig/SFTTrainer 新接口可用 | ⚠️ **Windows 编码坑：见 §7.3** |
| `peft` | **0.19.1** | ✅ `prepare_model_for_kbit_training` / `LoraConfig` 正常 | — |
| `bitsandbytes` | **0.49.2** | ✅ Windows wheel 直接装上，无需走 cu121 镜像 | `BitsAndBytesConfig(load_in_4bit=True, nf4, bf16)` 实测通过 |
| `datasets` | **4.8.5** | ✅ | — |
| `accelerate` | **1.13.0** | ✅ | `trl` 强依赖 |

> 上面这一行 pip 命令已实测可用，无需走特殊镜像。

### 7.2 Windows 上 bitsandbytes 三级降级策略（备用）

`bitsandbytes` 在 Windows 上历史上经常翻车，如果未来 wheel 失效，按下面顺序逐级降级：

1. **首选**：`pip install bitsandbytes>=0.49`（0.43.0 后官方已发布 Windows wheel，0.49 已稳定）。本项目实测可直接装上。
2. **次选**：如果 import 时报 `CUDA Setup failed`，安装匹配 CUDA 版本的预编译 wheel：
   ```bash
   pip install bitsandbytes --index-url https://download.pytorch.org/whl/cu121
   ```
3. **兜底**：若仍装不上，训练脚本通过 `--no_4bit` 开关降级到 **bf16 LoRA**（不做 4-bit 量化，显存约 4GB，4060 8GB 跑 1.5B 仍然够用），不阻塞主流程。

`train_lora_sft.py` 启动时会自动探测 `bitsandbytes` 是否可用，不可用时打印 warning 并自动走 bf16 路径（不需要用户手动加 `--no_4bit`）。

### 7.3 ⚠️ Windows + trl 1.x 必须用 UTF-8 模式启动 Python

实测发现：`trl 1.4.0` 的内部源码（`trl/trainer/sft_trainer.py` 等）含有 emoji 字符，但 .py 文件未声明 `# coding: utf-8`。Windows 上 Python 3.12 默认用 GBK 读取源文件，导致 `from trl import SFTTrainer` **直接 import 失败**：

```
RuntimeError: Failed to import trl.trainer.sft_trainer ...
'gbk' codec can't decode byte 0x9c in position 665: illegal multibyte sequence
```

**解决方案（必须做）**：所有需要 import trl 的 Python 进程必须以 UTF-8 模式启动，二选一：

- 启动参数：`python -X utf8 trainer/train_lora_sft.py ...`
- 环境变量：`PYTHONUTF8=1`（推荐，从 `train_manager.py` 子进程环境注入，对用户透明）

`train_manager.py` 已经为 MiniMind 训练设置了 `PYTHONIOENCODING=utf-8`，**还需要追加 `PYTHONUTF8=1`** 才能让 LoRA 训练子进程跑起来。这是必须改的最小一处后端改动（即使 Phase 2 还没动）。

---

---

## 8. 开发计划

### Phase 1：安装依赖 + QLoRA 训练脚本（核心） ✅ 已完成

| 步骤 | 任务 | 状态 |
|---|---|---|
| 1.1 | 安装 Python 依赖（trl、peft、bitsandbytes 等） | ✅ |
| 1.2 | 新建 `trainer/model_registry.py` — 模型注册表 | ✅ |
| 1.3 | 新建 `trainer/train_lora_sft.py` — QLoRA 训练脚本 | ✅ |
| 1.4 | 命令行测试训练脚本，确认能跑通 | ✅ |

**Phase 1 实测验收数据**（2026-05-18，RTX 4060 8GB / Windows 10）：

```
$ python -X utf8 trainer/train_lora_sft.py \
    --model_path models/deepseek-r1-1.5b \
    --data_path dataset/xiaohongshu.json \
    --epochs 1 --batch_size 1 --learning_rate 2e-4 \
    --max_seq_len 512 --accumulation_steps 4 \
    --save_name smoke_deepseek_v1 --save_dir ./out
```

| 指标 | 实测值 |
|---|---|
| 模型加载耗时 | 4.0 s |
| 训练总耗时（10 条样本） | 16.7 s |
| 量化模式 | 4-bit NF4 |
| 可训练参数 | 18.46M / 1140.45M (1.62%) |
| Loss 收敛 | 4.50 → 4.05（10 步） |
| **显存峰值** | **3.92 GB** |
| 产物 adapter 大小 | 70.5 MB |
| 日志格式 | `Epoch [1/1] Step [1/10] loss=4.4990 avg=4.4990 lr=2.00e-04 eta=0.1min` ✅ 严格对齐 MiniMind 版 |

产物目录 `out/smoke_deepseek_v1_lora/` 包含：
```
adapter_config.json        ← LoRA 配置（peft 标准）
adapter_model.safetensors  ← LoRA 增量权重 (70 MB)
tokenizer.json             ← tokenizer 一并保存，推理时不必再去基座目录找
tokenizer_config.json
special_tokens_map.json
chat_template.jinja
minimind_o_meta.json       ← 自定义 meta：base_model_path / quant_mode / 训练超参
README.md
```

### Phase 1 设计决策（备查）

实际开发中做了几个相对原计划的调整，记录在此：

1. **不依赖 `trl.SFTTrainer`，手写训练循环**
   - 原因：trl 1.4.0 的 callback 接口跟 WebUI 进度条日志格式不好对齐；
   - 收益：直接复用 `train_text_sft.py` 里那套已经验证过的 `ChatDataset`（assistant-only labels masking），数据处理零回归；日志格式 100% 可控；不被 trl 版本漂移绑死。
2. **复用 `ChatDataset` / `collate_fn` / `get_lr`**
   - 这三个函数原本就只依赖 tokenizer 的 `apply_chat_template`，不依赖任何 MiniMind 模型代码，DeepSeek/Qwen tokenizer 拿来即用。
3. **梯度累积默认 8 步**
   - LoRA 训练 batch 通常很小（=1），靠累积凑等效大 batch；前端「难度档位」之后传 `accumulation_steps` 进来时按需覆盖。
4. **保存目录里 dump `minimind_o_meta.json`**
   - 记录基座路径 + 量化模式，Phase 2 改造 `_load_test_model` 时可以直接读这个 meta 来反查基座，不必让 server 维护额外索引。

**验收结论**：✅ 完全打通。可以进入 Phase 2 后端改造。

### Phase 2：改造后端（train_manager + train_server） ✅ 已完成

| 步骤 | 任务 | 状态 |
|---|---|---|
| 2.1 | 修改 `train_manager.py` — 加入 model_type 分支 + `PYTHONUTF8=1` | ✅ |
| 2.2a | 修改 `train_server.py` — 新增 `/api/model_types` + `/api/model_type_config` | ✅ |
| 2.2b | 修改 `/api/start_train` — 接受 `model_type`，按 family 分支处理参数 | ✅ |
| 2.2c | 修改 `/api/models` + `/api/delete_model` — 同时识别 `.pth` 与 `_lora/` 目录 | ✅ |
| 2.2d | 修改 `/api/quick_test` + `_load_test_model` — 支持 LoRA 加载 + 单例缓存 | ✅ |
| 2.3 | 后端联调（HTTP + WebSocket 双层）+ MiniMind 回归 | ✅ |

**Phase 2 实测验收**（2026-05-18）：

| 测试项 | 实测 |
|---|---|
| 8 个 API 单元测试（`scripts/_smoke_api.py`） | ✅ 全过 |
| HTTP → 子进程拉起 → 训练真实运行（`scripts/_smoke_api_e2e.py`） | ✅ 11.8s 内出现首条 step 日志 |
| Step 日志格式匹配进度条正则 `Epoch \[(\d+)/(\d+)\] Step \[(\d+)/(\d+)\] loss=([\d.]+)` | ✅ `Epoch [1/1] Step [1/10] loss=4.7163 ...` |
| WebSocket `/ws/train_log` 实时推送 | ✅ 启动 5s 内收到首批日志 |
| `/api/stop_train` 优雅终止 subprocess | ✅ |
| **MiniMind 回归**（`scripts/_smoke_api_minimind_regression.py`） | ✅ 不传 `model_type` 默认 minimind；日志格式不变；17.4s 跑出首条 step |
| `/api/models` 同时识别 `_lora/` 目录 | ✅ `smoke_deepseek_v1_lora` 被正确标为 `format=lora, model_type=deepseek-r1-1.5b` |
| 未知 `model_type` 返回 400 | ✅ |

### Phase 2 设计决策（备查）

1. **LoRA `_load_test_model` 用 meta 文件反查基座路径**
   - 训练时把 `minimind_o_meta.json` 写到产物目录里，`_load_test_model_lora` 直接读这个文件拿基座路径；不需要 server 维护额外索引。
   - 退化路径：meta 缺失时读 `adapter_config.json.base_model_name_or_path`；再退化按目录名匹配注册表。
2. **`_TEST_CACHE` 单例策略只对 LoRA 生效**
   - MiniMind 体积小（~700MB），多缓存几个无所谓；
   - LoRA 即使 4-bit 也要 2GB+，新加载一个就 `_evict_lora_cache()` 释放掉之前所有的，配合 `torch.cuda.empty_cache()`。
3. **`/api/start_train` 用 `family` 分支处理「专属字段」**
   - MiniMind 分支保留原来的 `base_model` 短名拼接逻辑（向后兼容前端）；
   - DeepSeek 分支完全不走 `f"{base_model}_768.pth"` 拼接，从注册表里直接拿基座目录路径；
   - LoRA 专属字段 `lora_r / lora_alpha / lora_dropout / max_seq_len / accumulation_steps / no_4bit` 透传到 manager。
4. **删除模型加二次防御**
   - `os.path.abspath(path).startswith(OUT_DIR + os.sep)` 防符号链接路径穿越；
   - 目录形态只允许删 `_lora` 后缀的，防止用户输 `.` 或 `models` 误删基座目录。

**验收结论**：✅ 整条后端链路通畅。可以进入 Phase 3 前端改造。

### Phase 3：改造前端（web_train.html） ✅ 已完成

| 步骤 | 任务 | 状态 |
|---|---|---|
| 3.1 | 新增 CSS：模型类型卡片 / LoRA 面板 / LoRA badge / family-条件显示 | ✅ |
| 3.2 | Step 2 HTML：插入模型类型选择区 + LoRA 参数面板（4 字段） | ✅ |
| 3.3 | Step 2 JS：`loadModelTypes` / `selectModelType` 动态拉 presets；start_train 带 `model_type` 和 LoRA 参数 | ✅ |
| 3.4 | Step 3 JS：模型列表/下拉/badge 适配 `format` + `model_type` 字段 | ✅ |
| 3.5 | 浏览器端到端联调（含 LoRA 真实推理） | ✅ |

**Phase 3 实测验收**（Cursor 内置浏览器实操）：

| 测试 | 结果 |
|---|---|
| 启动后默认选中 MiniMind，显示 `基础模型` 下拉，难度档位预览 `lr=0.00002` | ✅ |
| 点击 DeepSeek 卡片：`基础模型` 下拉消失、`LoRA 参数` 入口出现、难度档位预览改为 `lr=0.0002` | ✅ |
| 展开 LoRA 面板：Rank/Alpha/Dropout/MaxSeqLen 默认值 `16/32/0.05/1024` 来自注册表 | ✅ |
| 提示语随类型切换（MiniMind「产物是单个 .pth」/ DeepSeek「LoRA 适配器目录」） | ✅ |
| 真启动 DeepSeek 训练 → 进度条实时推进 → 完成后自动跳 Step 3 | ✅ |
| Step 3 模型表格：紫色 `LoRA` badge + `DeepSeek-1.5B` 次级 badge；旧的 SFT/MiniMind 颜色不变；`llm_768.pth` 标「受保护」 | ✅ |
| LoRA quick_test 真实推理：4-bit 量化加载 → adapter 加载 → 输出真实文案（含 R1 的 `</think>` 标记） | ✅ |

### Phase 3 设计决策（备查）

1. **复用 `.diff-card` 视觉系统做模型类型卡片**
   - 新增 `.mt-card` 沿用 diff-card 的边框/选中态/字体；用户不会感到突兀。
   - 增加 `.mt-card.disabled` 灰显未下载基座（如未来加 7B 但用户没下时），点击不响应。
2. **CSS body class 控制 family 条件显示**
   - `body.fm-minimind .only-minimind {display: flex}` 这种 `body class + only-*` 模式，可以在 HTML 上零成本切换：每个组件标 `class="only-minimind"` 或 `class="only-hf_causal_lm"`，由 `selectModelType` 统一切 body class。
   - 未来加新 family（如 `family: "diffusion"`）只需要加一条 CSS 规则，HTML 不用动。
3. **`DIFF_PRESETS` 从常量改为动态变量**
   - 切换模型类型时 `selectModelType` 重写 `DIFF_PRESETS`、同步刷新三张档位卡的副标题，然后按当前 difficulty 重新填表单——用户无感切换。
4. **`loadBaseModels` 只列 `.pth` 且 `model_type=minimind`**
   - 防止用户把 DeepSeek 的 LoRA 目录误选成 MiniMind 的「基础模型」。
5. **模型下拉/表格的「次级 badge」展示 `model_type`**
   - 新增 `.badge-mt` 灰色边框 badge，挂在主 badge（LoRA/SFT/基座）右边，一眼看出每个模型是哪种基座训出来的。

**验收结论**：✅ 整个前端流程通畅。MiniMind 和 DeepSeek 两条路径都跑通了 → 训练 → 试试效果。**整个多模型训练系统（Phase 1-3）开发完成**。

### Phase 4：端到端测试 + 文档 ✅ 已并入 Phase 2/3

| 步骤 | 任务 | 状态 |
|---|---|---|
| 4.1 | 完整流程测试（MiniMind 训练不受影响） | ✅ Phase 2 `_smoke_api_minimind_regression.py` + Phase 3 浏览器实操 |
| 4.2 | 更新本文档为最终版 | ✅ v1.1（含 Phase 1/2/3 完整验收数据） |

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
| **bitsandbytes 兼容性** | Windows 下 bitsandbytes ≥ 0.43 才有官方 wheel，旧环境常装失败 | 三级降级：①官方 wheel → ② cu121 预编译 wheel → ③ 训练脚本自动检测并降到 bf16 LoRA |
| **trl 接口漂移** | trl 0.7 / 0.11 / 0.14 三个版本 SFTTrainer/SFTConfig 接口都有 break change | `requirements.txt` 锁 `trl>=0.11,<0.13`；脚本里用兼容写法（datasets 传入 + `dataset_text_field` 或 `formatting_func` 二选一） |
| **进度条解析对齐** | `web_train.html` 用正则匹配 `Epoch [x/y] Step [a/b] loss=... avg=... lr=... eta=...` 推进进度条 | LoRA 训练脚本必须输出完全一样的格式，否则前端会一直停留在「准备中…」 |
| **`base_model` 字段冲突** | `train_server.py` 现在拼 `f"{base_model}_768.pth"`，是 MiniMind 写死的 | 改造后按 `model_type` 分支：MiniMind 走原路径，DeepSeek 直接用注册表里的 `base_model_path` |
| **「试试效果」显存爆炸** | LoRA 模型缓存多个会撑爆 8GB | `_TEST_CACHE` 对 LoRA 模型只保留 1 个最近用过的，加载新模型前先释放 |
| **训练 ↔ 测试 GPU 互斥** | 训练运行时 quick_test 会冲突 OOM | 沿用现有「训练中禁用 quick_test」逻辑，对 LoRA 同样适用 |
| **显存不足（7B）** | 7B 模型即使 4-bit 也需要 6-8GB | 先从 1.5B 开始；7B 需 batch_size=1 + gradient_checkpointing |
| **训练速度慢** | 1.5B QLoRA 每步约 2-3 秒；7B 约 5-10 秒 | 数据量小时影响不大；页面提示预计时间 |
| **DeepSeek-R1 思维链格式** | R1 模型会输出 `<think>…</think>` 推理过程 | 微调数据不需要包含思维链；推理时如不想要可在 quick_test 里 strip 掉这一段 |
| **LoRA 合并部署** | LoRA 产物只有 adapter，需与基础模型合并才能推理 | quick_test 用 `PeftModel.from_pretrained(base, adapter_dir)` 在线合并，无需落盘 |
| **MiniMind 回归** | 改动不能破坏现有 MiniMind 训练 | 每个 Phase 完成后都跑一次 MiniMind 训练做回归 |
| **难度档位语义差异** | 现有「快速/正常/深度」三档参数是 MiniMind 全参数 SFT 的语义（lr=2e-5、batch=8） | 前端切到 DeepSeek 时，档位参数要从注册表里取（LoRA 习惯 lr=2e-4，batch 1~2，epoch 多一些） |

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
