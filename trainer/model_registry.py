"""
模型注册表
===========

把「模型类型 → 训练脚本 / 基座路径 / 产物格式 / 默认参数」这套元数据集中
管理，让上层（train_manager、train_server、前端）不再硬编码模型名字。

设计原则
--------
1. **新增模型 = 加一条配置**：未来想接 Qwen / Llama / 7B 模型，只要在
   ``MODEL_REGISTRY`` 里 append 一条，下游的 manager / server / 前端会
   自动识别。
2. **基座保护**：MiniMind 走 ``out/llm_768.pth``（单文件 .pth），DeepSeek
   走 ``models/deepseek-r1-1.5b/``（HuggingFace 目录）。两者完全不同的产物
   形态——单文件 vs 目录——通过 ``save_format`` 字段区分，后续 server 侧
   的扫描 / 删除 / 加载逻辑按此分支。
3. **训练难度档位**：保留前端「快速 / 正常 / 深度」三档，但具体参数从这里取
   （MiniMind 全参数 SFT 和 DeepSeek QLoRA 的合理参数区间差异很大，
   全参数常用 lr=2e-5，QLoRA 常用 lr=2e-4）。
"""

from __future__ import annotations

import os
from typing import Any


# ── 项目根目录（用于把相对路径解析成绝对路径） ────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(p: str) -> str:
    """注册表内的相对路径统一相对项目根目录"""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_ROOT, p))


# ── 模型类型注册表 ────────────────────────────────────────────────────
#
# 字段说明：
#   display_name      前端下拉框里展示的友好名字
#   family            "minimind" / "hf_causal_lm"——决定走哪条训练 / 推理路径
#   train_script      相对项目根的脚本路径，由 train_manager subprocess 拉起
#   base_model_path   基座权重位置（MiniMind 是 .pth 文件，HF 系是目录）
#   tokenizer_path    tokenizer 位置（MiniMind 单独维护在 model/；HF 系跟基座同目录）
#   save_format       产物形态："pth"（单文件）/ "lora"（HF adapter 目录）
#   save_ext          产物文件后缀 / 目录后缀（用户填 save_name 后拼出真正名字）
#   vram_hint         显存占用粗略提示（写给前端展示用）
#   presets           难度档位 → 训练参数；前端「快速/正常/深度」三档由此填充
#   extra_args        训练脚本额外需要的命令行参数（如 LoRA 的 rank/alpha）
#
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "minimind": {
        "display_name": "MiniMind-0.1B（全参数 SFT）",
        "family": "minimind",
        "train_script": "trainer/train_text_sft.py",
        "base_model_path": _abs("out/llm_768.pth"),
        "tokenizer_path": _abs("model"),
        "save_format": "pth",
        "save_ext": "_768.pth",
        "vram_hint": "约 1.5 GB",
        "presets": {
            "quick":  {"epochs": 1, "batch_size": 2, "learning_rate": 2e-5},
            "normal": {"epochs": 3, "batch_size": 2, "learning_rate": 2e-5},
            "deep":   {"epochs": 8, "batch_size": 2, "learning_rate": 1e-5},
        },
        "extra_args": {},  # 全参数 SFT 没有 LoRA 之类的额外参数
    },

    "deepseek-r1-1.5b": {
        "display_name": "DeepSeek-R1-Distill-Qwen-1.5B（QLoRA 微调）",
        "family": "hf_causal_lm",
        "train_script": "trainer/train_lora_sft.py",
        "base_model_path": _abs("models/deepseek-r1-1.5b"),
        "tokenizer_path": _abs("models/deepseek-r1-1.5b"),
        "save_format": "lora",
        "save_ext": "_lora",
        "vram_hint": "约 3~4 GB（4-bit）/ 4~5 GB（bf16）",
        # 注意：QLoRA 的 lr 普遍比全参数 SFT 大一个数量级（2e-4 vs 2e-5），
        # 这是 LoRA 适配器维度低、收敛慢导致的经验值。
        "presets": {
            "quick":  {"epochs": 1, "batch_size": 1, "learning_rate": 2e-4},
            "normal": {"epochs": 3, "batch_size": 1, "learning_rate": 2e-4},
            "deep":   {"epochs": 6, "batch_size": 1, "learning_rate": 1e-4},
        },
        "extra_args": {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "max_seq_len": 1024,
        },
    },

    # ── 未来扩展占位（保留注释，方便复制改） ──
    # "deepseek-r1-7b": { ... "base_model_path": _abs("models/deepseek-r1-7b"), ... },
    # "qwen2.5-0.5b":   { ... },
}


# 默认模型类型——给 train_manager 在 params 没传 model_type 时兜底，
# 保证现有 MiniMind 调用方完全不感知改造
DEFAULT_MODEL_TYPE = "minimind"


def list_model_types() -> list[dict[str, Any]]:
    """供 ``GET /api/model_types`` 用——只暴露前端要看的字段。"""
    out = []
    for type_id, cfg in MODEL_REGISTRY.items():
        out.append({
            "id": type_id,
            "display_name": cfg["display_name"],
            "family": cfg["family"],
            "save_format": cfg["save_format"],
            "vram_hint": cfg["vram_hint"],
            "base_available": _base_exists(cfg),
        })
    return out


def get_config(type_id: str) -> dict[str, Any]:
    """按 type_id 取完整配置；不存在直接 KeyError。"""
    if type_id not in MODEL_REGISTRY:
        raise KeyError(
            f"未知 model_type: {type_id!r}，可选: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[type_id]


def get_preset(type_id: str, level: str) -> dict[str, Any]:
    """按 (model_type, 难度档位) 取一组默认训练参数。

    level 可选: "quick" / "normal" / "deep"；未知档位返回 "normal" 兜底。
    """
    cfg = get_config(type_id)
    presets = cfg["presets"]
    return presets.get(level, presets["normal"])


def _base_exists(cfg: dict[str, Any]) -> bool:
    """基座模型是否在本地落盘，给前端展示「未下载」用。"""
    path = cfg["base_model_path"]
    if cfg["save_format"] == "pth":
        return os.path.isfile(path)
    # HF 目录至少要有 config.json
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def build_save_path(type_id: str, save_name: str, save_dir: str) -> str:
    """根据模型类型把用户填的 ``save_name`` 拼成最终落盘路径。

    - MiniMind:  ``out/{save_name}_768.pth``        （单个 .pth 文件）
    - DeepSeek:  ``out/{save_name}_lora``           （目录，里面放 adapter）
    """
    cfg = get_config(type_id)
    return os.path.normpath(os.path.join(save_dir, save_name + cfg["save_ext"]))
