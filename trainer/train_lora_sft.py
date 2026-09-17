"""
通用 QLoRA SFT 训练脚本 —— DeepSeek-R1 / Qwen / Llama 等 HuggingFace 因果模型
============================================================================

为什么不用 ``trl.SFTTrainer``
-----------------------------
* trl 1.x 的 callback 接口跟我们 WebUI 用的进度条日志格式不好对齐；
* 手写循环可以**直接复用** ``train_text_sft.py`` 里那套已经验证过的
  ChatDataset（assistant-only labels masking），数据处理零回归；
* 不被 trl 版本漂移绑死。

设计要点
--------
1. **日志格式严格对齐 MiniMind**：每一步 step 都打 ``Epoch [x/y] Step [a/b]
   loss=... avg=... lr=... eta=...`` 这一行；WebUI 进度条按这个正则解析。
2. **bitsandbytes 自动降级**：4-bit 不可用时自动走 bf16 LoRA，不阻塞用户。
3. **产物只保存 LoRA adapter**：``out/{save_name}_lora/`` 目录，含
   ``adapter_config.json + adapter_model.safetensors + tokenizer*``，几十 MB。
4. **Windows UTF-8 防爆**：与 MiniMind 训练脚本一致，强制 stdout/stderr 用
   UTF-8；train_manager 也会注入 ``PYTHONUTF8=1`` 让 trl import 不挂。

用法示例
--------
::

    python -X utf8 trainer/train_lora_sft.py \
        --model_path models/deepseek-r1-1.5b \
        --data_path dataset/xiaohongshu.json \
        --epochs 3 --batch_size 1 --learning_rate 2e-4 \
        --save_name xhs_deepseek_v1 --save_dir ./out \
        --lora_r 16 --lora_alpha 32
"""

import os
import sys
import math
import time
import argparse
import warnings

# ── Windows 终端编码兜底（同 train_text_sft.py） ──
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from torch import optim
from torch.utils.data import DataLoader

__package__ = "trainer"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 复用 MiniMind 训练脚本里那套数据集（apply_chat_template + assistant-only mask），
# 它本身不依赖任何 MiniMind 模型代码，对 DeepSeek/Qwen tokenizer 同样适用。
from trainer.train_text_sft import ChatDataset, collate_fn, get_lr  # noqa: E402

warnings.filterwarnings("ignore")


# ── bitsandbytes 健康检查 ───────────────────────────────────────────
def _probe_bnb() -> bool:
    """探测 bitsandbytes 是否真的能用（光 import 不够，得能构造 4-bit 配置）"""
    try:
        import bitsandbytes as _bnb  # noqa: F401
        from transformers import BitsAndBytesConfig
        # 构造一下，触发底层 CUDA setup
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.bfloat16)
        return True
    except Exception as e:
        print(f"  ⚠️ bitsandbytes 不可用（{type(e).__name__}: {e}），将自动降级到 bf16 LoRA")
        return False


# ── 主流程 ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("通用 QLoRA SFT (DeepSeek-R1 / Qwen / Llama)")
    # —— 与 MiniMind 版同名同义的参数（让 train_manager 拼 cmd 时少写分支）——
    parser.add_argument("--data_path", required=True, help="训练数据 JSON 路径")
    parser.add_argument("--save_dir", default="./out", help="模型保存目录")
    parser.add_argument("--save_name", default="lora_sft", help="保存目录前缀")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--accumulation_steps", type=int, default=8,
                        help="梯度累积步数；LoRA 训练 batch 通常很小，靠累积凑等效大 batch")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Windows 上 DataLoader workers > 0 经常出问题，默认 0")

    # —— LoRA / QLoRA 专属参数 ——
    parser.add_argument("--model_path", required=True,
                        help="HF 基座模型目录，例如 models/deepseek-r1-1.5b")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_4bit", action="store_true",
                        help="强制走 bf16 LoRA，不用 4-bit 量化")
    parser.add_argument("--resume_from_lora", default=None,
                        help="可选：从已有 LoRA adapter 目录继续训练。"
                             "传了这个之后，lora_r/lora_alpha/lora_dropout/target_modules "
                             "由该目录的 adapter_config.json 决定，命令行的对应参数会被忽略。")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.save_name + "_lora")
    os.makedirs(save_path, exist_ok=True)

    # ── 1. 加载 tokenizer ──
    from transformers import AutoTokenizer
    print(f"📝 加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # DeepSeek/Qwen 系常常没设 pad_token，用 eos 顶上
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. 决定走 4-bit 还是 bf16 ──
    use_4bit = (not args.no_4bit) and (args.device != "cpu") and _probe_bnb()
    quant_mode = "4-bit NF4" if use_4bit else f"{args.dtype} LoRA (无量化)"

    print(f"🧠 加载基座模型: {args.model_path}")
    print(f"  量化模式: {quant_mode}")

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_cfg,
            device_map={"": 0} if args.device != "cpu" else None,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=compute_dtype,
            device_map={"": 0} if args.device != "cpu" else None,
            trust_remote_code=True,
        )

    # ── 3. 注入 LoRA ──
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training, TaskType

    if use_4bit:
        # 4-bit 训练需要这一步打开 input grad、把 LN 升回 fp32 等
        model = prepare_model_for_kbit_training(model)

    if args.resume_from_lora:
        # —— 续训：从已有 adapter 加载，跳过结构注入 ——
        # is_trainable=True 关键，否则加载的 adapter 默认是 inference 模式（requires_grad=False）
        if not os.path.isdir(args.resume_from_lora):
            print(f"❌ resume_from_lora 不是有效目录: {args.resume_from_lora}")
            sys.exit(1)
        if not os.path.isfile(os.path.join(args.resume_from_lora, "adapter_config.json")):
            print(f"❌ 该目录缺少 adapter_config.json，不是有效的 LoRA 产物: {args.resume_from_lora}")
            sys.exit(1)
        print(f"🔁 续训模式：从 {args.resume_from_lora} 加载已有 LoRA adapter")
        model = PeftModel.from_pretrained(model, args.resume_from_lora, is_trainable=True)
        # 从加载好的 PeftConfig 反查实际的 r/alpha/dropout，覆盖命令行的（仅用于日志展示）
        peft_cfg = model.peft_config[model.active_adapter]
        eff_r = peft_cfg.r
        eff_alpha = peft_cfg.lora_alpha
        eff_dropout = peft_cfg.lora_dropout
    else:
        # —— 全新 LoRA：按命令行参数注入 ——
        # 目标模块：DeepSeek-R1-Distill-Qwen-1.5B 是 Qwen2 架构，下面 7 个 proj 是标准
        # 注入点；其他 HF 因果模型大多也是这套名字，复用即可。
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        eff_r, eff_alpha, eff_dropout = args.lora_r, args.lora_alpha, args.lora_dropout

    # 打印「可训练 / 总参数」给用户一个直观感受
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    src = f"续训自 {args.resume_from_lora}" if args.resume_from_lora else "全新初始化"
    print(f"  LoRA rank: {eff_r}, alpha: {eff_alpha}, dropout: {eff_dropout}  [{src}]")
    print(f"  可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M "
          f"({100*trainable/max(total,1):.2f}%)")

    # ── 4. 加载数据 ──
    print(f"📚 加载数据: {args.data_path}")
    ds = ChatDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    print(f"  样本数: {len(ds)}, epochs: {args.epochs}, batch_size: {args.batch_size}")
    if len(ds) == 0:
        print("❌ 数据集为空，终止训练")
        sys.exit(1)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"),
    )

    # ── 5. 优化器 + 显存提示 ──
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    total_steps = args.epochs * max(1, math.ceil(len(loader) / args.accumulation_steps))

    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 自适应日志间隔：和 MiniMind 版同款策略，保证小数据集每步都打
    steps_per_epoch = max(1, len(loader))
    log_interval = max(1, min(args.log_interval, steps_per_epoch // 10 or 1))
    print(f"🚀 开始训练 | epochs={args.epochs} | batch={args.batch_size} | "
          f"accum={args.accumulation_steps} | lr={args.learning_rate} | "
          f"steps_per_epoch={steps_per_epoch} | log_interval={log_interval}")

    # ── 6. 训练循环 ──
    # 注意：这里日志格式必须和 train_text_sft.py 完全一致，否则前端进度条会失效
    # 关键正则： Epoch \[(\d+)/(\d+)\] Step \[(\d+)/(\d+)\] loss=([\d.]+)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        optimizer.zero_grad(set_to_none=True)

        for step, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 学习率调度：和 MiniMind 同款 cosine
            lr = get_lr(global_step, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # 4-bit + bf16 计算下不需要 GradScaler；bf16 也不需要
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / args.accumulation_steps
            loss.backward()

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += loss.item() * args.accumulation_steps

            if (step + 1) % log_interval == 0 or (step + 1) == steps_per_epoch:
                avg = epoch_loss / (step + 1)
                steps_done = epoch * steps_per_epoch + (step + 1)
                total_inner = args.epochs * steps_per_epoch
                steps_left = total_inner - steps_done
                eta_min = (time.time() - start_time) / max(step + 1, 1) * steps_left / 60
                # ⚠️ 这一行格式必须严格对齐 train_text_sft.py，不要改顺序/字段
                print(f"  Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{steps_per_epoch}] "
                      f"loss={loss.item() * args.accumulation_steps:.4f} "
                      f"avg={avg:.4f} lr={lr:.2e} eta={eta_min:.1f}min")

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"✅ Epoch [{epoch+1}/{args.epochs}] 完成 | avg_loss={avg_loss:.4f} | "
              f"耗时={(time.time() - start_time) / 60:.1f}min")

    # ── 7. 保存 LoRA adapter ──
    print(f"💾 保存 LoRA 权重到: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 记一个 meta 文件，记录基座路径 + 续训来源（若有），方便推理时反查、后续追溯训练血统
    with open(os.path.join(save_path, "minimind_o_meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({
            "base_model_path": args.model_path,
            "quant_mode": "4bit_nf4" if use_4bit else "no_quant",
            "compute_dtype": args.dtype,
            "lora_r": eff_r,
            "lora_alpha": eff_alpha,
            "lora_dropout": eff_dropout,
            "resume_from_lora": args.resume_from_lora,   # v2.2：续训血统记录
            "train_samples": len(ds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        }, f, ensure_ascii=False, indent=2)

    if args.device != "cpu" and torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  显存峰值: {peak_gb:.2f} GB")
    print(f"🏁 训练完毕，LoRA 权重: {save_path}")


if __name__ == "__main__":
    main()
