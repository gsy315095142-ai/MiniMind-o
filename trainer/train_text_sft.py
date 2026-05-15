"""
纯文本 SFT 训练脚本 —— 基于 MiniMindForCausalLM（Thinker only）
不需要 Mimi / SenseVoice / SigLIP / Talker，只练语言能力。

用法示例（单卡）：
  python trainer/train_text_sft.py \
    --data_path ./dataset/xiaohongshu.json \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --save_name xhs

数据格式（JSON 文件，每行一个对象，或一个数组）：
  [
    {
      "messages": [
        {"role": "user", "content": "帮我写一篇小红书探店文案"},
        {"role": "assistant", "content": "📍杭州｜藏在巷子里的宝藏咖啡馆☕️…"}
      ]
    },
    ...
  ]
"""

import os, sys, json, argparse, math, time, warnings

# ── Windows 终端编码兜底 ──
# 训练脚本里大量使用了 emoji（📝 🚀 ✅ ...），但 Windows 上 Python 默认 stdout 编码是 GBK，
# 直接 print emoji 会触发 UnicodeEncodeError。两道保险：
#  1. 设置 PYTHONIOENCODING=utf-8 环境变量（对子进程生效）
#  2. 通过 stdout.reconfigure 把当前进程的 stdout/stderr 重设为 UTF-8 + errors='replace'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass  # 老 Python / 重定向场景没有这个方法，忽略

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

__package__ = "trainer"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

warnings.filterwarnings('ignore')


# ── 数据集 ───────────────────────────────────────────
IGNORE_INDEX = -100


class ChatDataset(Dataset):
    """标准 chat format → input_ids + labels

    labels 策略：只对 assistant 回复部分计算 loss，user / system 部分用 -100 mask 掉。

    实现方式：增量前缀编码
        1. 一次性 tokenize 整段对话，得到 input_ids
        2. 逐 i 编码 msgs[:i+1]，得到该前缀的 token 长度 cur_len
        3. 若 msgs[i] 是 assistant，则把 [prev_len, cur_len) 区段从 -100 还原为真实 token
    这种「前缀单调累加」的方式比逐 turn 独立编码再拼接稳健得多——
    后者会在 turn 边界处出现 special-token 不对齐，导致整段 labels 全被 mask（loss=NaN）。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # 兼容两种格式：数组 / 单个对象
        if isinstance(raw, list):
            samples = raw
        elif isinstance(raw, dict) and 'messages' in raw:
            samples = [raw]
        else:
            raise ValueError("不支持的数据格式，期望 [{...}, ...] 或 {'messages':[...]}")

        self.data = []
        skipped = 0
        for s in samples:
            msgs = s.get('messages') or s.get('conversation') or s
            if not isinstance(msgs, list) or len(msgs) < 2:
                skipped += 1
                continue

            # 1) 整体编码
            try:
                full_ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False
                )
            except Exception:
                skipped += 1
                continue

            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]
            input_ids = torch.tensor(full_ids, dtype=torch.long)
            labels = torch.full_like(input_ids, IGNORE_INDEX)

            # 2) 增量前缀编码，定位每个 turn 的 token 区间
            prev_len = 0
            for i, m in enumerate(msgs):
                try:
                    cur_ids = tokenizer.apply_chat_template(
                        msgs[:i + 1], tokenize=True, add_generation_prompt=False
                    )
                except Exception:
                    break
                cur_len = min(len(cur_ids), len(input_ids))
                # 只对 assistant turn 解 mask（含其 role header，简单稳定）
                if m.get('role') == 'assistant' and cur_len > prev_len:
                    labels[prev_len:cur_len] = input_ids[prev_len:cur_len]
                prev_len = cur_len
                if prev_len >= len(input_ids):
                    break

            # 3) 安全门：至少要有几个有效 token 才训练
            if (labels != IGNORE_INDEX).sum().item() < 2:
                skipped += 1
                continue

            self.data.append({'input_ids': input_ids, 'labels': labels})

        if skipped:
            print(f'  ⚠️ 跳过 {skipped} 条不合法/无效样本')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'] for b in batch]
    # padding
    max_len = max(len(ids) for ids in input_ids)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (ids, lbs) in enumerate(zip(input_ids, labels)):
        padded_ids[i, :len(ids)] = ids
        padded_labels[i, :len(lbs)] = lbs
    return padded_ids, padded_labels


# ── 学习率调度 ────────────────────────────────────────
def get_lr(step, total_steps, base_lr):
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total_steps)))


# ── 主训练函数 ───────────────────────────────────────
def main():
    parser = argparse.ArgumentParser("MiniMind 纯文本 SFT")
    parser.add_argument('--data_path', required=True, help='训练数据 JSON 路径')
    parser.add_argument('--base_model', default='../out/llm_768.pth', help='基座权重路径')
    parser.add_argument('--tokenizer_path', default='../model', help='tokenizer 路径')
    parser.add_argument('--save_dir', default='../out', help='模型保存目录')
    parser.add_argument('--save_name', default='text_sft', help='保存权重名前缀')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. 加载 tokenizer ──
    print(f'📝 加载 tokenizer: {args.tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # ── 2. 构建模型 ──
    print('🧠 构建 MiniMindForCausalLM (纯语言模型)')
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)

    # 加载基座权重
    if os.path.exists(args.base_model):
        print(f'📦 加载基座权重: {args.base_model}')
        state = torch.load(args.base_model, map_location='cpu')
        # 移除 loss 临时加载可能带来的多余 key
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'  ⚠️ 缺失 key: {missing[:5]}...' if len(missing) > 5 else f'  ⚠️ 缺失 key: {missing}')
    else:
        print(f'⚠️ 基座权重不存在: {args.base_model}，将从随机初始化开始训练')

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'  可训练参数: {trainable:.2f}M')

    # ── 3. 加载数据 ──
    print(f'📚 加载数据: {args.data_path}')
    ds = ChatDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    print(f'  样本数: {len(ds)}')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=args.num_workers,
                        pin_memory=(args.device != 'cpu'))

    # ── 4. 混合精度 & 优化器 ──
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16
    use_amp = args.device != 'cpu'
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    autocast_ctx = torch.cuda.amp.autocast(dtype=dtype) if use_amp else __import__('contextlib').nullcontext()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = args.epochs * len(loader)

    # ── 5. 训练循环 ──
    # 自适应日志间隔：小数据集每步都打，大数据集按 log_interval 节流，
    # 始终保证每个 epoch 至少打印 10 次进度，便于前端进度条解析。
    steps_per_epoch = max(1, len(loader))
    log_interval = max(1, min(args.log_interval, steps_per_epoch // 10 or 1))
    print(f'🚀 开始训练 | epochs={args.epochs} | batch={args.batch_size} | lr={args.learning_rate} | steps={total_steps} | log_interval={log_interval}')
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, (input_ids, labels) in enumerate(loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            lr = get_lr(global_step, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            with autocast_ctx:
                # MiniMindForCausalLM.forward 已经内置了 loss 计算
                res = model(input_ids=input_ids, labels=labels)
                loss = (res.loss + getattr(res, 'aux_loss', 0)) / args.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * args.accumulation_steps
            global_step += 1

            if (step + 1) % log_interval == 0 or (step + 1) == steps_per_epoch:
                avg = epoch_loss / (step + 1)
                # ETA：基于「整个训练剩余步数」估算，更稳更友好
                steps_done = epoch * steps_per_epoch + (step + 1)
                steps_left = total_steps - steps_done
                eta = (time.time() - start_time) / max(step + 1, 1) * steps_left / 60
                print(f'  Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{steps_per_epoch}] '
                      f'loss={loss.item() * args.accumulation_steps:.4f} avg={avg:.4f} lr={lr:.8f} eta={eta:.1f}min')

            if (step + 1) % args.save_interval == 0:
                save_path = f'{args.save_dir}/{args.save_name}_{config.hidden_size}.pth'
                torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, save_path)
                print(f'  💾 已保存: {save_path}')

        # epoch 结束
        avg_loss = epoch_loss / len(loader)
        print(f'✅ Epoch [{epoch+1}/{args.epochs}] 完成 | avg_loss={avg_loss:.4f} | '
              f'耗时={(time.time() - start_time) / 60:.1f}min')

    # ── 6. 最终保存 ──
    save_path = f'{args.save_dir}/{args.save_name}_{config.hidden_size}.pth'
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, save_path)
    print(f'🏁 训练完毕，最终权重: {save_path}')


if __name__ == '__main__':
    main()
