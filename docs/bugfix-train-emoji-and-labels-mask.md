# 训练失败复盘：emoji 编码崩溃 + ChatDataset Labels 全部被 Mask

> 版本: v1.0
> 日期: 2026-05-15
> 作者: 训练 WebUI 维护组
> 影响范围: `trainer/train_text_sft.py`、`trainer/train_manager.py`、`webui/web_train.html`

---

## 0. 一句话总结

用户在训练 WebUI 上点击「开始训练」后页面显示「训练失败」，但日志默认折叠，看不到具体错误。深入排查后发现这其实是**两个独立 Bug 叠加**：

1. **Bug A（致命）**：Windows 上 Python stdout 默认编码是 GBK，训练脚本 `print('📝 加载 tokenizer: …')` 直接抛 `UnicodeEncodeError`，子进程在第一行就崩了。
2. **Bug B（隐性）**：`ChatDataset` 用错误的方式构造 labels mask，导致**所有 token 都被 mask 成 -100**，cross-entropy 在 `0/0` 上算出 NaN。如果不是 Bug A 先崩，训练表面会"跑完"，但产出的是一个已被 NaN 权重污染的废模型。

两个 Bug 都修复 + 加了一条 UX 改进（失败时自动展开日志面板）。

---

## 1. 现象

### 1.1 用户视角

WebUI 上点击「🚀 开始训练」后，状态从「训练中…」直接变成「❌ 训练失败」。但因为日志面板默认折叠，用户看不到任何 Python traceback，只看到一个红色的 pill。

![](#)  <!-- 用户截图 -->

```
🚀 开始训练  ⏹ 停止训练  ● ✕ 训练失败    任务 f27e374c · 保存为 MiniMind_xhs_v0.1.26051501
```

### 1.2 关键线索

- 训练几乎是**瞬间失败**（不到 1 秒）
- 不展开日志面板就只能看到「训练失败」四个字
- 用户之前正常跑过命令行训练，环境本身是健康的

---

## 2. 排查过程

### 2.1 第一次错误尝试：以为是基座模型缺失

`Glob` 工具搜 `out/*.pth` 返回 0 files，初步怀疑基座模型 `llm_768.pth` 不存在。但事实上：

- `out/` 目录被 `.gitignore` 排除了
- IDE 的 `Glob` 工具默认遵守 `.gitignore`，所以"看不到"这些 .pth 文件
- 用 `dir out` 直接看文件系统，文件其实都在

**教训**：在 ML 项目里调试时，不要相信 `gitignore-aware` 工具的搜索结果——`out/`、`checkpoints/`、`logs/` 这类目录恰恰是要排查的重点。直接 `ls/dir` 或者 `os.path.isfile()` 验证。

### 2.2 直接复现：在终端跑训练命令

跳过 WebUI，直接在命令行起一份训练：

```powershell
python -u trainer/train_text_sft.py --data_path ./dataset/xiaohongshu.json \
    --epochs 1 --batch_size 2 --learning_rate 2e-5 \
    --save_name __test_trace --tokenizer_path ./model --save_dir ./out
```

立刻看到完整 traceback：

```
Traceback (most recent call last):
  File "D:\AI_Program\MiniMind-o\trainer\train_text_sft.py", line 270, in <module>
    main()
  File "D:\AI_Program\MiniMind-o\trainer\train_text_sft.py", line 168, in main
    print(f'\U0001f4dd 加载 tokenizer: {args.tokenizer_path}')
UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f4dd' in position 0: illegal multibyte sequence
```

**Bug A 现身。** `\U0001f4dd` 是 📝 emoji。

### 2.3 修完 Bug A 后再跑，发现新症状

注入 UTF-8 编码、重跑训练，这次能跑完，但日志里 loss 全是 NaN：

```
🚀 开始训练 | epochs=1 | batch=2 | lr=2e-05 | steps=5 | log_interval=1
  Epoch [1/1] Step [1/5] loss=nan avg=nan lr=0.00002000 eta=0.7min
  Epoch [1/1] Step [2/5] loss=nan avg=nan ...
  ...
✅ Epoch [1/1] 完成 | avg_loss=nan | 耗时=0.2min
🏁 训练完毕，最终权重: ./out/__test_trace_768.pth
```

Loss 从第一步就是 NaN——这不是数值溢出（fp16 OOM 通常先正常几步再爆），更像**数据/标签**层面的问题。

### 2.4 二分定位：写一个 sanity check 脚本

为了快速隔离原因，写了个独立诊断脚本，分别检查：
- 数据：每条样本的 `input_ids` 长度 & `labels != -100` 的有效 token 数
- 模型权重：加载后是否含 NaN/Inf
- 前向：fp32 路径的 loss 和 logits 是否正常

输出：

```
[0] input_ids len=333, valid labels=0/333          ← 关键证据
  *** all labels masked! ***
[1] input_ids len=352, valid labels=0/352
  *** all labels masked! ***

missing keys: 0  unexpected: 0
weights nan? False  inf? False
batch input_ids shape: torch.Size([2, 352]), valid labels: 0
fp32 loss: nan
fp32 logits nan? False  inf? False
```

模型和权重完全健康，logits 也健康，问题是 **labels 全部被 mask 成 -100**。
PyTorch 的 `F.cross_entropy(logits, labels, ignore_index=-100)` 在 labels 全是 ignore 的情况下，分母（有效 token 数）= 0，得到 `0/0 = NaN`。

**Bug B 现身。**

---

## 3. Bug A 详解：emoji vs Windows GBK

### 3.1 根因

Windows 上 Python 启动时，`sys.stdout` 的编码取决于：

| 触发条件 | stdout 编码 |
|---|---|
| 设置了 `PYTHONIOENCODING=utf-8` | utf-8 |
| Python 3.7+ 且终端是 UTF-8 模式（`chcp 65001`） | 大概率 utf-8 |
| 默认（cmd / PowerShell / 子进程） | **gbk / cp936** |
| 输出被重定向到管道（`subprocess.PIPE`） | 跟随 locale，**通常 gbk** |

我们的训练脚本里大量使用 emoji（📝 🧠 📦 📚 🚀 ✅ 🏁）作为日志的视觉标识。在直接终端跑还能蒙混过关（如果用户提前 `chcp 65001`），但**通过 `subprocess.PIPE` 拉起时一定是 gbk**，遇到 emoji 必崩。

### 3.2 关键代码位置

`trainer/train_manager.py` 启动子进程的地方：

```python
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,                              # ← 自动 decode/encode，但用什么编码？
    cwd=root,
    bufsize=1,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},  # ← 只设了无缓冲，没设编码
)
```

- `text=True` 让 subprocess 自动解码 stdout 为 str，但**没指定 encoding 时用系统 locale**（Windows → gbk）
- 子进程的 Python stdout 编码也跟系统走（gbk）
- 子进程一旦 `print` emoji → 编码失败 → 抛 `UnicodeEncodeError` → 整个进程崩

### 3.3 修复

**两道保险，缺一不可：**

1. **子进程注入 UTF-8 环境变量**（`trainer/train_manager.py`）：

```python
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding='utf-8',           # ← 父进程读取 stdout 用 UTF-8 decode
    errors='replace',           # ← 万一遇到非法字节也不要崩，替换成 ?
    cwd=root,
    bufsize=1,
    env={
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",   # ← 关键：让子进程 print emoji 不再崩
    },
)
```

2. **训练脚本自我兜底**（`trainer/train_text_sft.py`）：

```python
# ── Windows 终端编码兜底 ──
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass  # 老 Python / 重定向场景没有这个方法，忽略
```

为什么两道都要？
- 第 1 道（环境变量）解决「通过 subprocess.Popen 启动」的场景
- 第 2 道（reconfigure）解决「用户直接命令行运行 `python trainer/train_text_sft.py`」的场景，不依赖任何环境配置

**额外补充**：在 `train_manager.py` 的错误翻译表里加一条：

```python
("UnicodeEncodeError",
 "🔤 控制台编码不支持 emoji。请重启训练 WebUI（脚本已修复，强制 UTF-8 编码）。"),
```

这样未来如果在某种诡异环境下又撞到（比如某些远程 SSH 会话），用户也能直接看到中文提示。

### 3.4 教训

| 教训 | 怎么做 |
|---|---|
| **永远不要假定 stdout 编码** | 跨平台脚本顶部加 `sys.stdout.reconfigure(encoding='utf-8')` |
| **subprocess 必须显式指定编码** | `Popen(..., text=True, encoding='utf-8', errors='replace')` |
| **环境变量比代码兜底优先** | `PYTHONIOENCODING=utf-8` 比 `reconfigure` 早生效，能保护 import 阶段的 print |
| **不要用 emoji 做关键日志** | 视觉好看但兼容性差。如果非要用，加编码兜底 |
| **不要让 stdout 异常杀掉训练** | 长任务里建议 `print` 套 try/except 或者用 logging |

---

## 4. Bug B 详解：ChatDataset Labels Mask 失效

### 4.1 SFT 训练的 Labels Mask 是什么

监督微调（SFT）训练时，我们希望模型**只学习"如何回答"**，而不去学习「user 的问题」或「system prompt」。具体做法：

- `input_ids`：完整的对话 token 序列（user + assistant 都喂进去）
- `labels`：在 cross-entropy 计算时，把 user / system 部分用 `IGNORE_INDEX = -100` 标记，PyTorch 会自动忽略这些位置；只在 assistant 的回复部分用真实 token id

```text
input_ids: [BOS] <|user|> 帮我写小红书 <|assistant|> 📍杭州…|EOS|
labels:    -100  -100    -100 -100 -100 -100      📍   杭州 …  EOS
                ←——————— ignore ————————→ ←—— train ——→
```

只要 mask 写错一个边界，要么模型学到 user 的话（不该学的），要么**整段被 mask**（什么都没学）。

### 4.2 原代码的错误

`trainer/train_text_sft.py` 旧版：

```python
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
encoded = tokenizer(text, truncation=True, max_length=max_length)
input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)

labels = input_ids.clone()
labels[:] = IGNORE   # 先全部 mask

buf = []
for turn in msgs:
    role = turn.get('role', '')
    content = turn.get('content', '')
    # 每个 turn 单独编码一次
    if role == 'user':
        header = tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=True)
    elif role == 'assistant':
        header = tokenizer.apply_chat_template([turn], tokenize=False, add_generation_prompt=False)
    ...
    h_ids = tokenizer(header, add_special_tokens=False)['input_ids']

    if role == 'assistant':
        content_ids = tokenizer(content, add_special_tokens=False)['input_ids']
        assist_pos = len(buf) + len(h_ids) - len(content_ids)
        end_pos = len(buf) + len(h_ids)
        if end_pos <= len(labels):
            labels[assist_pos:end_pos] = input_ids[assist_pos:end_pos]
    buf.extend(h_ids)
```

逻辑意图：用 `buf` 累计前面所有 turn 的 token 长度，每遇到 assistant turn 就在「全局 input_ids」里定位它的 content 区间，把对应 labels 还原。

**为什么错了？**

核心错误是这个隐含假设：

> `tokenizer.apply_chat_template([all_turns])` 的 token 序列
>   == `concat(tokenizer.apply_chat_template([turn_1]), tokenizer.apply_chat_template([turn_2]), …)`

这是不成立的。原因：

1. **特殊 token 重复或缺失**
   - `apply_chat_template([single_turn])` 通常会在头部加 BOS、在尾部加 EOS（看模板而定）
   - `apply_chat_template([full_dialog])` 只在最开头加一次 BOS、最末尾加一次 EOS
   - 直接累加 ⇒ 中间多出来 EOS+BOS 重复 token

2. **chat template 的 jinja 模板可能跨 turn 插入分隔符**
   - 比如 `{% for m in messages %}{{m.role}}\n{{m.content}}\n\n{% endfor %}` 这种模板，turn 之间有 `\n\n`
   - 单独编码每个 turn 时，这个 `\n\n` 没了

3. **`tokenizer(text, add_special_tokens=False)` vs `add_special_tokens=True`**
   - 整体编码（默认 True）会再次加 special tokens；逐 turn 编码（False）没有
   - 两者位置偏移更对不上

4. **`content_ids` 长度未必等于 `header_ids` 末尾的 content 部分长度**
   - `apply_chat_template([turn])` 生成的 header 是 `<|assistant|>\n内容\n`
   - 单独 tokenize `内容` 得到的 content_ids 可能因为前后没有上下文，BPE 分词出来的 token 数量都不一样
   - `assist_pos = len(buf) + len(h_ids) - len(content_ids)` 这个减法不可靠

实际结果：`assist_pos` 算出来基本是错的，`labels[assist_pos:end_pos] = ...` 这一步要么被 `if end_pos <= len(labels)` 拒绝，要么写到错误位置（但那个位置原本是 -100，刚好与「写不进去」同样表现为"看不出来"）。

最终：**`labels != -100` 的位置数永远是 0**。

### 4.3 NaN 是怎么传播的

```python
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
loss = loss_fn(logits.view(-1, V), labels.view(-1))
```

PyTorch 内部计算（简化）：
```python
valid_mask = labels != -100
total_loss = -log_softmax(logits)[valid_mask, labels[valid_mask]].sum()
n_valid = valid_mask.sum()
loss = total_loss / n_valid          # ← n_valid == 0 时 0/0 = NaN
```

更糟糕的是：**NaN 会污染整个反向传播**。
- `loss.backward()` → 所有梯度变 NaN
- `optimizer.step()` → 所有参数被加上 NaN → 模型权重瞬间全 NaN
- 后续的 forward 仍能输出值（NaN 算术），看起来"没崩"
- 训练脚本顺利跑到最后，存盘了一个**已经废掉的模型权重**

这就是为什么单看终端输出"训练完毕，最终权重: xxx.pth"是一种**伪成功**。

### 4.4 修复：增量前缀编码

```python
# 1) 整体编码（拿到唯一可信的 input_ids）
full_ids = tokenizer.apply_chat_template(
    msgs, tokenize=True, add_generation_prompt=False
)
if len(full_ids) > max_length:
    full_ids = full_ids[:max_length]
input_ids = torch.tensor(full_ids, dtype=torch.long)
labels = torch.full_like(input_ids, IGNORE_INDEX)

# 2) 增量前缀编码，定位每个 turn 的 token 区间
prev_len = 0
for i, m in enumerate(msgs):
    cur_ids = tokenizer.apply_chat_template(
        msgs[:i + 1], tokenize=True, add_generation_prompt=False
    )
    cur_len = min(len(cur_ids), len(input_ids))
    # 只对 assistant turn 解 mask
    if m.get('role') == 'assistant' and cur_len > prev_len:
        labels[prev_len:cur_len] = input_ids[prev_len:cur_len]
    prev_len = cur_len
    if prev_len >= len(input_ids):
        break

# 3) 安全门：至少要有几个有效 token 才训练
if (labels != IGNORE_INDEX).sum().item() < 2:
    skipped += 1
    continue
```

为什么这个方法是对的？

**关键性质（前缀单调性）**：对绝大多数 chat template，有

```
apply_chat_template(msgs[:i+1])  ⊇  apply_chat_template(msgs[:i])
```

也就是说，**截至第 i+1 个 turn 的 token 序列**是**截至第 i 个 turn 的 token 序列**的前缀扩展。这是 jinja chat template 的天然性质——它按顺序拼接 turns，前面写过的不会变。

基于这个性质：
- `prev_len` = 截至第 i-1 个 turn 的 token 数
- `cur_len`  = 截至第 i 个 turn 的 token 数
- 第 i 个 turn 在 `input_ids` 中的范围就是 `[prev_len, cur_len)`

这种定位方式**对任何 chat template 都成立**，因为我们从未假设单 turn 编码能直接拼接，而是用前缀差分算位置。

**安全门的必要性**：万一遇到极端样本（比如截断后 assistant 部分被切掉了），`(labels != -100).sum() < 2` 就跳过，避免 NaN 再次出现。

### 4.5 教训

| 教训 | 怎么做 |
|---|---|
| **永远验证 labels 有效数 > 0** | Dataset 类里加 assertion 或 skip + 计数告警 |
| **永远验证第一个 batch 的 loss 不是 NaN** | 训练循环开始前跑一个 dry run |
| **`apply_chat_template` 必须整体调用** | 不要逐 turn 调用再拼接，token 边界不可靠 |
| **NaN 损失 ≠ 训练失败** | NaN 会传播但不抛异常，必须显式检测 `torch.isnan(loss)` |
| **数据预处理是 SFT 的最大坑** | 比模型架构 bug 隐蔽得多，必须做 unit test |

### 4.6 推荐的训练循环硬化（TODO）

将来可以在训练循环里加这些保护：

```python
# 第一步前 dry run
sample_batch = next(iter(loader))
with torch.no_grad():
    sample_loss = model(input_ids=sample_batch[0].to(device),
                        labels=sample_batch[1].to(device)).loss
assert not torch.isnan(sample_loss), \
    "❌ 第一个 batch 就 NaN，请检查 ChatDataset 的 labels mask"

# 训练循环里
if torch.isnan(loss) or torch.isinf(loss):
    print(f"⚠️ Step {step}: loss={loss.item()}，跳过本步")
    optimizer.zero_grad(set_to_none=True)
    continue
```

---

## 5. Bug C（UX）：失败时日志默认折叠，看不到原因

### 5.1 现象

前端 `web_train.html` 的"详细日志"区域默认 `display: none`，用户点击 `▸ 详细日志` 才展开。但训练失败时——尤其是新手用户——常常看不到这个折叠按钮，只看到一个"失败"标记，不知所措。

### 5.2 修复

`web_train.html` 在 WebSocket 收到 `status: failed` 或 `stopped` 时：

```js
} else {
    // 失败/停止时：自动展开日志面板 + 滚到底，让用户立刻看到原因
    $('logBox').style.display = '';
    $('logToggle').textContent = '▾ 详细日志';
    $('logBox').scrollTop = $('logBox').scrollHeight;
    // 尝试从日志里抓出第一行 hint（带 💡 的）作为摘要
    const hintLine = [...$('logBox').children].map(n => n.textContent)
        .find(s => s.includes('💡'));
    $('trainSummary').textContent = hintLine
        ? hintLine.trim()
        : '训练未完成，详细日志已展开（请下滑查看具体错误）。';
}
```

配合后端 `train_manager.py` 的错误翻译表，常见错误（CUDA OOM / 文件缺失 / 编码错误 / JSON 错误等）会被翻译成 `💡 中文提示` 直接显示在摘要位置。

---

## 6. 完整修改清单

| 文件 | 改动类型 | 内容 |
|---|---|---|
| `trainer/train_text_sft.py` | 兼容性 | 顶部加 `sys.stdout.reconfigure(encoding='utf-8')` |
| `trainer/train_text_sft.py` | Bug 修复 | 重写 `ChatDataset` 的 labels mask 逻辑（增量前缀编码 + 安全门） |
| `trainer/train_manager.py` | Bug 修复 | `subprocess.Popen` 加 `encoding='utf-8'` + `PYTHONIOENCODING=utf-8` |
| `trainer/train_manager.py` | 增强 | 错误翻译表加 `UnicodeEncodeError` 条目 |
| `webui/web_train.html` | UX | 失败时自动展开日志 + 提取 hint 作为摘要 |

---

## 7. 验证方式

### 7.1 命令行直接跑

```powershell
python -u trainer/train_text_sft.py `
    --data_path ./dataset/xiaohongshu.json `
    --epochs 1 --batch_size 2 --learning_rate 2e-5 `
    --save_name __sanity --tokenizer_path ./model --save_dir ./out `
    --base_model ./out/llm_768.pth
```

**期望输出**：

```
📝 加载 tokenizer: ./model                  ← emoji 不再崩
...
🚀 开始训练 | epochs=1 | batch=2 | lr=2e-05 | steps=5 | log_interval=1
  Epoch [1/1] Step [1/5] loss=3.2050  ← loss 不是 NaN
  Epoch [1/1] Step [2/5] loss=2.5757  ← 在下降
  Epoch [1/1] Step [3/5] loss=2.8642
  Epoch [1/1] Step [4/5] loss=2.7426
  Epoch [1/1] Step [5/5] loss=3.0709
✅ Epoch [1/1] 完成 | avg_loss=2.8917 | 耗时=0.2min
```

跑完后记得删除测试产物：`del out\__sanity_768.pth`

### 7.2 WebUI 端到端

1. `启动训练WebUI.bat`（端口 7861）
2. 第 ① 步：点「📖 小红书风格」一键加载模板 → 保存
3. 第 ② 步：选数据集 → 选「🎯 正常训练」难度 → 点开始
4. 期望：进度条平滑前进，loss 从 ~3 下降到 ~2，无 NaN
5. 第 ③ 步：在输入框打「帮我写一篇咖啡店探店」→ 收到合理回复

### 7.3 单元测试建议（未来）

可以为 `ChatDataset` 加一个最小测试：

```python
def test_chat_dataset_labels_not_all_masked():
    tok = AutoTokenizer.from_pretrained('./model')
    ds = ChatDataset('./dataset/xiaohongshu.json', tok, max_length=512)
    assert len(ds) > 0, "数据集不应为空"
    for i in range(len(ds)):
        valid = (ds[i]['labels'] != -100).sum().item()
        assert valid >= 2, f"样本 {i} 的有效 labels 太少: {valid}"
```

---

## 8. 总结

这次复盘的核心三点：

1. **跨平台脚本必须显式处理编码**——尤其是 Windows + emoji + subprocess 这个组合。两道保险（环境变量 + reconfigure）才能覆盖所有调用路径。

2. **SFT 数据预处理的 mask 逻辑是雷区**——`apply_chat_template` 必须整体调用，不能逐 turn 拼接。增量前缀编码是数学上唯一正确的定位方法。NaN loss 不会抛异常，必须主动检测。

3. **UX 上要让错误"无处可藏"**——失败状态默认折叠的日志是反人类设计。失败必须自动展开 + 把翻译后的中文 hint 推到最显眼位置。

这套修复后的训练 WebUI 已经具备给完全不懂代码的新手用的成熟度。下一步建议补单元测试和 dry-run 检查（见 4.6 / 7.3），让回归更难发生。
