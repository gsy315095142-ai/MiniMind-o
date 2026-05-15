# MiniMind-O 训练端 WebUI 设计与开发文档

> 版本: v1.0  
> 日期: 2026-05-11  
> 目的：将命令行训练流程改造为可视化网页操作，降低使用门槛。

---

## 1. 背景与目标

### 1.1 现状

MiniMind-O 项目已具备完整的命令行训练能力：

- 纯文本 SFT：`trainer/train_text_sft.py`（本项目新增）
- Omni 全模态 SFT：`trainer/train_sft_omni.py`（项目自带）
- 推理服务：`webui/web_demo.py`（Flask + WebSocket，已提供 API）

### 1.2 痛点

当前训练流程依赖终端操作，对非技术用户不友好：

- 需要手动编辑 JSON 数据文件
- 需要记忆命令行参数（`--epochs`、`--batch_size` 等）
- 无法直观看到训练进度和日志
- 版本管理靠手动命名

### 1.3 目标

在现有 `web_demo.py` 基础上新增「训练管理」页面，实现：

- ✅ 网页端上传 / 粘贴训练数据
- ✅ 可视化设置训练参数（滑块、下拉框）
- ✅ 一键启动训练，实时查看日志
- ✅ 管理已有模型版本（查看、切换、删除）
- ✅ 训练完成后直接在网页测试效果

---

## 2. 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 前端 | HTML + CSS + 原生 JS | 不引入前端框架，复用 `web_demo.html` 风格 |
| 实时通信 | WebSocket（Flask-Sock） | 训练日志实时推送到浏览器，项目已集成 |
| 后端 | Python Flask | 复用 `web_demo.py`，新增路由 |
| 进程管理 | `subprocess.Popen` | 拉起训练子进程，不阻塞 Flask 主线程 |
| 数据存储 | JSON 文件 + 本地文件系统 | 训练数据存 `dataset/`，权重存 `out/` |

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────┐
│                 浏览器（网页前端）                  │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ 数据编辑  │  │ 参数设置  │  │ 实时日志面板   │  │
│  │ (粘贴JSON)│  │ (表单控件) │  │ (滚动输出)    │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                   │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ 模型列表  │  │ 一键训练  │  │ 推理测试      │  │
│  │ (下拉框)  │  │ (按钮)    │  │ (对话框)      │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
└────────────┬──────────────┬──────────────────────┘
             │ HTTP/REST    │ WebSocket
┌────────────▼──────────────▼──────────────────────┐
│              Flask 后端 (web_demo.py)             │
│                                                   │
│  /train_page     → 返回训练页 HTML                │
│  /api/upload_data → 接收上传/粘贴的训练数据        │
│  /api/start_train → 启动训练子进程                 │
│  /api/models      → 列出 out/*.pth               │
│  /api/delete_model→ 删除指定权重                   │
│  /ws/train_log    → WebSocket 推送训练日志         │
│                                                   │
│  subprocess.Popen                                 │
│       │                                           │
│       ▼                                           │
│  ┌─────────────────────┐                          │
│  │ train_text_sft.py   │  ── stdout ──→ WS 推送   │
│  └─────────────────────┘                          │
└───────────────────────────────────────────────────┘
```

---

## 4. 功能模块详设

### 4.1 训练数据管理

**功能**：用户无需手动找目录建 JSON 文件，直接在网页操作。

| 操作 | 实现 |
|---|---|
| 粘贴 JSON 文本 | `<textarea>`，前端校验 JSON 格式 |
| 上传 JSON 文件 | `<input type="file">`，读取后填入 textarea |
| 保存到本地 | 后端 `POST /api/upload_data` → `write_file` 到 `dataset/` |
| 加载已有数据 | 下拉框列出 `dataset/*.json`，选中加载到 textarea |

**目标体验**：打开页面 → 粘贴小红书对话 → 点保存 → 数据就绪。

### 4.2 训练参数设置

**功能**：把命令行参数变成可视化表单。

| 参数 | 控件 | 默认值 | 说明 |
|---|---|---|---|
| 训练数据 | 下拉框 | — | 选择 `dataset/` 下的 JSON 文件 |
| Epochs | 数字输入 / 滑块 | 3 | 训练轮数（1~20） |
| Batch Size | 下拉框 | 2 | 1 / 2 / 4 / 8 / 16 |
| 学习率 | 文本输入 | 2e-5 | 科学计数法 |
| 模型名称 | 文本输入 | `xhs_v1` | 保存为 `{name}_768.pth` |
| 基础模型 | 下拉框 | `llm` | 从哪个权重开始训练 |

**后台映射**：

```
表单 → CLI 参数
epochs     → --epochs 3
batch_size → --batch_size 2
lr         → --learning_rate 2e-5
save_name  → --save_name xhs_v1
base_model → --base_model ./out/llm_768.pth
```

### 4.3 一键训练 + 实时日志

**流程**：

```
用户点「开始训练」
    │
    ▼
前端 POST /api/start_train（携带参数）
    │
    ▼
后端校验参数 → 创建 subprocess.Popen → 返回 {"task_id": "xxx"}
    │
    ▼
前端连接 WebSocket ws://host/ws/train_log?task_id=xxx
    │
    ▼
后端逐行读取 train_text_sft.py 的 stdout
    → 通过 WebSocket 推送给前端
    │
    ▼
前端 <pre> 区域逐行追加显示
    │
    ▼
训练结束 → 后端发送 {"status": "done", "model": "xhs_v1_768.pth"}
    → 前端显示「训练完成」，自动刷新模型列表
```

**关键设计点**：

- 用 `threading.Thread` 包裹训练进程，不阻塞 Flask 主线程
- WebSocket 按 `task_id` 区分，多个用户互不干扰（单用户无需多路复用，但留扩展余地）
- 训练日志同时写入 `out/{save_name}.log`，方便离线查看

### 4.4 模型版本管理

**后端**：

```
GET /api/models  → 扫描 out/*.pth，返回列表
  [
    {"name": "llm_768.pth",        "size": "131 MB", "type": "基座"},
    {"name": "xhs_v1_768.pth",     "size": "131 MB", "type": "微调"},
    {"name": "xhs_v2_768.pth",     "size": "131 MB", "type": "微调"},
    {"name": "sft_omni_768.pth",   "size": "225 MB", "type": "Omni"},
  ]

DELETE /api/delete_model?name=xhs_v1_768.pth → 删除文件（需确认）
```

**前端**：

- 表格展示：名称 | 大小 | 类型 | 训练时间（从文件时间戳读取）| 操作（删除）
- 删除按钮带二次确认弹窗
- 不允许删除 `llm_768.pth`（基座保护）

### 4.5 训练后测试

训练完成后，无缝切换到推理测试：

- 训练完毕自动将新模型加入「当前推理模型」下拉框
- 切换到「对话测试」标签页
- 选新模型 → 输入 prompt → 看回复有没有小红书的味儿
- 不满意 → 回到训练页调整数据 → 重新训练

---

## 5. 文件结构

```
MiniMind-o/
├── webui/
│   ├── web_demo.py          ← 现有推理服务（新增 train 路由）
│   ├── web_demo.html        ← 现有推理页面（不变）
│   └── web_train.html       ← 新增：训练管理页面
│
├── trainer/
│   ├── train_text_sft.py    ← 纯文本训练脚本（已完成）
│   ├── train_sft_omni.py    ← Omni 训练脚本（现有）
│   └── train_manager.py     ← 新增：训练任务管理器（线程+进程封装）
│
├── dataset/
│   ├── xiaohongshu.json     ← 示例训练数据（已完成）
│   └── ...                  ← 用户通过网页上传的数据
│
└── out/
    ├── llm_768.pth          ← 基座（只读保护）
    ├── xhs_v1_768.pth       ← 训练产物
    └── xhs_v1.log           ← 训练日志
```

---

## 6. API 接口定义

### 6.1 页面

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/train` | 返回训练管理页面 HTML |

### 6.2 数据管理

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/datasets` | 列出 `dataset/*.json` |
| POST | `/api/upload_data` | 保存训练数据到 `dataset/` |
| GET | `/api/load_data?name=xxx.json` | 返回指定 JSON 内容 |

### 6.3 训练控制

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/start_train` | 启动训练，返回 task_id |
| GET | `/api/train_status?task_id=xxx` | 查询训练状态 |

### 6.4 模型管理

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/models` | 列出 `out/*.pth` |
| DELETE | `/api/delete_model?name=xxx.pth` | 删除指定权重 |

### 6.5 实时日志

| 方法 | 路径 | 说明 |
|---|---|---|
| WS | `/ws/train_log?task_id=xxx` | 订阅训练日志流 |

---

## 7. 开发计划

### Phase 1：MVP（最小可用版）

| 任务 | 预计工作量 | 优先级 |
|---|---|---|
| `trainer/train_manager.py` — 训练进程管理器 | 30 分钟 | P0 |
| `webui/web_demo.py` — 新增 `/train` + API 路由 | 40 分钟 | P0 |
| `webui/web_train.html` — 训练页前端 | 1 小时 | P0 |
| 联调测试 | 20 分钟 | P0 |

**MVP 功能**：粘贴数据 → 调参数 → 点训练 → 看日志滚动 → 完成。

### Phase 2：V1 增强

| 任务 | 说明 |
|---|---|
| 模型列表 + 删除 | 下拉框切换、一键删除 |
| 训练历史 | 记录每次训练的参数和结果 |
| 错误提示优化 | 参数校验、JSON 格式检查 |

### Phase 3：V2 进阶

| 任务 | 说明 |
|---|---|
| Loss 曲线图 | 用 Chart.js 绘制实时 loss 曲线 |
| 断点续训 | 训练中断后可以从 checkpoint 恢复 |
| 多数据集混合训练 | 勾选多个 JSON 合并训练 |

---

## 8. 注意事项与风险

| 事项 | 说明 |
|---|---|
| **GPU 阻塞** | 训练占满 GPU 时推理会卡顿，建议训练期间不推理。可在页面加提示。 |
| **并发训练** | MVP 阶段限制同时只能有一个训练任务，后续可通过 GPU 分片支持多任务。 |
| **基座保护** | `llm_768.pth` 只读，不允许覆盖或删除。 |
| **内存安全** | 训练数据上传前校验 JSON 格式，防止写入非法内容。 |
| **进程清理** | 页面关闭时需提示是否终止训练（WebSocket 断开时触发清理）。 |

---

## 9. 与现有系统的关系

```
web_demo.py（现有）
    │
    ├── 推理路由（保留不变）
    │   ├── /chat、/voices、/models、/switch_model 等
    │   └── /ws/realtime
    │
    └── 训练路由（新增）
        ├── /train          → 训练页面
        ├── /api/upload_data
        ├── /api/start_train
        ├── /api/models
        └── /ws/train_log
```

**不改动**任何现有推理 API，训练功能作为独立模块插入，不影响原有服务。

---

## 10. 总结

**一句话**：在现有 Flask + WebSocket 基础上，加一个训练管理页面，后端用 `subprocess` 拉起 `train_text_sft.py`，前端通过 WebSocket 实时看日志。不引入新依赖，不改动现有推理 API，MVP 约 2 小时可完成。
