"""
MiniMind-O 训练管理 WebUI 服务
独立于推理服务 (web_demo.py)，可同时运行在不同端口。

用法:
  python webui/train_server.py
  python webui/train_server.py --port 7861
"""

import argparse, os, sys, json, glob, time, threading, shutil

from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from flask_sock import Sock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.train_manager import TrainManager
from trainer.model_registry import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_TYPE,
    list_model_types,
    get_config,
    get_preset,
)

app = Flask(__name__, static_folder='.')
CORS(app)
sock = Sock(app)

# 全局管理器
manager = TrainManager()

# 路径常量
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, 'dataset')
OUT_DIR = os.path.join(ROOT, 'out')

# 「试试效果」模块用到的：训练完毕后可以直接在训练页内联测试模型
# 模型加载较慢（几百 MB），加载一次后缓存到内存里。
_TEST_CACHE = {}        # cache_key -> (model, tokenizer, device)
_TEST_LOCK = threading.Lock()

# ── 内置数据模板 ──
# 让小白能一键加载示范数据集，避免对着空 textarea 不知道怎么写。
_TEMPLATES = [
    {
        "id": "xiaohongshu",
        "name": "📖 小红书风格",
        "desc": "标题+正文+emoji+标签的小红书爆款文案生成（含 10 条精挑示例）",
        "file": "xiaohongshu.json",
    },
]


def json_resp(data, status=200):
    return Response(json.dumps(data, ensure_ascii=False), status=status, mimetype='application/json')


# ======================== 页面 ========================

@app.route('/')
def index():
    return send_from_directory('.', 'web_train.html')


# ======================== 数据集管理 ========================

@app.route('/api/datasets')
def list_datasets():
    """列出 dataset/ 下所有 .json 文件"""
    if not os.path.isdir(DATASET_DIR):
        return json_resp({"datasets": []})
    files = sorted(glob.glob(os.path.join(DATASET_DIR, '*.json')))
    datasets = []
    for f in files:
        name = os.path.basename(f)
        size = os.path.getsize(f)
        # 尝试读取样本数
        count = 0
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    count = len(data)
        except Exception:
            pass
        datasets.append({"name": name, "size": size, "count": count})
    return json_resp({"datasets": datasets})


@app.route('/api/load_data')
def load_data():
    """读取指定数据集内容"""
    name = request.args.get('name', '')
    path = os.path.join(DATASET_DIR, name)
    if not name or not os.path.isfile(path):
        return json_resp({"error": "文件不存在"}, 400)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return json_resp({"name": name, "content": content})


@app.route('/api/templates')
def list_templates():
    """列出内置数据模板（用于新手一键导入示例数据）"""
    out = []
    for t in _TEMPLATES:
        path = os.path.join(DATASET_DIR, t["file"])
        count = 0
        exists = os.path.isfile(path)
        if exists:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
            except Exception:
                pass
        out.append({**t, "count": count, "available": exists})
    return json_resp({"templates": out})


@app.route('/api/load_template')
def load_template():
    """读取指定模板内容（默认填充到「数据编辑器」）"""
    tid = (request.args.get('id') or '').strip()
    tpl = next((t for t in _TEMPLATES if t["id"] == tid), None)
    if not tpl:
        return json_resp({"error": "模板不存在"}, 400)
    path = os.path.join(DATASET_DIR, tpl["file"])
    if not os.path.isfile(path):
        return json_resp({"error": f"模板文件缺失：{tpl['file']}"}, 400)
    with open(path, 'r', encoding='utf-8') as f:
        return json_resp({"id": tid, "name": tpl["name"], "file": tpl["file"], "content": f.read()})


@app.route('/api/save_data', methods=['POST'])
def save_data():
    """保存训练数据到 dataset/"""
    d = request.json or {}
    name = (d.get('name') or '').strip()
    content = d.get('content', '')
    if not name:
        return json_resp({"error": "文件名不能为空"}, 400)
    if not name.endswith('.json'):
        name += '.json'
    # 安全校验：不允许路径穿越
    if '/' in name or '\\' in name or '..' in name:
        return json_resp({"error": "文件名不合法"}, 400)
    # 校验 JSON 格式
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        return json_resp({"error": f"JSON 格式错误: {e}"}, 400)
    os.makedirs(DATASET_DIR, exist_ok=True)
    path = os.path.join(DATASET_DIR, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return json_resp({"ok": True, "name": name})


# ======================== 模型类型注册表 ========================

@app.route('/api/model_types')
def api_model_types():
    """列出可用模型类型（MiniMind / DeepSeek / ...），给前端下拉框用"""
    return json_resp({
        "default": DEFAULT_MODEL_TYPE,
        "types": list_model_types(),
    })


@app.route('/api/model_type_config')
def api_model_type_config():
    """返回某个模型类型的完整配置（默认参数 + 难度档位预设）。
    前端切换模型时拉一次，自动填充参数面板。
    """
    type_id = (request.args.get('type') or DEFAULT_MODEL_TYPE).strip()
    try:
        cfg = get_config(type_id)
    except KeyError as e:
        return json_resp({"error": str(e)}, 400)

    # 给前端展示用的基座信息：相对项目根的短路径 + 是否存在
    # （不暴露绝对路径以免泄露用户机器目录结构）
    base_abs = cfg["base_model_path"]
    try:
        base_rel = os.path.relpath(base_abs, ROOT).replace("\\", "/")
    except ValueError:
        base_rel = os.path.basename(base_abs)

    return json_resp({
        "id": type_id,
        "display_name": cfg["display_name"],
        "family": cfg["family"],
        "save_format": cfg["save_format"],
        "save_ext": cfg["save_ext"],
        "vram_hint": cfg["vram_hint"],
        "presets": cfg["presets"],
        "extra_args": cfg.get("extra_args", {}),
        "base_available": _base_exists_for(cfg),
        "base_model_rel": base_rel,                  # v2.1：基座相对路径
        "base_model_size_mb": _safe_base_size(cfg),  # v2.1：基座体积
    })


@app.route('/api/resumable_loras')
def api_resumable_loras():
    """列出可作为「续训起点」的已有 LoRA 目录（v2.2）。

    入参：?type=deepseek-r1-1.5b
    返回：[{"name": "smoke_deepseek_v1_lora", "base_model": "...", "size_mb": 12.3,
            "time": "2026-05-18 11:00", "compatible": true}, ...]

    `compatible=True` 表示该 LoRA 的基座与当前 model_type 的基座一致，可安全续训；
    `compatible=False` 表示基座不匹配（如把 deepseek-1.5b 的 LoRA 当 7b 的来续），
    前端应该灰掉这些项。
    """
    type_id = (request.args.get('type') or DEFAULT_MODEL_TYPE).strip()
    try:
        cfg = get_config(type_id)
    except KeyError as e:
        return json_resp({"error": str(e)}, 400)

    # 只对 LoRA 类模型有意义
    if cfg.get('save_format') != 'lora':
        return json_resp({"items": []})

    items = []
    if not os.path.isdir(OUT_DIR):
        return json_resp({"items": items})

    for entry in sorted(os.listdir(OUT_DIR)):
        full = os.path.join(OUT_DIR, entry)
        # 必须是 _lora 目录 + 含 adapter_config.json
        if not (entry.endswith('_lora') and os.path.isdir(full)):
            continue
        if not os.path.isfile(os.path.join(full, 'adapter_config.json')):
            continue

        meta_path = os.path.join(full, 'minimind_o_meta.json')
        base_model_path = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                base_model_path = meta.get('base_model_path')
            except (OSError, ValueError):
                pass

        # 没有 meta 的旧产物：兜底视为不兼容，避免误用
        compatible = bool(base_model_path) and \
            _same_base_path(base_model_path, cfg['base_model_path'])

        # 目录体积
        size = 0
        for root, _, files in os.walk(full):
            for fn in files:
                fp = os.path.join(root, fn)
                if os.path.isfile(fp):
                    size += os.path.getsize(fp)

        items.append({
            "name": entry,
            "base_model": base_model_path or "(未知，缺失 meta)",
            "size_mb": round(size / 1024 / 1024, 1),
            "time": time.strftime('%Y-%m-%d %H:%M',
                                  time.localtime(os.path.getmtime(full))),
            "compatible": compatible,
        })
    return json_resp({"items": items})


def _same_base_path(a: str, b: str) -> bool:
    """兼容比较两个基座路径——
    旧产物的 meta 里可能存的是项目相对路径（如 "models/deepseek-r1-1.5b"），
    新产物存的是绝对路径，直接 normpath 比较会判错。
    策略：两个路径都解析为相对项目根 ROOT 的形式后比较 basename + 路径尾部。
    """
    if not a or not b:
        return False

    def _normalize(p):
        p = os.path.normpath(p)
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(ROOT, p))
        return p
    return _normalize(a) == _normalize(b)


def _safe_base_size(cfg) -> float:
    """计算基座体积（MB）：.pth 算文件；目录递归算 .safetensors / .bin / .pth 总和"""
    path = cfg["base_model_path"]
    try:
        if cfg["save_format"] == "pth":
            return round(os.path.getsize(path) / 1024 / 1024, 1) if os.path.isfile(path) else 0
        # HF 目录：累加权重文件大小
        total = 0
        if os.path.isdir(path):
            for fn in os.listdir(path):
                if fn.endswith(('.safetensors', '.bin', '.pth', '.pt')):
                    fp = os.path.join(path, fn)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
        return round(total / 1024 / 1024, 1)
    except OSError:
        return 0


def _base_exists_for(cfg) -> bool:
    """基座是否在本地——同 model_registry._base_exists 但避免循环引用"""
    path = cfg["base_model_path"]
    if cfg["save_format"] == "pth":
        return os.path.isfile(path)
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


# ======================== 模型管理 ========================

@app.route('/api/models')
def list_models():
    """列出 out/ 下所有可用模型——支持两种形态：
      * MiniMind 全参数 SFT：单文件 .pth
      * LoRA 适配器（DeepSeek 等）：以 _lora 结尾的目录

    返回每项加 `format` 字段（"pth" / "lora"），前端按类型展示和路由。
    """
    if not os.path.isdir(OUT_DIR):
        return json_resp({"models": []})

    models = []

    # 1) 扫单文件 .pth（MiniMind）
    for f in sorted(glob.glob(os.path.join(OUT_DIR, '*.pth'))):
        name = os.path.basename(f)
        size = os.path.getsize(f)
        mtime = os.path.getmtime(f)
        models.append({
            "name": name,
            "format": "pth",
            "model_type": "minimind",
            "size_mb": round(size / 1024 / 1024, 1),
            "time": time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)),
            "is_base": (name == 'llm_768.pth'),
        })

    # 2) 扫 _lora/ 目录（LoRA adapter）
    for d in sorted(glob.glob(os.path.join(OUT_DIR, '*_lora'))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        # 必须含 adapter_config.json 才算有效 LoRA 产物
        cfg_path = os.path.join(d, 'adapter_config.json')
        if not os.path.isfile(cfg_path):
            continue
        # 体积按 adapter_model.safetensors 算（用户最关心的就是这个增量大小）
        adapter_file = os.path.join(d, 'adapter_model.safetensors')
        size = os.path.getsize(adapter_file) if os.path.isfile(adapter_file) else 0
        mtime = os.path.getmtime(cfg_path)
        # 从 meta 反查 model_type；找不到就标 unknown
        model_type = _infer_model_type_from_lora_dir(d)
        # v2.2：从 meta 里再读出「续训自哪个 LoRA」（如果有）
        resume_from = None
        meta_path = os.path.join(d, 'minimind_o_meta.json')
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    rf = (json.load(f) or {}).get('resume_from_lora')
                # 只保留目录名，不要绝对路径泄露
                if rf:
                    resume_from = os.path.basename(rf.rstrip('/\\'))
            except (OSError, ValueError):
                pass
        models.append({
            "name": name,
            "format": "lora",
            "model_type": model_type,
            "size_mb": round(size / 1024 / 1024, 1),
            "time": time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)),
            "is_base": False,    # LoRA 产物不可能是基座
            "resume_from": resume_from,   # v2.2：续训血统（None 表示从零训）
        })

    # 按时间倒序（最近训练的排前面）
    models.sort(key=lambda m: m["time"], reverse=True)
    return json_resp({"models": models})


def _infer_model_type_from_lora_dir(lora_dir: str) -> str:
    """从 LoRA 产物目录里反查它是哪个基座训出来的。

    优先读 minimind_o_meta.json（我们的训练脚本自己写的），
    退化：读 adapter_config.json 的 base_model_name_or_path 反向匹配注册表。
    """
    meta_path = os.path.join(lora_dir, 'minimind_o_meta.json')
    base_path = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                base_path = (json.load(f) or {}).get('base_model_path')
        except Exception:
            pass
    if not base_path:
        cfg_path = os.path.join(lora_dir, 'adapter_config.json')
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                base_path = (json.load(f) or {}).get('base_model_name_or_path')
        except Exception:
            return 'unknown'

    base_path_norm = os.path.normpath(base_path) if base_path else ''
    base_basename = os.path.basename(base_path_norm)
    for type_id, cfg in MODEL_REGISTRY.items():
        reg_norm = os.path.normpath(cfg['base_model_path'])
        if reg_norm == base_path_norm or os.path.basename(reg_norm) == base_basename:
            return type_id
    return 'unknown'


@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    """删除指定模型权重——支持 .pth 文件和 _lora/ 目录两种。
    不允许删除基座（`llm_768.pth`）。
    """
    d = request.json or {}
    name = (d.get('name') or '').strip()
    if not name:
        return json_resp({"error": "请指定模型名"}, 400)

    # 防路径穿越（同 save_data 校验）
    if '/' in name or '\\' in name or '..' in name:
        return json_resp({"error": "模型名不合法"}, 400)
    if name == 'llm_768.pth':
        return json_resp({"error": "基座模型不允许删除"}, 400)

    path = os.path.join(OUT_DIR, name)
    # 二次防御：解析后必须仍在 OUT_DIR 之下
    if not os.path.abspath(path).startswith(os.path.abspath(OUT_DIR) + os.sep):
        return json_resp({"error": "模型名不合法"}, 400)

    if os.path.isfile(path):
        os.remove(path)
        return json_resp({"ok": True, "deleted": name, "format": "pth"})
    if os.path.isdir(path):
        # 仅允许删 _lora 结尾的目录，防止误删用户其他目录
        if not name.endswith('_lora'):
            return json_resp({"error": "目录形态的模型只支持 _lora 后缀"}, 400)
        shutil.rmtree(path)
        return json_resp({"ok": True, "deleted": name, "format": "lora"})
    return json_resp({"error": "文件不存在"}, 400)


# ======================== 训练控制 ========================

@app.route('/api/start_train', methods=['POST'])
def start_train():
    """启动训练任务。

    必填字段：
      dataset       数据集文件名（dataset/ 下）
      save_name     模型保存名（不带后缀）
    可选字段：
      model_type    模型类型 id（默认 "minimind"）
      epochs / batch_size / learning_rate / device
    MiniMind 专属可选：
      base_model    基座短名（如 "llm" → llm_768.pth），仅当 model_type=minimind 时生效
    LoRA 专属可选：
      lora_r / lora_alpha / lora_dropout / max_seq_len /
      accumulation_steps / no_4bit
    """
    d = request.json or {}

    dataset = (d.get('dataset') or '').strip()
    save_name = (d.get('save_name') or '').strip()
    if not dataset:
        return json_resp({"error": "请选择训练数据集"}, 400)
    if not save_name:
        return json_resp({"error": "请填写模型名称"}, 400)

    if '/' in save_name or '\\' in save_name or '..' in save_name:
        return json_resp({"error": "模型名称不合法"}, 400)

    data_path = os.path.join(DATASET_DIR, dataset)
    if not os.path.isfile(data_path):
        return json_resp({"error": f"数据文件不存在: {dataset}"}, 400)

    # ── 模型类型校验 ──
    model_type = (d.get('model_type') or DEFAULT_MODEL_TYPE).strip()
    try:
        cfg = get_config(model_type)
    except KeyError as e:
        return json_resp({"error": str(e)}, 400)
    if not _base_exists_for(cfg):
        return json_resp(
            {"error": f"基座模型未下载或路径不存在：{cfg['base_model_path']}"}, 400)

    # ── 取该模型类型的默认参数，再让用户传入的字段覆盖 ──
    preset = get_preset(model_type, 'normal')

    params = {
        "model_type": model_type,
        "data_path": data_path,
        "epochs": int(d.get('epochs', preset['epochs'])),
        "batch_size": int(d.get('batch_size', preset['batch_size'])),
        "learning_rate": float(d.get('learning_rate', preset['learning_rate'])),
        "save_name": save_name,
        "save_dir": OUT_DIR,
    }

    # ── 按 family 分支处理「专属字段」 ──
    if cfg["family"] == "minimind":
        # MiniMind 的旧逻辑：用户传 base_model 短名 → 拼成 .pth 路径
        # ⚠️ 这条路径**仅对 MiniMind 有效**，DeepSeek 一定不能走这里
        base_model = d.get('base_model', 'llm')
        base_path = os.path.join(OUT_DIR, f"{base_model}_768.pth")
        if os.path.isfile(base_path):
            params["base_model"] = base_path
        else:
            alt = os.path.join(OUT_DIR, base_model)
            if os.path.isfile(alt):
                params["base_model"] = alt
            # 找不到就让 manager 用注册表里的默认 base_model_path

    elif cfg["family"] == "hf_causal_lm":
        # LoRA 类训练：把 LoRA 相关字段透传给 manager，由它转成 train_lora_sft.py 的 CLI
        for k in ("lora_r", "lora_alpha", "lora_dropout", "max_seq_len",
                  "accumulation_steps"):
            if k in d and d[k] not in (None, ''):
                params[k] = d[k]
        if d.get('no_4bit'):
            params["no_4bit"] = True

        # v2.2 续训：用户选了已有 LoRA 作为起点
        # 前端传过来的是相对名（如 "my_v1_lora"），后端负责转绝对路径 + 安全校验
        resume_name = (d.get('resume_from') or '').strip()
        if resume_name:
            if '/' in resume_name or '\\' in resume_name or '..' in resume_name:
                return json_resp({"error": "续训源名称不合法"}, 400)
            if not resume_name.endswith('_lora'):
                return json_resp({"error": "续训源必须是 _lora 目录"}, 400)
            if resume_name == save_name + cfg.get('save_ext', '_lora') or \
               resume_name == save_name:
                return json_resp(
                    {"error": "新模型名不能与续训源同名，否则会覆盖原产物，请换一个名字"},
                    400)
            resume_path = os.path.join(OUT_DIR, resume_name)
            # 二次防御：解析后必须仍在 OUT_DIR 之下
            if not os.path.abspath(resume_path).startswith(
                    os.path.abspath(OUT_DIR) + os.sep):
                return json_resp({"error": "续训源路径不合法"}, 400)
            if not os.path.isdir(resume_path):
                return json_resp(
                    {"error": f"续训源不存在或不是目录: {resume_name}"}, 400)
            if not os.path.isfile(os.path.join(resume_path, 'adapter_config.json')):
                return json_resp(
                    {"error": f"续训源缺少 adapter_config.json，不是有效的 LoRA 产物"},
                    400)
            # 跨基座续训风险：检查 meta 里的基座是否与当前 model_type 的基座一致
            # 注意：旧产物的 meta 可能存的是相对路径，需要兼容比较
            meta_path = os.path.join(resume_path, 'minimind_o_meta.json')
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    meta_base = meta.get('base_model_path')
                    if meta_base and not _same_base_path(meta_base, cfg['base_model_path']):
                        return json_resp({
                            "error": f"续训源是基于「{meta_base}」训练的，"
                                     f"与当前选择的基座「{cfg['base_model_path']}」"
                                     f"不一致，无法续训"
                        }, 400)
                except (OSError, ValueError):
                    pass  # meta 损坏不阻塞，仅做尽力而为的校验
            params["resume_from_lora"] = resume_path

    # 设备
    device = d.get('device', 'auto')
    if device != 'auto':
        params["device"] = device

    try:
        task_id = manager.start_train(params)
        return json_resp({"ok": True, "task_id": task_id, "model_type": model_type})
    except RuntimeError as e:
        return json_resp({"error": str(e)}, 400)
    except (KeyError, ValueError) as e:
        return json_resp({"error": str(e)}, 400)


@app.route('/api/stop_train', methods=['POST'])
def stop_train():
    """停止当前训练"""
    d = request.json or {}
    task_id = d.get('task_id', '')
    ok = manager.stop_train(task_id)
    if ok:
        return json_resp({"ok": True})
    return json_resp({"error": "没有正在运行的训练任务"}, 400)


@app.route('/api/train_status')
def train_status():
    """查询训练状态"""
    task_id = request.args.get('task_id', '')
    task = manager.get_task(task_id)
    if not task:
        return json_resp({"error": "任务不存在"}, 404)
    return json_resp(task.to_dict())


@app.route('/api/tasks')
def list_tasks():
    """列出所有任务"""
    return json_resp({"tasks": manager.list_tasks()})


# ======================== 「试试效果」内联测试 ========================
# 训练完成后，让用户在训练页里就能直接和新模型聊一句，验证效果。
# 两种形态：
#   * MiniMind .pth：用 MiniMindForCausalLM 加载（原有逻辑）
#   * LoRA _lora/：用 transformers + peft 加载基础模型 + 套上 adapter
#
# v2 缓存策略（关键）：
#   * MiniMind 模型体积小（~700MB）可以多缓存几个；
#   * LoRA 类模型即使 4-bit 也要占 2GB+ 显存，**同时只缓存 1 个**，
#     加载新 LoRA 前必须把旧的释放掉，否则 8GB 显存会被挤爆。

# 缓存 key 结构：
#   "minimind:xxx.pth:mtime"     → (model, tokenizer, device)
#   "lora:xxx_lora:mtime"        → (model, tokenizer, device)
# `_lora` 类只保留最近 1 个；MiniMind 类不主动驱逐


def _detect_model_form(model_name: str):
    """返回 ("pth" or "lora", abs_path)。文件/目录都不存在则 raise FileNotFoundError"""
    path = os.path.join(OUT_DIR, model_name)
    if os.path.isfile(path):
        return "pth", path
    if os.path.isdir(path):
        return "lora", path
    raise FileNotFoundError(f"模型不存在: {model_name}")


def _evict_lora_cache(except_key: str = None):
    """LoRA 单例缓存策略：释放除 except_key 外的所有 LoRA 缓存项"""
    import torch
    for k in list(_TEST_CACHE.keys()):
        if k.startswith("lora:") and k != except_key:
            try:
                model, _, _ = _TEST_CACHE.pop(k)
                del model
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_test_model_minimind(model_name: str, path: str):
    """原有逻辑：MiniMind .pth → MiniMindForCausalLM"""
    import torch
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

    mtime = os.path.getmtime(path)
    cache_key = f"minimind:{model_name}:{mtime}"
    if cache_key in _TEST_CACHE:
        return _TEST_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
    config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state, strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    if device.type == 'cuda':
        model = model.half()

    # 同名不同 mtime 的旧缓存清掉
    for k in list(_TEST_CACHE.keys()):
        if k.startswith(f"minimind:{model_name}:") and k != cache_key:
            _TEST_CACHE.pop(k, None)
    _TEST_CACHE[cache_key] = (model, tokenizer, device)
    return model, tokenizer, device


def _load_test_model_lora(model_name: str, lora_dir: str):
    """LoRA：从产物目录里读 meta → 加载基础模型（4-bit 优先）→ 套 adapter。

    `meta.base_model_path` 必须存在；否则直接报错让用户重新训。
    """
    import torch

    mtime = os.path.getmtime(os.path.join(lora_dir, "adapter_config.json"))
    cache_key = f"lora:{model_name}:{mtime}"
    if cache_key in _TEST_CACHE:
        return _TEST_CACHE[cache_key]

    # ── 释放其他 LoRA，防止显存堆叠 ──
    _evict_lora_cache(except_key=cache_key)

    # ── 读基座路径 ──
    meta_path = os.path.join(lora_dir, "minimind_o_meta.json")
    base_path = None
    quant_mode = "4bit_nf4"   # 默认按 4-bit 加载
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                base_path = meta.get("base_model_path")
                quant_mode = meta.get("quant_mode", quant_mode)
        except Exception:
            pass
    if not base_path:
        # 退化：从 adapter_config.json 取
        with open(os.path.join(lora_dir, "adapter_config.json"), 'r', encoding='utf-8') as f:
            base_path = (json.load(f) or {}).get("base_model_name_or_path")
    if not base_path or not os.path.isdir(base_path):
        # 再退化：去注册表里找一个 family=hf_causal_lm 的，按名字匹配
        for type_id, cfg in MODEL_REGISTRY.items():
            if cfg["family"] != "hf_causal_lm":
                continue
            if os.path.basename(os.path.normpath(cfg["base_model_path"])) == \
                    os.path.basename(os.path.normpath(base_path or "")):
                base_path = cfg["base_model_path"]
                break
    if not base_path or not os.path.isdir(base_path):
        raise FileNotFoundError(
            f"找不到该 LoRA 的基座模型目录（meta.base_model_path={base_path!r}），"
            f"无法推理。请检查 models/ 下基座是否还在。")

    # ── 加载基座 + tokenizer ──
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    # tokenizer 用 lora 目录里那份（训练时保存的，和基座一致），避免基座目录里被改过
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = (quant_mode == "4bit_nf4") and torch.cuda.is_available()
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, quantization_config=bnb,
            device_map={"": 0}, trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16,
            device_map={"": 0} if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _TEST_CACHE[cache_key] = (model, tokenizer, device)
    return model, tokenizer, device


def _load_test_model(model_name: str):
    """按模型名加载/复用：自动识别 .pth 还是 _lora/ 形态"""
    form, path = _detect_model_form(model_name)
    with _TEST_LOCK:
        if form == "pth":
            return _load_test_model_minimind(model_name, path)
        elif form == "lora":
            return _load_test_model_lora(model_name, path)
        else:
            raise RuntimeError(f"未知模型形态: {form}")


@app.route('/api/quick_test', methods=['POST'])
def quick_test():
    """快速测试训练完的模型：文本输入 → 文本输出。
    支持两种模型形态：MiniMind (.pth) 和 LoRA 适配器 (_lora/)。
    """
    if manager.get_running_task() is not None:
        return json_resp({"error": "训练正在进行中，请等训练完成后再测试"}, 400)

    d = request.json or {}
    model_name = (d.get('model') or '').strip()
    prompt = (d.get('prompt') or '').strip()
    if not model_name or not prompt:
        return json_resp({"error": "缺少 model 或 prompt"}, 400)
    if '/' in model_name or '\\' in model_name or '..' in model_name:
        return json_resp({"error": "模型名不合法"}, 400)

    try:
        model, tokenizer, device = _load_test_model(model_name)
    except FileNotFoundError as e:
        return json_resp({"error": str(e)}, 400)
    except Exception as e:
        return json_resp({"error": f"加载模型失败：{e}"}, 500)

    try:
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(text, return_tensors='pt')['input_ids'].to(device)

        max_new_tokens = int(d.get('max_tokens', 512))
        temperature = float(d.get('temperature', 0.7))

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.85,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id or 2,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 2,
            )

        new_tokens = out[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return json_resp({"response": response, "model": model_name})
    except Exception as e:
        return json_resp({"error": f"生成失败：{e}"}, 500)


# ======================== WebSocket 日志 ========================

@sock.route('/ws/train_log')
def train_log(ws):
    """
    实时推送训练日志。
    客户端连接后发送 task_id，服务端持续推送新日志行。
    """
    # 接收 task_id
    msg = ws.receive(timeout=10)
    if not msg:
        return
    data = json.loads(msg) if isinstance(msg, str) else {}
    task_id = data.get('task_id', '')
    task = manager.get_task(task_id)
    if not task:
        ws.send(json.dumps({"type": "error", "message": "任务不存在"}))
        return

    idx = 0
    try:
        while True:
            # 推送新日志
            while idx < len(task.logs):
                ws.send(json.dumps({
                    "type": "log",
                    "line": task.logs[idx],
                    "index": idx,
                }))
                idx += 1

            # 任务结束
            if task.status != "running":
                ws.send(json.dumps({
                    "type": "status",
                    "status": task.status,
                    "elapsed": round(task.elapsed, 1),
                }))
                break

            # 等待新日志
            task.log_event.wait(timeout=2)
            task.log_event.clear()

    except Exception:
        pass  # 客户端断开


# ======================== 启动 ========================

if __name__ == '__main__':
    p = argparse.ArgumentParser("MiniMind-O 训练管理 WebUI")
    p.add_argument('--port', default=7861, type=int, help='服务端口（默认 7861，与推理服务 7860 错开）')
    p.add_argument('--host', default='0.0.0.0', help='监听地址')
    args = p.parse_args()

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f'╔══════════════════════════════════════╗')
    print(f'║  MiniMind-O 训练管理 WebUI           ║')
    print(f'║  http://localhost:{args.port}              ║')
    print(f'║  数据目录: {DATASET_DIR}')
    print(f'║  输出目录: {OUT_DIR}')
    print(f'╚══════════════════════════════════════╝')

    app.run(host=args.host, port=args.port, threaded=True)
