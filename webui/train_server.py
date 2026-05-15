"""
MiniMind-O 训练管理 WebUI 服务
独立于推理服务 (web_demo.py)，可同时运行在不同端口。

用法:
  python webui/train_server.py
  python webui/train_server.py --port 7861
"""

import argparse, os, sys, json, glob, time, threading

from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS
from flask_sock import Sock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.train_manager import TrainManager

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


# ======================== 模型管理 ========================

@app.route('/api/models')
def list_models():
    """列出 out/ 下所有 .pth 文件"""
    if not os.path.isdir(OUT_DIR):
        return json_resp({"models": []})
    files = sorted(glob.glob(os.path.join(OUT_DIR, '*.pth')))
    models = []
    for f in files:
        name = os.path.basename(f)
        size = os.path.getsize(f)
        mtime = os.path.getmtime(f)
        # 基座保护
        is_base = (name == 'llm_768.pth')
        models.append({
            "name": name,
            "size_mb": round(size / 1024 / 1024, 1),
            "time": time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime)),
            "is_base": is_base,
        })
    return json_resp({"models": models})


@app.route('/api/delete_model', methods=['POST'])
def delete_model():
    """删除指定模型权重（不允许删除基座）"""
    d = request.json or {}
    name = (d.get('name') or '').strip()
    if not name:
        return json_resp({"error": "请指定模型名"}, 400)
    if name == 'llm_768.pth':
        return json_resp({"error": "基座模型不允许删除"}, 400)
    path = os.path.join(OUT_DIR, name)
    if not os.path.isfile(path):
        return json_resp({"error": "文件不存在"}, 400)
    os.remove(path)
    return json_resp({"ok": True, "deleted": name})


# ======================== 训练控制 ========================

@app.route('/api/start_train', methods=['POST'])
def start_train():
    """启动训练任务"""
    d = request.json or {}

    # 必填参数
    dataset = (d.get('dataset') or '').strip()
    save_name = (d.get('save_name') or '').strip()
    if not dataset:
        return json_resp({"error": "请选择训练数据集"}, 400)
    if not save_name:
        return json_resp({"error": "请填写模型名称"}, 400)

    # 安全检查
    if '/' in save_name or '\\' in save_name or '..' in save_name:
        return json_resp({"error": "模型名称不合法"}, 400)

    data_path = os.path.join(DATASET_DIR, dataset)
    if not os.path.isfile(data_path):
        return json_resp({"error": f"数据文件不存在: {dataset}"}, 400)

    # 构建参数
    params = {
        "data_path": data_path,
        "epochs": int(d.get('epochs', 3)),
        "batch_size": int(d.get('batch_size', 2)),
        "learning_rate": float(d.get('learning_rate', 2e-5)),
        "save_name": save_name,
        "save_dir": OUT_DIR,
    }

    # 基座模型
    base_model = d.get('base_model', 'llm')
    base_path = os.path.join(OUT_DIR, f"{base_model}_768.pth")
    if os.path.isfile(base_path):
        params["base_model"] = base_path
    else:
        # 尝试不带 _768 后缀
        alt = os.path.join(OUT_DIR, base_model)
        if os.path.isfile(alt):
            params["base_model"] = alt
        # 找不到就用默认

    # 设备
    device = d.get('device', 'auto')
    if device != 'auto':
        params["device"] = device

    try:
        task_id = manager.start_train(params)
        return json_resp({"ok": True, "task_id": task_id})
    except RuntimeError as e:
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
# 这里只加载纯文本 LLM（MiniMindForCausalLM），不依赖 Mimi/SigLIP/SenseVoice，
# 加载快、占显存少，跟推理服务（web_demo.py）解耦。

def _load_test_model(model_name: str):
    """按模型名加载/复用一个 MiniMindForCausalLM；返回 (model, tokenizer, device)。"""
    path = os.path.join(OUT_DIR, model_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"模型文件不存在: {model_name}")

    # mtime 入 key —— 如果用户重新训练同名模型，缓存自动失效
    mtime = os.path.getmtime(path)
    cache_key = f"{model_name}:{mtime}"

    with _TEST_LOCK:
        if cache_key in _TEST_CACHE:
            return _TEST_CACHE[cache_key]

        # 延迟导入：训练时不需要这些
        import torch
        from transformers import AutoTokenizer
        from model.model_minimind import MiniMindForCausalLM, MiniMindConfig

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, "model"))
        config = MiniMindConfig(hidden_size=768, num_hidden_layers=8)
        model = MiniMindForCausalLM(config)
        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state, strict=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        if device.type == 'cuda':
            model = model.half()

        # 清掉旧版本缓存（同名不同 mtime）
        for k in list(_TEST_CACHE.keys()):
            if k.startswith(model_name + ':'):
                _TEST_CACHE.pop(k, None)
        _TEST_CACHE[cache_key] = (model, tokenizer, device)
        return model, tokenizer, device


@app.route('/api/quick_test', methods=['POST'])
def quick_test():
    """快速测试训练完的模型：文本输入 → 文本输出。"""
    # 训练进行中不允许测试（GPU 会冲突）
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
