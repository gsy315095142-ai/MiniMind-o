"""
训练进程管理器
- 通过 subprocess 拉起 train_text_sft.py
- 后台线程逐行读取 stdout
- 支持多个训练任务（MVP 限制同时只跑一个）
- 提供 logs 列表 + 事件通知，供 WebSocket 推送
"""

import subprocess, sys, threading, time, os, uuid


# ── 错误信息翻译表 ──
# 把训练子进程吐出来的英文 traceback / 异常关键字翻译成新手能看懂的中文提示。
# 越靠前的规则优先级越高（匹配到就停止）。
_ERROR_HINTS = [
    ("CUDA out of memory",
     "💔 显存不够了！请把「Batch Size」改小一点（比如改成 1），或选择更小的训练数据。"),
    ("out of memory",
     "💔 内存/显存不足，请减小 Batch Size 或关闭其他占显存的程序后重试。"),
    ("No such file or directory",
     "📁 找不到指定的文件，请检查数据集 / 基座模型路径是否还存在。"),
    ("FileNotFoundError",
     "📁 找不到指定的文件，请检查数据集 / 基座模型路径是否还存在。"),
    ("JSONDecodeError",
     "📝 训练数据 JSON 格式有问题，请回到「准备数据」页用「验证格式」检查后重新保存。"),
    ("UnicodeDecodeError",
     "📝 文件编码异常，请确保 JSON 数据用 UTF-8 编码保存。"),
    ("UnicodeEncodeError",
     "🔤 控制台编码不支持 emoji。请重启训练 WebUI（脚本已修复，强制 UTF-8 编码）。"),
    ("PermissionError",
     "🔒 文件权限不足，请确认 out/ 目录可写、模型文件没被其他程序占用。"),
    ("ConnectionError",
     "🌐 网络连接失败，请检查网络后重试。"),
    ("CUDA error",
     "🎮 显卡运行异常，请尝试重启电脑后再训练，或切换到 CPU（在「高级参数」里改 device=cpu）。"),
    ("size mismatch",
     "🧩 基础模型与当前模型结构不匹配，请确认选择的是同一个版本的基座权重。"),
    ("KeyboardInterrupt",
     "⚠️ 训练被中断。"),
]


def translate_error(text: str) -> str | None:
    """根据日志/异常文本，返回友好的中文提示；找不到返回 None。"""
    if not text:
        return None
    for keyword, hint in _ERROR_HINTS:
        if keyword in text:
            return hint
    return None


class TrainTask:
    """单个训练任务的状态容器"""

    def __init__(self, task_id: str, proc: subprocess.Popen, save_name: str):
        self.task_id = task_id
        self.proc = proc
        self.save_name = save_name
        self.status = "running"       # running / done / failed / stopped
        self.logs: list[str] = []     # 所有日志行
        self.log_event = threading.Event()  # 有新日志时 set
        self.start_time = time.time()
        self.reader_thread: threading.Thread | None = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "save_name": self.save_name,
            "status": self.status,
            "elapsed": round(self.elapsed, 1),
        }


class TrainManager:
    """管理所有训练任务（单例）"""

    def __init__(self):
        self.tasks: dict[str, TrainTask] = {}
        self._lock = threading.Lock()

    # ---- 启动训练 ----
    def start_train(self, params: dict) -> str:
        """
        params 必须包含:
          data_path, epochs, batch_size, learning_rate, save_name
        可选:
          base_model, max_seq_len, dtype, device
        返回 task_id
        """
        with self._lock:
            # MVP: 同时只能有一个训练
            for t in self.tasks.values():
                if t.status == "running":
                    raise RuntimeError("已有训练任务正在运行，请等待完成或停止后再试")

        task_id = uuid.uuid4().hex[:8]
        save_name = params["save_name"]

        # 项目根目录 = train_manager.py 的上上级
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # sys.executable 保证使用的是当前 Python 解释器（不再硬编码 "python"）
        # -u 保证 stdout 无缓冲，日志可实时推送
        cmd = [
            sys.executable, "-u", "trainer/train_text_sft.py",
            "--data_path", params["data_path"],
            "--epochs", str(params["epochs"]),
            "--batch_size", str(params["batch_size"]),
            "--learning_rate", str(params["learning_rate"]),
            "--save_name", save_name,
            "--save_dir", params.get("save_dir", os.path.join(root, "out")),
            "--tokenizer_path", os.path.join(root, "model"),
        ]

        if params.get("base_model"):
            cmd += ["--base_model", params["base_model"]]
        if params.get("max_seq_len"):
            cmd += ["--max_seq_len", str(params["max_seq_len"])]
        if params.get("dtype"):
            cmd += ["--dtype", params["dtype"]]
        if params.get("device"):
            cmd += ["--device", params["device"]]

        # Windows 上 Python 默认 stdout 编码是 GBK，训练脚本里有 emoji（📝🚀✅ 等）
        # 一旦 print 就会触发 UnicodeEncodeError。这里强制子进程 stdout 用 UTF-8，
        # 并用 errors='replace' 兜底，保证父子进程都不会因为非法字节挂掉。
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=root,
            bufsize=1,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8",  # ← 关键：让子进程 print emoji 不再崩
            },
        )

        task = TrainTask(task_id, proc, save_name)
        self.tasks[task_id] = task

        # 后台线程读取 stdout
        def _reader():
            try:
                for line in proc.stdout:
                    task.logs.append(line)
                    task.log_event.set()
            except Exception:
                pass
            finally:
                proc.wait()
                # status 可能已被 stop_train 设置为 stopped，不要覆盖
                if task.status == "running":
                    task.status = "done" if proc.returncode == 0 else "failed"

                # 收尾日志 + 失败时尝试翻译最后几行 stderr
                if task.status == "done":
                    task.logs.append("\n✅ 训练结束\n")
                else:
                    # 在最后 50 行里搜索关键字给出友好提示
                    tail = ''.join(task.logs[-50:])
                    hint = translate_error(tail)
                    if hint:
                        task.logs.append(f"\n💡 {hint}\n")
                    task.logs.append(f"❌ 训练结束 (exit code: {proc.returncode})\n")
                task.log_event.set()

        t = threading.Thread(target=_reader, daemon=True)
        task.reader_thread = t
        t.start()

        return task_id

    # ---- 停止训练 ----
    def stop_train(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if not task or task.status != "running":
            return False
        task.proc.terminate()
        task.status = "stopped"
        task.logs.append("\n⚠️ 训练已被手动停止\n")
        task.log_event.set()
        return True

    # ---- 查询 ----
    def get_task(self, task_id: str) -> TrainTask | None:
        return self.tasks.get(task_id)

    def list_tasks(self) -> list[dict]:
        return [t.to_dict() for t in self.tasks.values()]

    def get_running_task(self) -> TrainTask | None:
        for t in self.tasks.values():
            if t.status == "running":
                return t
        return None
