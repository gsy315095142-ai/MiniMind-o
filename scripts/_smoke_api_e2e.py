"""通过 HTTP 启动 DeepSeek QLoRA 训练，跑到出现第一行 step 日志就停掉，
验证 train_server → train_manager → train_lora_sft 整条链路通畅。
"""
import json, re, time, urllib.request

BASE = "http://127.0.0.1:7861"


def _get(url):
    with urllib.request.urlopen(BASE + url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url, body):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + url, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


STEP_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] Step \[(\d+)/(\d+)\] loss=([\d.]+) avg=([\d.]+) lr=([\d.e+-]+) eta=([\d.]+)min"
)


def main():
    # 用 WebSocket 接日志比较麻烦，这里直接靠 task.logs 反查（manager 把日志存在内存里）
    # 但 server 没暴露 logs 列表的 HTTP 接口（只有 WebSocket /ws/train_log），
    # 那就先开训练，再走 WebSocket 拉日志。
    import websocket   # noqa: PLC0415

    print("启动 DeepSeek 训练…")
    r = _post("/api/start_train", {
        "dataset": "xiaohongshu.json",
        "save_name": "_e2e_deepseek",
        "model_type": "deepseek-r1-1.5b",
        "epochs": 1,
        "batch_size": 1,
        "accumulation_steps": 4,
        "max_seq_len": 256,
    })
    assert r.get("ok"), f"启动失败: {r}"
    task_id = r["task_id"]
    print(f"  task_id={task_id}, model_type={r['model_type']}")

    ws = websocket.create_connection("ws://127.0.0.1:7861/ws/train_log", timeout=300)
    ws.send(json.dumps({"task_id": task_id}))

    deadline = time.time() + 240   # 给到 4 分钟（要 4-bit 量化加载 + 数据预处理）
    matched_step = None
    log_buf = []
    saw_first_log = False
    while time.time() < deadline:
        try:
            msg = ws.recv()
        except Exception as e:
            print(f"  ws 断开: {e}")
            break
        d = json.loads(msg) if isinstance(msg, str) else {}
        if d.get("type") == "log":
            line = d["line"].rstrip("\n")
            log_buf.append(line)
            if not saw_first_log:
                print(f"  📨 收到首行日志: {line!r}")
                saw_first_log = True
            m = STEP_RE.search(line)
            if m:
                matched_step = m.group(0)
                print(f"  ✅ 命中进度条正则: {matched_step}")
                break
        elif d.get("type") == "status":
            print(f"  task 已结束: {d}")
            break

    try:
        ws.close()
    except Exception:
        pass

    print("\n--- 停止训练 ---")
    stop = _post("/api/stop_train", {"task_id": task_id})
    print(f"  停止: {stop}")
    # 给子进程一点时间退出
    time.sleep(2)
    status = _get(f"/api/train_status?task_id={task_id}")
    print(f"  最终 task 状态: {status}")

    print("\n--- 收到的日志（前 30 行）---")
    for line in log_buf[:30]:
        print("  " + line)

    if matched_step:
        print("\n✅ 端到端联调成功：训练真的跑起来了，且日志格式严格匹配进度条正则")
    else:
        print("\n❌ 没等到 step 日志，请检查 train_lora_sft 是否成功启动")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
