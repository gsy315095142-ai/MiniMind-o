"""MiniMind 训练回归测试：Phase 2 改了 train_manager.start_train，
必须证明 MiniMind 路径不坏——拉起、收到首条 step 日志，再停掉。
"""
import json, re, time, urllib.request, websocket

BASE = "http://127.0.0.1:7861"
STEP_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\] Step \[(\d+)/(\d+)\] loss=([\d.]+)"
)


def _post(url, body):
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + url, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url):
    with urllib.request.urlopen(BASE + url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    print("启动 MiniMind 训练（不传 model_type，应默认 minimind）…")
    r = _post("/api/start_train", {
        "dataset": "xiaohongshu.json",
        "save_name": "_regression_minimind",
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 2e-5,
    })
    assert r.get("ok"), f"启动失败: {r}"
    task_id = r["task_id"]
    print(f"  task_id={task_id}, model_type={r.get('model_type')}")
    assert r.get("model_type") == "minimind"

    ws = websocket.create_connection("ws://127.0.0.1:7861/ws/train_log", timeout=120)
    ws.send(json.dumps({"task_id": task_id}))

    deadline = time.time() + 90
    matched = None
    logs = []
    while time.time() < deadline:
        try:
            msg = ws.recv()
        except Exception:
            break
        d = json.loads(msg) if isinstance(msg, str) else {}
        if d.get("type") == "log":
            line = d["line"].rstrip("\n")
            logs.append(line)
            m = STEP_RE.search(line)
            if m:
                matched = m.group(0)
                print(f"  ✅ MiniMind 首条 step 日志: {matched}")
                break
        elif d.get("type") == "status":
            break

    try:
        ws.close()
    except Exception:
        pass

    stop = _post("/api/stop_train", {"task_id": task_id})
    print(f"  停止: {stop}")
    time.sleep(2)
    status = _get(f"/api/train_status?task_id={task_id}")
    print(f"  最终状态: {status}")

    print("\n--- 前 20 行 ---")
    for ln in logs[:20]:
        print("  " + ln)

    assert matched, "❌ MiniMind 没等到 step 日志，可能回归被破坏"
    print("\n✅ MiniMind 回归通过：训练能拉起、日志格式仍然对齐")


if __name__ == "__main__":
    main()
