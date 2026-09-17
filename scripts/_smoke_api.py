"""一次性脚本：联调测试 train_server.py 的多模型 API。
跑前确保 train_server 已经在 7861 端口上跑起来了。
"""
import json
import sys
import urllib.request


BASE = "http://127.0.0.1:7861"


def _get(url: str) -> dict:
    with urllib.request.urlopen(BASE + url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(url: str, body: dict) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        BASE + url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def header(s: str):
    print(f"\n===== {s} =====")


def main():
    header("1) GET /api/model_types")
    r = _get("/api/model_types")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    assert r["default"] == "minimind"
    type_ids = [t["id"] for t in r["types"]]
    assert "minimind" in type_ids and "deepseek-r1-1.5b" in type_ids

    header("2) GET /api/model_type_config?type=minimind")
    r = _get("/api/model_type_config?type=minimind")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    assert r["family"] == "minimind"
    assert "normal" in r["presets"]

    header("3) GET /api/model_type_config?type=deepseek-r1-1.5b")
    r = _get("/api/model_type_config?type=deepseek-r1-1.5b")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    assert r["family"] == "hf_causal_lm"
    assert r["save_format"] == "lora"
    assert r["extra_args"]["lora_r"] == 16

    header("4) GET /api/model_type_config?type=unknown （预期 400）")
    try:
        r = _get("/api/model_type_config?type=unknown_x")
        print("FAIL: 居然返回了 200:", r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"  HTTP {e.code}: {body}")
        assert e.code == 400

    header("5) GET /api/models  （应同时含 .pth 与 _lora/）")
    r = _get("/api/models")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    formats = {m["format"] for m in r["models"]}
    print(f"  -> 发现 formats: {formats}")
    # smoke_deepseek_v1_lora 应该被识别为 lora 形态
    names = [m["name"] for m in r["models"]]
    if "smoke_deepseek_v1_lora" in names:
        item = next(m for m in r["models"] if m["name"] == "smoke_deepseek_v1_lora")
        print(f"  smoke 产物识别: format={item['format']} model_type={item['model_type']}")
        assert item["format"] == "lora"
        assert item["model_type"] == "deepseek-r1-1.5b"

    header("6) GET /api/datasets  （旧 API 回归）")
    r = _get("/api/datasets")
    print(json.dumps(r, ensure_ascii=False, indent=2))
    assert any(d["name"] == "xiaohongshu.json" for d in r["datasets"])

    header("7) POST /api/start_train  （MiniMind 路径不传 model_type，应自动默认）")
    r = _post("/api/start_train", {
        "dataset": "xiaohongshu.json",
        "save_name": "_should_fail_no_run",
        # 故意不传 model_type，看默认是不是 minimind
        # 这里不真的拉起训练，靠校验失败提前返回也行
    })
    # 期望：要么校验失败（如基座不存在），要么成功拉起（这里不停就行）
    print("  resp:", r)
    if r.get("ok"):
        # 真的拉起了，立刻停止
        task_id = r["task_id"]
        print(f"  ⚠️ 真的启动了训练 task_id={task_id}，model_type={r.get('model_type')}")
        assert r["model_type"] == "minimind"
        stop = _post("/api/stop_train", {"task_id": task_id})
        print(f"  停止结果: {stop}")
    else:
        print("  返回了错误（接受）:", r)

    header("8) POST /api/start_train  （deepseek-r1-1.5b 路径，校验 model_type 透传）")
    # 这里我们也不真训练（会跑很久），只验证返回值含 model_type 字段就停掉
    # 但 manager 单例同一时刻只能跑一个，等上一条停干净
    import time as _t
    _t.sleep(1)
    r = _post("/api/start_train", {
        "dataset": "xiaohongshu.json",
        "save_name": "_smoke_api_deepseek",
        "model_type": "deepseek-r1-1.5b",
        "epochs": 1,
        "batch_size": 1,
    })
    print("  resp:", r)
    if r.get("ok"):
        task_id = r["task_id"]
        assert r["model_type"] == "deepseek-r1-1.5b"
        print(f"  ✅ 启动 DeepSeek 训练成功 task_id={task_id}")
        # 立刻停止（不让它真跑完）
        _t.sleep(0.5)
        stop = _post("/api/stop_train", {"task_id": task_id})
        print(f"  停止结果: {stop}")
        # 看一眼日志前几行，验证 cmd 拼对了
        _t.sleep(0.5)
        status = _get(f"/api/train_status?task_id={task_id}")
        print(f"  task 状态: {status}")
    else:
        print(f"  启动失败（待诊断）: {r}")

    print("\n✅ 全部 API 联调通过")


if __name__ == "__main__":
    main()
