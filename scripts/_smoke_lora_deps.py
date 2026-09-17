"""一次性脚本：验证 LoRA 训练所需依赖是否就绪。
跑完即可删除，不进版本控制（位于 scripts/ 下，可自行清理）。
"""
import sys


def check_imports():
    ok = True
    for m in ["trl", "peft", "bitsandbytes", "datasets", "accelerate"]:
        try:
            mod = __import__(m)
            print(f"  {m:14s} {getattr(mod, '__version__', '?'):10s} OK")
        except Exception as e:
            ok = False
            print(f"  {m:14s} FAIL: {type(e).__name__}: {e}")
    return ok


def check_bnb_cuda():
    """单独验证 bnb 在当前 GPU/CUDA 下是否真的能跑——光 import 不够，要构造一个 4-bit 配置"""
    try:
        import torch
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"  BitsAndBytesConfig: OK ({cfg.bnb_4bit_quant_type}/{cfg.bnb_4bit_compute_dtype})")
        return True
    except Exception as e:
        print(f"  BitsAndBytesConfig FAIL: {type(e).__name__}: {e}")
        return False


def check_trl_api():
    """trl 的 SFTConfig/SFTTrainer 接口在 0.x → 1.x 之间漂移过，明确探测一下用哪一套"""
    try:
        from trl import SFTConfig, SFTTrainer  # noqa
        print("  trl.SFTConfig: OK (≥ 0.12 新接口)")
        return True
    except Exception as e:
        print(f"  trl.SFTConfig FAIL: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("=== 依赖 import ===")
    ok1 = check_imports()
    print("=== bitsandbytes 4-bit 配置 ===")
    ok2 = check_bnb_cuda()
    print("=== trl 新接口 ===")
    ok3 = check_trl_api()
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)
