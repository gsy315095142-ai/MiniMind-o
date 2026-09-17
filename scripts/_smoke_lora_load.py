"""一次性脚本：纯加载 smoke test。
只走 tokenizer → 4-bit 模型加载 → LoRA 注入 → 一次前向，不跑训练。
目的是在真正训练前先把 bitsandbytes / peft / DeepSeek 这条加载链路打通。
"""
import os, sys, time, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = "models/deepseek-r1-1.5b"


def main():
    t0 = time.time()
    print(f"[1/5] 加载 tokenizer：{MODEL_PATH}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  vocab={len(tok)}  pad_token_id={tok.pad_token_id}  eos_token_id={tok.eos_token_id}")

    print(f"[2/5] 构造 BitsAndBytesConfig (4-bit NF4 + bf16 compute)")
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[3/5] 加载模型 (4-bit)…")
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        trust_remote_code=True,
    )
    print(f"  加载耗时 {time.time()-t1:.1f}s")
    print(f"  显存占用 {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    print(f"[4/5] 注入 LoRA")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练 {trainable/1e6:.2f}M / 总 {total/1e6:.2f}M  "
          f"({100*trainable/max(total,1):.2f}%)")

    print(f"[5/5] 一次前向 + 一次反传")
    msgs = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，请问需要什么帮助？"},
    ]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    input_ids = torch.tensor([ids], device="cuda")
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    print(f"  forward 完成，loss={out.loss.item():.4f}")
    out.loss.backward()
    print(f"  backward 完成（梯度已计算）")
    print(f"  显存峰值 {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"=== 总耗时 {time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
