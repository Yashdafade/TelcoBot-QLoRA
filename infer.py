# ============================================================
# infer.py — Base vs Fine-Tuned Model Comparison
# TelecomLLM — FINETUNING_002
# ============================================================

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ── CONFIG ──────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH   = "./outputs"
MAX_NEW_TOKENS = 300


# ── QUANTIZATION ────────────────────────────────────────────
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )


# ── LOAD TOKENIZER ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── GENERATE FUNCTION ───────────────────────────────────────
def generate(model, scenario):
    prompt = "You are a telecom customer support agent. Handle the following support conversation:\n\n" + scenario
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    tokens    = len(generated)
    return response, latency, tokens


# ── PRINT RESULT BLOCK ──────────────────────────────────────
def print_result(label, response, latency, tokens, is_finetuned=False):
    tag = "✦ FINETUNED" if is_finetuned else "  BASE     "
    print(f"\n  ┌─ [{tag}] ─────────────────────────────────────")
    # Word wrap at 72 chars
    words = response.split()
    line  = "  │  "
    for word in words:
        if len(line) + len(word) + 1 > 76:
            print(line)
            line = "  │  " + word
        else:
            line += (" " if line != "  │  " else "") + word
    if line != "  │  ":
        print(line)
    print(f"  │")
    print(f"  │  Latency : {latency:.2f}s   Tokens : {tokens}")
    print(f"  └─────────────────────────────────────────────────")


# ── COMPARE ─────────────────────────────────────────────────
def compare(scenario, description):
    print(f"\n{'═' * 60}")
    print(f"  SCENARIO : {description}")
    print(f"{'═' * 60}")
    print(f"  Input    : {scenario}{'...' if len(scenario) > 120 else ''}")
    print(f"{'─' * 60}")

    # Base model
    print("\n  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
    base_resp, base_lat, base_tok = generate(base_model, scenario)
    print_result("BASE", base_resp, base_lat, base_tok, is_finetuned=False)
    del base_model
    torch.cuda.empty_cache()

    # Fine-tuned model
    print("\n  Loading fine-tuned model...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_PATH)
    ft_resp, ft_lat, ft_tok = generate(ft_model, scenario)
    print_result("FINETUNED", ft_resp, ft_lat, ft_tok, is_finetuned=True)
    del ft_model
    torch.cuda.empty_cache()

    # Delta summary
    lat_delta = ((ft_lat - base_lat) / base_lat) * 100
    tok_delta = ((ft_tok - base_tok) / base_tok) * 100
    print(f"\n  ┌─ DELTA ──────────────────────────────────────────")
    print(f"  │  Latency : {base_lat:.2f}s → {ft_lat:.2f}s  ({lat_delta:+.1f}%)")
    print(f"  │  Tokens  : {base_tok}  → {ft_tok}  ({tok_delta:+.1f}%)")
    print(f"  └──────────────────────────────────────────────────")


# ── TEST SCENARIOS ──────────────────────────────────────────
scenarios = [
{
    "description": "eSIM Transfer Between Devices",
    "input": "client: I want to transfer my eSIM from my old phone to my new Samsung Galaxy S24. The old phone is broken so I can't access it anymore. Can you help?\nagent:"
},
{
    "description": "Unexpected Bill Charge",
    "input": "client: My bill this month is $30 higher than usual and I don't understand why. I haven't changed my plan or used anything extra. Can you explain these charges?\nagent:"
},
]


# ── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  TelecomLLM — Base vs Fine-Tuned Comparison")
    print("  Use Case : FINETUNING_002")
    print("═" * 60)

    for s in scenarios:
        compare(s["input"], s["description"])

    print(f"\n{'═' * 60}")
    print("  Inference complete. Run evaluate.py for ROUGE metrics.")
    print(f"{'═' * 60}\n")