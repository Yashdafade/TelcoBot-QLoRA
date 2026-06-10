# ============================================================
# evaluate.py — ROUGE Evaluation
# TelecomLLM — FINETUNING_002
# ============================================================

import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from rouge_score import rouge_scorer


# ── CONFIG ──────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH   = "./outputs"
DATASET_NAME   = "akshayjambhulkar/telecom-conversational-support-chat-pre-processed-with-agent"
EVAL_SAMPLES   = 100
MAX_NEW_TOKENS = 200


# ── QUANTIZATION ────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ── LOAD TOKENIZER ──────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── GENERATE FUNCTION ───────────────────────────────────────
def generate(model, conversation_start):
    # Feed first half of conversation, ask model to complete
    prompt = "You are a telecom customer support agent. Handle the following support conversation:\n\n" + conversation_start
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    start  = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency    = time.time() - start
    generated  = outputs[0][inputs["input_ids"].shape[1]:]
    response   = tokenizer.decode(generated, skip_special_tokens=True).strip()
    num_tokens = len(generated)
    return response, latency, num_tokens


# ── EVALUATE FUNCTION ───────────────────────────────────────
def evaluate_model(model, eval_data, label):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl, lats, toks = [], [], [], [], []

    print(f"\n  Evaluating {label}...")
    print(f"  {'Example':<10} {'ROUGE-1':<10} {'ROUGE-2':<10} {'ROUGE-L':<10} {'Latency'}")
    print(f"  {'─' * 55}")

    for i, example in enumerate(eval_data):
        text = example["text"]
        # Split conversation in half — first half is input, second half is reference
        midpoint  = len(text) // 2
        input_part = text[:midpoint]
        reference  = text[midpoint:]

        generated, latency, num_tokens = generate(model, input_part)
        scores = scorer.score(reference, generated)

        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
        lats.append(latency)
        toks.append(num_tokens)

        if (i + 1) % 10 == 0:
            avg_r1 = sum(r1) / len(r1)
            avg_r2 = sum(r2) / len(r2)
            avg_rl = sum(rl) / len(rl)
            print(f"  {i+1:<10} {avg_r1:<10.3f} {avg_r2:<10.3f} {avg_rl:<10.3f} {latency:.2f}s")

    return {
        "model"          : label,
        "rouge1_avg"     : sum(r1)   / len(r1),
        "rouge2_avg"     : sum(r2)   / len(r2),
        "rougeL_avg"     : sum(rl)   / len(rl),
        "avg_latency_sec": sum(lats) / len(lats),
        "avg_tokens"     : sum(toks) / len(toks),
    }


# ── LOAD EVAL DATASET ───────────────────────────────────────
print("\n" + "═" * 60)
print("  TelecomLLM — ROUGE Evaluation")
print("  Use Case : FINETUNING_002")
print("═" * 60)

print(f"\n  Loading {EVAL_SAMPLES} evaluation examples...")
# Use examples beyond training range (training used first 5000)
eval_dataset = load_dataset(DATASET_NAME, split=f"train[5000:5100]")
print(f"  Loaded {len(eval_dataset)} examples (unseen during training)")


# ── EVALUATE BASE ───────────────────────────────────────────
print("\n  ┌─ Phase 1: Base Model ─────────────────────────────")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
base_results = evaluate_model(base_model, eval_dataset, "Base Qwen2.5-1.5B")
del base_model
torch.cuda.empty_cache()
print("  └───────────────────────────────────────────────────")


# ── EVALUATE FINE-TUNED ─────────────────────────────────────
print("\n  ┌─ Phase 2: Fine-Tuned Model ───────────────────────")
ft_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_PATH)
ft_results = evaluate_model(ft_model, eval_dataset, "TelecomLLM (Fine-Tuned)")
del ft_model
torch.cuda.empty_cache()
print("  └───────────────────────────────────────────────────")


# ── RESULTS TABLE ───────────────────────────────────────────
print(f"\n{'═' * 60}")
print(f"  FINAL RESULTS")
print(f"{'═' * 60}")
print(f"  {'Metric':<16} {'Base':<12} {'TelecomLLM':<12} {'Δ Change'}")
print(f"  {'─' * 54}")

metrics = [
    ("ROUGE-1",    "rouge1_avg"),
    ("ROUGE-2",    "rouge2_avg"),
    ("ROUGE-L",    "rougeL_avg"),
    ("Latency (s)","avg_latency_sec"),
    ("Tokens",     "avg_tokens"),
]

for label, key in metrics:
    bv = base_results[key]
    fv = ft_results[key]
    delta = ((fv - bv) / bv) * 100 if bv > 0 else 0
    arrow = "▲" if delta > 0 else "▼"
    print(f"  {label:<16} {bv:<12.3f} {fv:<12.3f} {arrow} {abs(delta):.1f}%")

print(f"{'═' * 60}")


# ── SAVE ────────────────────────────────────────────────────
summary = {
    "base_model"       : base_results,
    "fine_tuned_model" : ft_results,
    "eval_samples"     : len(eval_dataset),
    "dataset"          : DATASET_NAME,
    "eval_range"       : "train[5000:5100]",
}
with open("./outputs/eval_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Results saved → outputs/eval_results.json")
print(f"{'═' * 60}\n")