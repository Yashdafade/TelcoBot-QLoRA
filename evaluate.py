# ============================================================
# evaluate.py — Measure Base vs Fine-Tuned Model Performance
# Uses the proper test split from sweatSmile/FinanceQA
# Run this AFTER train.py has completed
# ============================================================

import torch
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from rouge_score import rouge_scorer
import os
from huggingface_hub import login

# Try loading from a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Read token from environment variables
hf_token = os.getenv("HF_TOKEN")

# Fallback to Colab secrets if running in Google Colab
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
    except ImportError:
        pass

if hf_token:
    login(token=hf_token)
else:
    print("Warning: HF_TOKEN not found. Gated dataset access may fail.")


# ── CONFIG ──────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH   = "./outputs"
DATASET_NAME   = "sweatSmile/FinanceQA"
EVAL_SAMPLES   = 100       # use first 100 from test split
MAX_NEW_TOKENS = 128


# ── WHAT IS ROUGE ───────────────────────────────────────────
# Measures overlap between generated answer and reference answer
# ROUGE-1 = word overlap, ROUGE-2 = bigram overlap, ROUGE-L = sequence
# Score: 0.0 to 1.0 — higher means closer to reference answer


# ── QUANTIZATION ────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ── LOAD TOKENIZER ──────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── GENERATE FUNCTION ───────────────────────────────────────
def generate_response(model, context, question):
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": user_content}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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

    latency   = time.time() - start
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response  = tokenizer.decode(generated, skip_special_tokens=True)
    num_tokens = len(generated)

    return response, latency, num_tokens


# ── EVALUATE FUNCTION ───────────────────────────────────────
def evaluate_model(model, eval_data, label):
    print(f"\nEvaluating {label} on {len(eval_data)} examples...")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1, r2, rl, lats, toks = [], [], [], [], []

    for i, example in enumerate(eval_data):
        question  = example["QUERY"]
        context   = example["CONTEXT"]
        reference = example["ANSWER"]

        generated, latency, num_tokens = generate_response(model, context, question)
        scores = scorer.score(reference, generated)

        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
        lats.append(latency)
        toks.append(num_tokens)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_data)}] ROUGE-1: {scores['rouge1'].fmeasure:.3f}")

    return {
        "model"          : label,
        "rouge1_avg"     : sum(r1)   / len(r1),
        "rouge2_avg"     : sum(r2)   / len(r2),
        "rougeL_avg"     : sum(rl)   / len(rl),
        "avg_latency_sec": sum(lats) / len(lats),
        "avg_tokens"     : sum(toks) / len(toks),
    }


# ── LOAD TEST DATASET ───────────────────────────────────────
print("Loading test dataset...")
# Use the proper test split — model never saw these during training
eval_dataset = load_dataset(DATASET_NAME, split=f"test[:{EVAL_SAMPLES}]")
print(f"Evaluation set: {len(eval_dataset)} examples")


# ── EVALUATE BASE MODEL ─────────────────────────────────────
print("\nLoading BASE model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_results = evaluate_model(base_model, eval_dataset, "Base Model")

del base_model
torch.cuda.empty_cache()


# ── EVALUATE FINE-TUNED MODEL ───────────────────────────────
print("\nLoading FINE-TUNED model...")
ft_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_PATH)
ft_results = evaluate_model(ft_model, eval_dataset, "FinSense (Fine-Tuned)")

del ft_model
torch.cuda.empty_cache()


# ── PRINT RESULTS ───────────────────────────────────────────
print("\n" + "═" * 60)
print("EVALUATION RESULTS — FinSense LLM")
print("═" * 60)

metrics = ["rouge1_avg", "rouge2_avg", "rougeL_avg", "avg_latency_sec", "avg_tokens"]
labels  = ["ROUGE-1    ", "ROUGE-2    ", "ROUGE-L    ", "Latency(s) ", "Tokens     "]

print(f"\n{'Metric':<14} {'Base':<12} {'FinSense':<12} {'Change'}")
print("-" * 52)

for metric, label in zip(metrics, labels):
    bv = base_results[metric]
    fv = ft_results[metric]
    if bv > 0:
        change = ((fv - bv) / bv) * 100
        cs = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
    else:
        cs = "N/A"
    print(f"{label:<14} {bv:<12.3f} {fv:<12.3f} {cs}")

print("═" * 52)


# ── SAVE RESULTS ────────────────────────────────────────────
summary = {
    "base_model"       : base_results,
    "fine_tuned_model" : ft_results,
    "eval_samples"     : len(eval_dataset),
    "dataset"          : DATASET_NAME,
    "split"            : "test",
}

with open("./outputs/eval_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nResults saved to outputs/eval_results.json")
print("Copy these numbers into Slide 4 of your presentation.")