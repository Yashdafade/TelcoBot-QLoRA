# ============================================================
# infer.py — Compare Base Model vs Fine-Tuned Model
# Run this AFTER train.py has completed
# Shows side by side output for finance questions
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ── CONFIG ──────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH   = "./outputs"
MAX_NEW_TOKENS = 256


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
    # Same format as training — context + question in user turn
    user_content = f"Context:\n{context}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": user_content}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ── COMPARE FUNCTION ────────────────────────────────────────
def compare(context, question, reference=None):
    print("\n" + "═" * 60)
    print(f"QUESTION: {question}")
    if reference:
        print(f"REFERENCE ANSWER: {reference}")
    print("═" * 60)

    # Base model
    print("\nLoading BASE model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_response = generate_response(base_model, context, question)
    print(f"\n[BASE MODEL]\n{base_response}")

    del base_model
    torch.cuda.empty_cache()

    # Fine-tuned model
    print("\nLoading FINE-TUNED model...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_PATH)

    ft_response = generate_response(ft_model, context, question)
    print(f"\n[FINE-TUNED — FinSense]\n{ft_response}")

    del ft_model
    torch.cuda.empty_cache()

    print("\n" + "═" * 60)


# ── TEST EXAMPLES ───────────────────────────────────────────
# Real examples from the dataset — these are unseen test cases
# The fine-tuned model should answer more accurately and concisely

examples = [
    {
        "context": "Symbol: INFY Company Name: Infosys Ltd. Equity Share Capital: 2079 Total Share Capital: 2079 Reserves and Surplus: 74710 Total Reserves and Surplus: 74710 Total Shareholders Funds: 76789 Long Term Borrowings: 0 Total Non-Current Liabilities: 458 Trade Payables: 3200 Total Current Liabilities: 18435 Total Capital And Liabilities: 95682",
        "question": "What is the total shareholders fund of the company?",
        "reference": "The total shareholders fund of the company is 76789."
    },
    {
        "context": "Symbol: TCS Company Name: Tata Consultancy Services Ltd. Revenue From Operations: 213200 Other Income: 4200 Total Revenue: 217400 Employee Benefit Expense: 112000 Total Expenses: 168900 Profit Before Tax: 48500 Tax Expense: 12100 Profit After Tax: 36400",
        "question": "What is the profit after tax of the company?",
        "reference": "The profit after tax of the company is 36400."
    },
    {
        "context": "Symbol: RELIANCE Company Name: Reliance Industries Ltd. Total Assets: 1650000 Total Non-Current Assets: 980000 Total Current Assets: 670000 Cash And Cash Equivalents: 85000 Short Term Borrowings: 45000 Long Term Borrowings: 230000 Total Debt: 275000",
        "question": "What is the total debt of the company?",
        "reference": "The total debt of the company is 275000."
    },
    {
        "context": "Symbol: HDFC Company Name: HDFC Bank Ltd. Net Interest Income: 85000 Other Income: 22000 Operating Expenses: 45000 Provisions: 12000 Profit Before Tax: 50000 Tax: 12500 Net Profit: 37500 Total Assets: 1800000 Gross NPA: 9200 Net NPA: 3100",
        "question": "What is the Net NPA of the company?",
        "reference": "The Net NPA of the company is 3100."
    },
]

# ── RUN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FinSense LLM — Base vs Fine-Tuned Comparison")
    print("=" * 60)

    for ex in examples:
        compare(ex["context"], ex["question"], ex.get("reference"))

    print("\nDone. Record this output for your demo video.")