# ============================================================
# train.py — QLoRA Fine-tuning for Finance Q&A LLM
# Model  : Qwen2.5-1.5B-Instruct
# Method : QLoRA (4-bit quantization + LoRA adapters)
# Data   : sweatSmile/FinanceQA
# ============================================================

# ── 1. IMPORTS ──────────────────────────────────────────────
import os
import torch

from huggingface_hub import login

# Try loading from a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Read token from environment variables (for AMD Cloud, Local, or other clouds)
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


from transformers import (
    AutoModelForCausalLM,   # loads the LLM
    AutoTokenizer,           # loads the tokenizer (text → tokens)
    BitsAndBytesConfig,      # controls 4-bit quantization
    TrainingArguments,       # all training hyperparameters
)

from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset


# ── 2. CONFIGURATION ────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"  # base model to fine-tune
DATASET_NAME = "sweatSmile/FinanceQA"         # finance Q&A dataset
OUTPUT_DIR   = "./outputs"                     # where to save fine-tuned model

# Dataset has 3705 train examples — use all of them
# Small dataset = more epochs needed to see improvement
TRAIN_SAMPLES = 3705

# LoRA hyperparameters
LORA_R       = 16     # rank of adapter matrices
LORA_ALPHA   = 32     # scaling factor (always 2x rank)
LORA_DROPOUT = 0.05   # regularization dropout

# Training hyperparameters
EPOCHS      = 3       # 3 epochs since dataset is small (3705 examples)
BATCH_SIZE  = 4       # examples processed at once
LR          = 2e-4    # learning rate
MAX_SEQ_LEN = 768     # increased from 512 — finance context strings are longer


# ── 3. QUANTIZATION CONFIG ──────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ── 4. LOAD MODEL + TOKENIZER ───────────────────────────────
print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

print(f"Model loaded. Parameters: {model.num_parameters():,}")


# ── 5. LORA CONFIG ──────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ── 6. LOAD + FORMAT DATASET ────────────────────────────────
print("Loading dataset...")

dataset = load_dataset(DATASET_NAME, split="train")

# This dataset has QUERY, ANSWER, CONTEXT columns — not chat format
# We build the chat messages manually:
# - user turn  = CONTEXT (financial data) + QUERY (question)
# - assistant  = ANSWER
# Including CONTEXT is important — it gives the model the financial figures
# to reason over, not just memorize answers

def format_chat(example):
    # Combine context + query into user message
    # This mirrors how the model would be used in production:
    # user provides financial data + asks a question
    user_content = f"Context:\n{example['CONTEXT']}\n\nQuestion: {example['QUERY']}"

    messages = [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": example["ANSWER"]},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_chat)
print(f"Dataset loaded. {len(dataset)} examples.")
print("Sample formatted example:")
print(dataset[0]["text"][:500], "...")   # print first 500 chars only
print("─" * 60)


# ── 7. TRAINING ARGUMENTS ───────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.001,
    bf16=True,
    logging_steps=25,          # log more frequently (smaller dataset)
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    optim="paged_adamw_8bit",
)


# ── 8. TRAINER ──────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,   # ← updated
    train_dataset=dataset,
    args=training_args,
)


# ── 9. TRAIN ────────────────────────────────────────────────
print("Starting training...")
print(f"Training on {len(dataset)} examples for {EPOCHS} epochs")
print(f"LoRA rank={LORA_R}, alpha={LORA_ALPHA}")
print("─" * 60)

trainer.train()


# ── 10. SAVE ────────────────────────────────────────────────
print("Saving fine-tuned model...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
print("Training complete.")