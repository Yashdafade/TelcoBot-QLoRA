# ============================================================
# train.py — QLoRA Fine-tuning for Telecom Support LLM
# Model  : Qwen2.5-1.5B-Instruct
# Method : QLoRA (4-bit quantization + LoRA adapters)
# Data   : akshayjambhulkar/telecom-conversational-support-chat-pre-processed-with-agent
# Use Case: FINETUNING_002 — Telco Specific Customer LLM
# ============================================================

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset


# ── CONFIGURATION ───────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "akshayjambhulkar/telecom-conversational-support-chat-pre-processed-with-agent"
OUTPUT_DIR   = "./outputs"

TRAIN_SAMPLES = 5000   # out of 228k — enough for strong improvement
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
EPOCHS        = 3
BATCH_SIZE    = 4
LR            = 2e-4
MAX_SEQ_LEN   = 1024   # conversations are longer than Q&A pairs


# ── QUANTIZATION CONFIG ─────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ── LOAD MODEL + TOKENIZER ──────────────────────────────────
print("\n" + "═" * 60)
print("  TelecomLLM — QLoRA Fine-Tuning Pipeline")
print("  Use Case : FINETUNING_002 — Telco Customer LLM")
print("  Model    : Qwen2.5-1.5B-Instruct")
print("  Method   : QLoRA (4-bit NF4 + LoRA Adapters)")
print("═" * 60)

print("\n[1/5] Loading tokenizer and base model...")

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

print(f"    Base model loaded  : {MODEL_NAME}")
print(f"    Total parameters   : {model.num_parameters():,}")


# ── LORA CONFIG ─────────────────────────────────────────────
print("\n[2/5] Applying LoRA adapters...")

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"    Trainable params   : {trainable:,} ({100 * trainable / total:.3f}% of total)")
print(f"    LoRA rank          : r={LORA_R}, alpha={LORA_ALPHA}")
print(f"    Target modules     : q/k/v/o_proj + gate/up/down_proj (MLP)")


# ── LOAD + FORMAT DATASET ───────────────────────────────────
print(f"\n[3/5] Loading dataset ({TRAIN_SAMPLES} examples)...")

dataset = load_dataset(DATASET_NAME, split=f"train[:{TRAIN_SAMPLES}]")

# Dataset has 'text' column with full agent-client conversations
# Format: "agent: ... client: ... agent: ..."
# We wrap each conversation in the chat template as a single user turn
# The model learns the full support conversation pattern

def format_chat(example):
    messages = [
        {
            "role": "user",
            "content": "You are a telecom customer support agent. Handle the following support conversation:\n\n" + example["text"]
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"formatted_text": text}

dataset = dataset.map(format_chat, remove_columns=["conversation_id", "text"])

print(f"    Dataset loaded     : {DATASET_NAME}")
print(f"    Training examples  : {len(dataset)}")
print(f"    Sample preview     :")
preview = dataset[0]["formatted_text"][:300].replace("\n", " ")
print(f"    {preview}...")


# ── TRAINING ARGUMENTS ──────────────────────────────────────
print("\n[4/5] Configuring training...")

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
    logging_steps=25,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    optim="paged_adamw_8bit",
)

print(f"    Epochs             : {EPOCHS}")
print(f"    Batch size         : {BATCH_SIZE} (effective {BATCH_SIZE * 4} with grad accum)")
print(f"    Learning rate      : {LR}")
print(f"    Precision          : bf16")
print(f"    Optimizer          : paged_adamw_8bit")


# ── TRAINER ─────────────────────────────────────────────────
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    dataset_text_field="formatted_text",
    max_seq_length=MAX_SEQ_LEN,
    data_collator=collator,
    args=training_args,
)


# ── TRAIN ───────────────────────────────────────────────────
print("\n[5/5] Starting training...\n")
print("─" * 60)
print(f"  {'Step':<8} {'Loss':<12} {'Progress'}")
print("─" * 60)

trainer.train()


# ── SAVE ────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("\nSaving fine-tuned LoRA adapters...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n{'═' * 60}")
print(f"  Training complete!")
print(f"  Model saved to     : {OUTPUT_DIR}")
print(f"  Run infer.py next  : python infer.py")
print(f"{'═' * 60}\n")