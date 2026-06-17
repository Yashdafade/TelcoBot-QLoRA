# 📞 Project: TelcoBot-QLoRA

**Description:** A telecom customer support LLM built by fine-tuning the `Qwen2.5-1.5B-Instruct` model.

**Dataset:** `akshayjambhulkar/telecom-conversational-support-chat-pre-processed-with-agent`

**Methodology:** QLoRA (Low-Rank Adaptation) fine-tuning using 4-bit NF4 quantization on AMD Instinct MI300X, optimized to use native `bfloat16` precision during inference and evaluation to bypass quantization latency overhead.

### ⚙️ Key Components
- **`train.py`**: SFTTrainer implementation with custom hyperparameter tuning (20k samples, batch size 4x2).
- **`infer.py`**: Interactive inference and base-model vs. fine-tuned model comparison pipeline.
- **`evaluate.py`**: Quantitative evaluation script for computing ROUGE scores, latencies, and token efficiency.

### 🎯 Objective
To create an efficient, polite, and concise support agent capable of resolving telecom issues such as:
- VPN connectivity
- International roaming
- SIM replacements
- Billing disputes
