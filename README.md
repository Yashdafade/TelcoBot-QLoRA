# Telecom Support LLM Fine-Tuning on AMD GPUs

This repository contains a modular codebase for fine-tuning the **Qwen2.5-1.5B-Instruct** model on telecom customer support conversations to handle customer inquiries effectively. This project is configured to run on the **AMD Developer Cloud** utilizing the **ROCm ecosystem** for hardware acceleration on AMD GPUs.

---

## 📌 Project Overview

- **Dataset:** `akshayjambhulkar/telecom-conversational-support-chat-pre-processed-with-agent`
- **Task:** Conversational Telecom Support — act as a customer support agent to resolve issues like VPN connectivity, roaming, SIM replacements, and billing disputes.
- **Method:** QLoRA fine-tuning on `Qwen2.5-1.5B-Instruct`
- **Training Optimization:** Utilizes `DataCollatorForLanguageModeling` along with parameter adjustments (e.g., increased training samples to 20,000, reduced sequence lengths and gradient accumulation steps) specifically tuned to accommodate AMD GPU constraints on Jupyter Notebooks.

---

## 🛠️ Project Architecture & State

The codebase is modularly structured:
- `train.py`: Handles the QLoRA fine-tuning loop for the model.
- `infer.py`: Runs model inference and provides side-by-side comparison of the base model vs. the fine-tuned model across various telecom support scenarios.
- `evaluate.py`: Computes validation metrics (such as ROUGE scores) to numerically demonstrate model improvement.
- `requirements.txt`: Manages python dependencies.
- `.gitignore`: Prevents checking in weights, logs, cache, or output checkpoints.

---

## ⚡ AMD Hardware & Environment (ROCm)

To run this project on the AMD Developer Cloud, verify ROCm compatibility:
1. Ensure the ROCm-compatible version of PyTorch is installed:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
   ```
2. Verify PyTorch detects your AMD GPU:
   ```python
   import torch
   print("ROCm / CUDA Available:", torch.cuda.is_available())
   print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
   ```

---

## 🚀 Getting Started

### 1. Setup & Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

#### Hugging Face Authentication Setup (For Gated Dataset & Models)
1. Copy the `.env.example` template:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and paste your Hugging Face write token:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```
   *(The `.env` file is automatically ignored by Git to keep your token secure).*

### 2. Fine-tuning the Model
Run the training script:
```bash
python train.py
```
This loads the base model in 4-bit, attaches LoRA adapters, processes the dataset, and saves the adapter weights to `./outputs/`.

### 3. Evaluation
Calculate quantitative performance metrics (ROUGE scores, latencies) on the test/evaluation set:
```bash
python evaluate.py
```

### 4. Interactive Inference
Compare the base model and fine-tuned model on sample test questions:
```bash
python infer.py
```

---

## 📈 Deliverables
- **Codebase:** Pushed to GitHub and cloned onto the AMD Developer Cloud instance.
- **Model Checkpoints:** Saved locally in `./outputs/` (adapters only, keeping storage usage light).
- **Evaluation Results:** Extracted in `outputs/eval_results.json` to be used for the presentation slides.
