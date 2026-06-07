# Financial Q&A LLM Fine-Tuning on AMD GPUs

This repository contains a modular codebase for fine-tuning the **Qwen2.5-1.5B-Instruct** model on financial data to perform context-based question answering. This project is configured to run on the **AMD Developer Cloud** utilizing the **ROCm ecosystem** for hardware acceleration on AMD GPUs.

---

## 📌 Project Overview

- **Dataset:** `sweatSmile/FinanceQA` — 3705 train, 927 test examples
- **Task:** Context-based Financial Q&A — given company financial data + question, generate accurate answer
- **Method:** QLoRA fine-tuning on `Qwen2.5-1.5B-Instruct`

---

## 🛠️ Project Architecture & State

The codebase is modularly structured:
- `train.py`: Handles the QLoRA fine-tuning loop for the model.
- `infer.py`: Runs model inference and provides side-by-side comparison of the base model vs. the fine-tuned model.
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

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

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
