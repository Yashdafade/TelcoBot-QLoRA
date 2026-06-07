# Financial Document Question Answering (DocVQA) on AMD GPUs

This repository contains a modular codebase for fine-tuning a text-based Large Language Model (LLM) to perform **Financial Document Question Answering (DocVQA)**. Designed as part of the **TCS AMD AI Hackathon**, this project runs on the **AMD Developer Cloud** utilizing the **ROCm ecosystem** for hardware acceleration on AMD GPUs.

---

## 📌 Project Overview & Pivot

- **Previous Scope:** Text-to-text IT support generation using `benjaminmacklin/IT_Support_V2`.
- **Current Scope:** Context-Based Financial Document Question Answering (DocVQA) using the `majorSeaweed/financeQA_100K` dataset.
- **Migration Strategy (Text-Only Context):** To fit a tight sprint timeline, we bypass raw image parsing and frame this as a **Context-Based Text QA** task. We fine-tune a model (e.g., `Qwen2.5-1.5B-Instruct` or similar) using **QLoRA** (4-bit quantization + LoRA adapters) to answer questions solely using extracted document details (OCR/Markdown text).

---

## 📊 Data Mapping & Formatting

The raw dataset contains raw images, OCR/Markdown textual breakdowns of documents (metadata, key details, insights), and a JSON array of localized `[{"question": "...", "answer": "..."}]` pairs.

During preprocessing, each row is exploded into multiple training samples (one per question-answer pair) using the following instruction template:

```markdown
### System:
You are an expert financial analysis AI. Answer the question accurately using ONLY the provided document details.

### Document Details:
[Insert Document Type + Key Details + Insights here]

### Question:
[Insert Question from JSON array]

### Answer:
[Insert Corresponding Answer from JSON array]
```

---

## 🛠️ Project Architecture & State

The codebase is modularly structured:
- `train.py`: Preprocesses the `financeQA_100K` dataset, formats samples, and runs the QLoRA training loop.
- `infer.py`: Handles model inference and side-by-side comparisons of the base model vs. the fine-tuned model.
- `evaluate.py`: Computes validation metrics (ROUGE-1, ROUGE-2, ROUGE-L) to numerically demonstrate model improvement.
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
This loads the base model in 4-bit, attaches LoRA adapters, processes the new dataset, and saves the adapter weights to `./outputs/`.

### 3. Evaluation
Calculate quantitative performance metrics (ROUGE scores, latencies) on the validation set:
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
- **Evaluation Results:** Extracted in `outputs/eval_results.json` to be used for the 3-5 slide PPT and presentation video.
