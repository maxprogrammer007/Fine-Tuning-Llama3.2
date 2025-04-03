# Fine-Tuning LLaMA 3.2 (3B) with Unsloth

## 📌 Project Overview
This repository contains the implementation for fine-tuning the **LLaMA 3.2 (3B) model** using **PEFT (Parameter-Efficient Fine-Tuning)** techniques. The goal is to enhance the performance of the model using a dataset from **Hugging Face** while keeping memory usage efficient.

## 🚀 Features
- **Uses FastLanguageModel from Unsloth** for optimized performance
- **PEFT with LoRA (Low-Rank Adaptation)** for efficient fine-tuning
- **Standardized Chat Formatting** using Unsloth's chat templates
- **Fine-tuning with SFTTrainer** for better response generation
- **Supports inference on fine-tuned model**

## 📂 Dataset
We used the **mlabonne/FineTome-100k** dataset from Hugging Face for training.

## 📌 Installation
Ensure you have Python 3.8+ and install dependencies:
```bash
pip install unsloth transformers trl datasets
```

## 🛠 Fine-Tuning Process
1. **Load Base Model & Tokenizer**  
   - Utilizes "unsloth/Llama-3.2-3B-Instruct" for initialization.

2. **Apply PEFT & LoRA**  
   - Fine-tunes select layers for efficient training.

3. **Preprocess Dataset**  
   - Uses Unsloth's `standardize_sharegpt()` for structured chat data.

4. **Train the Model**  
   - Configured with gradient accumulation, mixed precision (FP16/BF16), and SFTTrainer.

5. **Save and Deploy**  
   - Saves fine-tuned model and loads it for inference.

## 🔬 Inference Example
After fine-tuning, you can run inference using:
```python
prompts = ["Explain the principles of investment."]
inputs = tokenizer.apply_chat_template([{ "role": "user", "content": prompts[0] }], tokenize=False)
output = model.generate(**tokenizer(inputs, return_tensors="pt").to("cuda"))
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 📌 Training Configuration
- **Batch Size:** 2 per device
- **Gradient Accumulation:** 4 steps
- **Warmup Steps:** 5
- **Max Steps:** 60
- **Learning Rate:** 2e-4
- **Mixed Precision:** FP16/BF16
- **Logging:** Every step


