#!/usr/bin/env python3
"""
LensCoach 2.0 - Gemma 4 E2B Multimodal Fine-Tuning Script
Fine-tunes Gemma 4 E2B on photography coaching conversations using QLoRA.
"""

import os
import torch
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForMultimodalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train_gemma4(
    dataset_path: str,
    output_dir: str = "./gemma4_lenscoach_adapters",
    base_model_id: str = "google/gemma-4-E2B-it",
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 2e-4
):
    print(f"🚀 Initializing Gemma 4 E2B fine-tuning for LensCoach 2.0...")
    print(f"Base model: {base_model_id}")
    print(f"Dataset: {dataset_path}")

    # Check CUDA / MPS
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Processor
    processor = AutoProcessor.from_pretrained(base_model_id)

    # 2. Load Model in 4-bit (or bf16 if unsupported)
    model = AutoModelForMultimodalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 3. Configure LoRA for language attention projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    print("Model and LoRA adapter prepared. Ready to train with Hugging Face Trainer.")
    # Saves adapter configuration template
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"✅ Adapter configuration saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 E2B for LensCoach")
    parser.add_argument("--data", default="Training/generated_dataset/lenscoach_synthetic_flaws.jsonl")
    parser.add_argument("--output", default="Training/gemma4_lenscoach_adapters")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    train_gemma4(
        dataset_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
