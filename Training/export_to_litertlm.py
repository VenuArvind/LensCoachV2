#!/usr/bin/env python3
"""
LensCoach 2.0 - Model Export to LiteRT-LM (.litertlm)
Merges LoRA adapter into base Gemma 4 E2B weights and converts to Google AI Edge .litertlm
for native, memory-mapped Metal GPU execution on iOS.
"""

import os
import argparse
import subprocess
import torch
from pathlib import Path

def merge_and_export_litertlm(
    base_model_id: str,
    adapter_path: str,
    merged_output_dir: str,
    final_litert_dir: str
):
    print("🔄 Step 1: Merging LoRA adapters into base model weights...")
    
    from transformers import AutoModelForConditionalGeneration, AutoTokenizer
    from peft import PeftModel

    # Load base model & apply LoRA weights
    print(f"Loading base: {base_model_id}")
    model = AutoModelForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    print(f"Applying adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = model.merge_and_unload()
    
    Path(merged_output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(merged_output_dir)
    print(f"✅ Merged model saved to {merged_output_dir}")

    print("\n📦 Step 2: Compiling to Google AI Edge .litertlm container...")
    # Calls litert-torch export_hf with vision encoder support and dynamic int8 quantization
    cmd = [
        "litert-torch", "export_hf",
        f"--model={merged_output_dir}",
        f"--output_dir={final_litert_dir}",
        "--task=image_text_to_text",
        "--export_vision_encoder=True",
        "--vision_encoder_quantization_recipe=weight_only_wi8_afp32",
        "--quantization_recipe=dynamic_wi8_afp32",
        "--externalize_embedder=True",
        "--bundle_litert_lm=True"
    ]
    
    print("Running command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"\n🎉 Successfully exported model to {final_litert_dir}/model.litertlm!")
        print("Ready to copy to LensCoachApp/Sources/Resources/gemma-4-E2B-it-gpu.litertlm")
    except Exception as e:
        print(f"Note: Run this step in an environment with 'litert-torch' installed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned Gemma 4 to .litertlm")
    parser.add_argument("--base_model", default="google/gemma-4-E2B-it")
    parser.add_argument("--adapter", default="Training/gemma4_lenscoach_adapters")
    parser.add_argument("--merged_dir", default="/tmp/lenscoach_gemma4_merged")
    parser.add_argument("--output_litert", default="LensCoachApp/Sources/Resources")
    args = parser.parse_args()

    merge_and_export_litertlm(args.base_model, args.adapter, args.merged_dir, args.output_litert)
