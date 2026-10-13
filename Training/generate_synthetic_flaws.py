#!/usr/bin/env python3
"""
LensCoach 2.0 - Synthetic Photography Flaw Generator
Generates realistic photography flaws (horizon tilt, framing offsets, exposure errors)
from high-aesthetic images, creating paired (flawed_image, ground_truth_telemetry, target_fix_json)
strictly matching the output schemas in lenscoach_v2_plan.md Section 7.
"""

import os
import json
import random
import argparse
from pathlib import Path

def apply_horizon_tilt(image, angle_degrees):
    """Rotates image by angle and crops back to center to simulate tilted camera."""
    from PIL import Image
    w, h = image.size
    rotated = image.rotate(angle_degrees, resample=Image.Resampling.BICUBIC, expand=False)
    # Center crop by 90% to eliminate black rotated corners
    crop_w, crop_h = int(w * 0.90), int(h * 0.90)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return rotated.crop((left, top, left + crop_w, top + crop_h))

def apply_exposure_shift(image, factor):
    """Adjusts exposure factor (< 1.0 is underexposed, > 1.0 is overexposed)."""
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def generate_synthetic_samples(input_images_dir, output_dir, count=500):
    """Generates synthetic flaw dataset for Gemma 4 coach fine-tuning."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. Please install pillow: pip install pillow")
        return

    output_dir = Path(output_dir)
    images_out = output_dir / "synthetic_images"
    images_out.mkdir(parents=True, exist_ok=True)
    
    input_files = [
        f for f in Path(input_images_dir).rglob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    
    if not input_files:
        print(f"No source images found in {input_images_dir}. Creating synthetic color cards...")
        # Create base compositional images if source directory is empty
        input_files = []
        for dummy_i in range(10):
            dummy_path = images_out / f"base_{dummy_i}.jpg"
            img = Image.new("RGB", (600, 450), color=(
                random.randint(50, 200),
                random.randint(60, 210),
                random.randint(70, 220)
            ))
            img.save(dummy_path)
            input_files.append(dummy_path)

    print(f"Found {len(input_files)} source images. Generating {count} synthetic coaching pairs...")
    live_records = []
    critique_records = []

    for i in range(count):
        src_path = random.choice(input_files)
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                
                # Alternate between Live Tip and Shot Critique samples
                is_critique_sample = (i % 3 == 0)
                flaw_type = random.choice(["tilt", "underexposed", "overexposed", "framing"])
                record_id = f"synth_{i:05d}"
                out_filename = f"{record_id}.jpg"
                out_filepath = images_out / out_filename
                
                if flaw_type == "tilt":
                    tilt = random.choice([-1, 1]) * round(random.uniform(2.0, 7.5), 1)
                    flawed_img = apply_horizon_tilt(img, tilt)
                    flawed_img.save(out_filepath, quality=90)
                    
                    target_dir = "clockwise" if tilt > 0 else "counter-clockwise"
                    tip_text = f"Tilt ~{abs(tilt):.0f}° {target_dir} to level the horizon"
                    live_model_json = {
                        "tip": tip_text,
                        "action": "rotate",
                        "params": {"degrees": abs(tilt)},
                        "priority": "high"
                    }
                    critique_model_json = {
                        "overall": 6.4,
                        "summary": "Dynamic capture but noticeably tilted horizon line distracts from subject.",
                        "scores": {"composition": 6, "light": 7, "color": 7, "focus": 8, "subject": 7},
                        "strengths": ["Clean focal plane on main subject", "Natural color reproduction"],
                        "issues": [{
                            "aspect": "composition",
                            "problem": f"Horizon line is tilted {abs(tilt):.1f}° {target_dir}.",
                            "fix": f"Rotate camera ~{abs(tilt):.1f}° {target_dir} to level framing.",
                            "region": [0.0, 0.45, 1.0, 0.55]
                        }],
                        "crop": [0.05, 0.05, 0.95, 0.95],
                        "edits": {"exposure": 0.0, "contrast": 0.05, "temperature": 0},
                        "next_shot": "Align the horizon with the lower third grid line and keep camera level."
                    }
                    
                elif flaw_type == "underexposed":
                    factor = random.uniform(0.40, 0.65)
                    flawed_img = apply_exposure_shift(img, factor)
                    flawed_img.save(out_filepath, quality=90)
                    
                    ev = round((1.0 / factor) - 1.0, 1)
                    tip_text = f"Increase exposure (+{ev} EV) to lift shadow details"
                    live_model_json = {
                        "tip": tip_text,
                        "action": "exposure",
                        "params": {"ev": ev},
                        "priority": "high"
                    }
                    critique_model_json = {
                        "overall": 5.9,
                        "summary": "Framing is deliberate but underexposure crushes shadow nuances.",
                        "scores": {"composition": 7, "light": 5, "color": 6, "focus": 7, "subject": 6},
                        "strengths": ["Balanced compositional thirds", "Good edge-to-edge sharpness"],
                        "issues": [{
                            "aspect": "light",
                            "problem": "Heavy underexposure losing shadow tones.",
                            "fix": f"Lift exposure by +{ev} EV and apply mild tone curve.",
                            "region": [0.15, 0.20, 0.85, 0.80]
                        }],
                        "crop": None,
                        "edits": {"exposure": ev, "contrast": 0.1, "temperature": 0},
                        "next_shot": "Meter for subject faces or tap screen to adjust exposure bias before shooting."
                    }
                    
                elif flaw_type == "overexposed":
                    factor = random.uniform(1.4, 1.8)
                    flawed_img = apply_exposure_shift(img, factor)
                    flawed_img.save(out_filepath, quality=90)
                    
                    tip_text = "Lower exposure by ~1 stop to recover blown highlights"
                    live_model_json = {
                        "tip": tip_text,
                        "action": "exposure",
                        "params": {"ev": -0.7},
                        "priority": "medium"
                    }
                    critique_model_json = {
                        "overall": 6.1,
                        "summary": "Highlight clipping in bright regions washes out sky and skin textures.",
                        "scores": {"composition": 7, "light": 5, "color": 6, "focus": 8, "subject": 6},
                        "strengths": ["Pleasing subject posture", "Sharp foreground focus"],
                        "issues": [{
                            "aspect": "light",
                            "problem": "Blown highlights across upper frame.",
                            "fix": "Reduce exposure bias by -0.7 EV to preserve highlight gradients.",
                            "region": [0.0, 0.0, 1.0, 0.40]
                        }],
                        "crop": None,
                        "edits": {"exposure": -0.6, "contrast": 0.08, "temperature": -100},
                        "next_shot": "Shoot with light source behind you or use HDR capture mode."
                    }
                    
                else: # framing
                    cropped = img.crop((0, 0, w, int(h * 0.82)))
                    cropped.save(out_filepath, quality=90)
                    
                    tip_text = "Step back slightly to avoid clipping the base of your subject"
                    live_model_json = {
                        "tip": tip_text,
                        "action": "step_back",
                        "params": {"dist": 0.5},
                        "priority": "high"
                    }
                    critique_model_json = {
                        "overall": 6.5,
                        "summary": "Composition feels constrained; subject bottom edge is awkwardly clipped.",
                        "scores": {"composition": 6, "light": 7, "color": 7, "focus": 8, "subject": 6},
                        "strengths": ["Crisp focal detail", "Well-balanced color saturation"],
                        "issues": [{
                            "aspect": "composition",
                            "problem": "Subject cutoff at the bottom boundary creates tension.",
                            "fix": "Step back 2-3 feet or lower angle to include complete subject contour.",
                            "region": [0.10, 0.70, 0.90, 1.0]
                        }],
                        "crop": [0.08, 0.04, 0.92, 0.96],
                        "edits": {"exposure": 0.0, "contrast": 0.05, "temperature": 0},
                        "next_shot": "Give breathing room around subject limbs and lower camera slightly."
                    }
                
                # Assemble record according to sample type
                if is_critique_sample:
                    record = {
                        "id": record_id,
                        "image": str(out_filepath.name),
                        "mode": "critique",
                        "conversations": [
                            {
                                "role": "user",
                                "content": "<image>\nCaptured photograph telemetry: AADB Aesthetic Score 6.2/10. "
                                           "Generate full structured photographic critique JSON."
                            },
                            {
                                "role": "model",
                                "content": json.dumps(critique_model_json)
                            }
                        ]
                    }
                    critique_records.append(record)
                else:
                    record = {
                        "id": record_id,
                        "image": str(out_filepath.name),
                        "mode": "live_tip",
                        "conversations": [
                            {
                                "role": "user",
                                "content": "<image>\nViewfinder telemetry: Aesthetic score: 5.8/10. "
                                           "Give one immediate coaching instruction JSON."
                            },
                            {
                                "role": "model",
                                "content": json.dumps(live_model_json)
                            }
                        ]
                    }
                    live_records.append(record)

        except Exception as e:
            continue

    all_records = live_records + critique_records
    output_jsonl = output_dir / "lenscoach_synthetic_dataset.jsonl"
    with open(output_jsonl, "w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    print(f"✅ Generated {len(live_records)} Live Tips and {len(critique_records)} Critiques.")
    print(f"✅ Saved full dataset to {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic photography flaws dataset matching v2 plan")
    parser.add_argument("--src_dir", default="Training/data", help="Path to input images directory")
    parser.add_argument("--out_dir", default="Training/generated_dataset", help="Path to output directory")
    parser.add_argument("--count", type=int, default=300, help="Number of pairs to generate")
    args = parser.parse_args()
    
    generate_synthetic_samples(args.src_dir, args.out_dir, args.count)
