#!/usr/bin/env python3
"""
LensCoach 2.0 - Synthetic Photography Flaw Generator
Generates realistic photography flaws (horizon tilt, framing offsets, exposure errors)
from high-aesthetic images, creating paired (flawed_image, ground_truth_telemetry, target_fix_json).
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
        print(f"No source images found in {input_images_dir}. Creating generator template.")
        return

    print(f"Found {len(input_files)} source images. Generating {count} synthetic coaching pairs...")
    dataset_records = []

    for i in range(count):
        src_path = random.choice(input_files)
        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                
                # Pick a synthetic flaw scenario
                flaw_type = random.choice(["tilt", "underexposed", "overexposed", "framing"])
                record_id = f"synth_{i:05d}"
                out_filename = f"{record_id}.jpg"
                out_filepath = images_out / out_filename
                
                if flaw_type == "tilt":
                    # Tilt between -8° and +8° (excluding near-zero)
                    tilt = random.choice([-1, 1]) * random.uniform(2.0, 7.5)
                    flawed_img = apply_horizon_tilt(img, tilt)
                    flawed_img.save(out_filepath, quality=90)
                    
                    target_fix_direction = "clockwise" if tilt > 0 else "counter-clockwise"
                    target_action = "level_horizon"
                    tip_text = f"Tilt camera {abs(tilt):.1f}° {target_fix_direction} to level the horizon line"
                    
                    record = {
                        "id": record_id,
                        "image": str(out_filepath.name),
                        "telemetry": {
                            "horizon_tilt_deg": round(tilt, 2),
                            "aesthetic_score": 0.58,
                            "primary_subject_box": [0.35, 0.30, 0.65, 0.75]
                        },
                        "conversations": [
                            {
                                "role": "user",
                                "content": "<image>\nViewfinder telemetry: Horizon tilt is "
                                           f"{tilt:+.1f}°. Aesthetic score: 5.8/10. "
                                           "Give one immediate coaching instruction."
                            },
                            {
                                "role": "model",
                                "content": json.dumps({
                                    "tip": tip_text,
                                    "action": target_action,
                                    "degrees": round(abs(tilt), 1),
                                    "priority": "high"
                                })
                            }
                        ]
                    }
                    dataset_records.append(record)
                    
                elif flaw_type == "underexposed":
                    factor = random.uniform(0.40, 0.65)
                    flawed_img = apply_exposure_shift(img, factor)
                    flawed_img.save(out_filepath, quality=90)
                    
                    ev_needed = round(1.0 / factor, 1)
                    tip_text = f"Increase exposure (+{ev_needed} EV) to lift shadow details on the subject"
                    
                    record = {
                        "id": record_id,
                        "image": str(out_filepath.name),
                        "telemetry": {
                            "horizon_tilt_deg": 0.1,
                            "aesthetic_score": 0.45,
                            "weakest_attribute": "Light"
                        },
                        "conversations": [
                            {
                                "role": "user",
                                "content": "<image>\nViewfinder telemetry: Horizon is level. "
                                           "Lighting score is low (4.5/10). Give one coaching instruction."
                            },
                            {
                                "role": "model",
                                "content": json.dumps({
                                    "tip": tip_text,
                                    "action": "increase_light",
                                    "priority": "medium"
                                })
                            }
                        ]
                    }
                    dataset_records.append(record)
                    
                elif flaw_type == "overexposed":
                    factor = random.uniform(1.4, 1.8)
                    flawed_img = apply_exposure_shift(img, factor)
                    flawed_img.save(out_filepath, quality=90)
                    
                    tip_text = "Lower exposure by ~1 stop to recover blown highlight details"
                    record = {
                        "id": record_id,
                        "image": str(out_filepath.name),
                        "telemetry": {
                            "horizon_tilt_deg": 0.0,
                            "aesthetic_score": 0.50,
                            "weakest_attribute": "Light"
                        },
                        "conversations": [
                            {
                                "role": "user",
                                "content": "<image>\nViewfinder telemetry: Highlights are clipping. Give one coaching instruction."
                            },
                            {
                                "role": "model",
                                "content": json.dumps({
                                    "tip": tip_text,
                                    "action": "soften_light",
                                    "priority": "medium"
                                })
                            }
                        ]
                    }
                    dataset_records.append(record)
                    
                else: # framing
                    w, h = img.size
                    # Awkward tight crop cutting bottom 20%
                    cropped = img.crop((0, 0, w, int(h * 0.80)))
                    cropped.save(out_filepath, quality=90)
                    
                    tip_text = "Step back slightly to avoid clipping the base of your subject"
                    record = {
                        "id": record_id,
                        "image": str(out_filepath.name),
                        "telemetry": {
                            "horizon_tilt_deg": 0.0,
                            "aesthetic_score": 0.52,
                            "weakest_attribute": "Balance"
                        },
                        "conversations": [
                            {
                                "role": "user",
                                "content": "<image>\nViewfinder telemetry: Subject boundary is clipped near the bottom frame. Give one coaching instruction."
                            },
                            {
                                "role": "model",
                                "content": json.dumps({
                                    "tip": tip_text,
                                    "action": "step_back",
                                    "priority": "high"
                                })
                            }
                        ]
                    }
                    dataset_records.append(record)

        except Exception as e:
            continue

    # Write out JSONL dataset
    output_jsonl = output_dir / "lenscoach_synthetic_flaws.jsonl"
    with open(output_jsonl, "w") as f:
        for r in dataset_records:
            f.write(json.dumps(r) + "\n")

    print(f"✅ Successfully wrote {len(dataset_records)} coaching samples to {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic photography flaws dataset")
    parser.add_argument("--src_dir", default="Training/data", help="Path to input images directory")
    parser.add_argument("--out_dir", default="Training/generated_dataset", help="Path to output directory")
    parser.add_argument("--count", type=int, default=300, help="Number of pairs to generate")
    args = parser.parse_args()
    
    generate_synthetic_samples(args.src_dir, args.out_dir, args.count)
