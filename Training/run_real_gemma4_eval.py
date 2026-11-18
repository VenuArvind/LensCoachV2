#!/usr/bin/env python3
"""
Real multimodal evaluation of Base Gemma 4 E2B (google/gemma-4-E2B-it)
vs. Fine-Tuned LensCoach QLoRA (VenuArvind/lenscoach-gemma4-e2b)
on held-out synthetic photographic flaws + real AADB test photos.
"""

import csv
import gc
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageEnhance
from transformers import AutoModelForMultimodalLM, AutoProcessor


REQUIRED_KEYS = {"tip", "action", "params", "priority"}


def parse_json_response(text: str) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    """
    Returns (strict_json_valid, any_json_valid, parsed_dict).
    strict_json_valid: raw text parses directly as JSON with REQUIRED_KEYS.
    any_json_valid: JSON object extracted from markdown/preamble parses with REQUIRED_KEYS.
    """
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict) and REQUIRED_KEYS.issubset(obj.keys()):
            return True, True, obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and REQUIRED_KEYS.issubset(obj.keys()):
                return False, True, obj
        except Exception:
            pass
    return False, False, None


def build_held_out_dataset(data_dir: Path, n_synth: int = 20, n_real: int = 10, seed: int = 2026) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    img_dir = data_dir / "aadb-imagedatabase" / "datasetImages_warp256 2"
    test_csv = data_dir / "aadb-imagedatabase" / "Dataset_test.csv"

    test_files: List[Tuple[Path, float]] = []
    if test_csv.exists():
        with open(test_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("ImageFile") or row.get("filename") or row.get("image") or list(row.values())[0]
                p = img_dir / fname
                if p.exists():
                    score = 5.8
                    for k in ("score", "Score", "aesthetic_score"):
                        if k in row and row[k]:
                            try:
                                score = round(float(row[k]) * 10.0, 1) if float(row[k]) <= 1.0 else round(float(row[k]), 1)
                            except ValueError:
                                pass
                    test_files.append((p, score))
    if not test_files:
        for p in sorted(img_dir.glob("*.jpg"))[-100:]:
            test_files.append((p, 5.8))

    rng.shuffle(test_files)
    cases: List[Dict[str, Any]] = []
    flaw_types = ["tilt", "underexposed", "overexposed", "framing"]

    # 1. Held-out synthetic flaws (5 per flaw type)
    for i in range(n_synth):
        src_path, _ = test_files[i % len(test_files)]
        ft = flaw_types[i % len(flaw_types)]
        with Image.open(src_path) as raw_img:
            img = raw_img.convert("RGB")
            w, h = img.size
            if ft == "tilt":
                angle = round(rng.choice([-1, 1]) * rng.uniform(3.0, 8.0), 1)
                rotated = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
                cw, ch = int(w * 0.90), int(h * 0.90)
                flawed = rotated.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))
                gt_action, gt_deg = "rotate", abs(angle)
            elif ft == "underexposed":
                flawed = ImageEnhance.Brightness(img).enhance(rng.uniform(0.35, 0.55))
                gt_action, gt_deg = "exposure", None
            elif ft == "overexposed":
                flawed = ImageEnhance.Brightness(img).enhance(rng.uniform(1.55, 1.85))
                gt_action, gt_deg = "exposure", None
            else:
                # Tight crop simulating clipped subject
                flawed = img.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88))).resize((w, h))
                gt_action, gt_deg = "step_back", None

        cases.append({
            "id": f"synth_test_{i:02d}",
            "split": "synthetic_flaw",
            "flaw_type": ft,
            "source_file": src_path.name,
            "image": flawed,
            "prompt_text": "Viewfinder telemetry: Aesthetic score: 5.8/10. Respond with raw JSON only (no markdown) with exact keys: tip, action (one of rotate, exposure, step_back, reframe, good_to_shoot), params (dict), priority (high/medium/low).",
            "gt_action": gt_action,
            "gt_degrees": gt_deg,
        })

    # 2. Real unperturbed AADB test photos
    for j in range(n_real):
        src_path, score = test_files[(n_synth + j) % len(test_files)]
        with Image.open(src_path) as raw_img:
            real_img = raw_img.convert("RGB")
        cases.append({
            "id": f"real_aadb_{j:02d}",
            "split": "real_aadb",
            "flaw_type": "real_photo",
            "source_file": src_path.name,
            "image": real_img,
            "prompt_text": f"Viewfinder telemetry: Aesthetic score: {score:.1f}/10. Respond with raw JSON only (no markdown) with exact keys: tip, action (one of rotate, exposure, step_back, reframe, good_to_shoot), params (dict), priority (high/medium/low).",
            "gt_action": None,
            "gt_degrees": None,
        })

    return cases


def evaluate_model(model_id: str, cases: List[Dict[str, Any]], device: str) -> Dict[str, Any]:
    print(f"\n=== Evaluating {model_id} on {device} ({len(cases)} images) ===")
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForMultimodalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    records = []
    latencies_ms = []
    strict_valid_synth = 0
    any_valid_synth = 0
    strict_valid_real = 0
    any_valid_real = 0
    action_correct = 0
    tilt_errors = []

    for idx, case in enumerate(cases):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": case["image"]},
                    {"type": "text", "text": case["prompt_text"]},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        input_len = inputs["input_ids"].shape[-1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        if device == "mps":
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        gen_text = processor.decode(out_ids[0][input_len:], skip_special_tokens=True)
        strict_ok, any_ok, parsed = parse_json_response(gen_text)

        if case["split"] == "synthetic_flaw":
            strict_valid_synth += int(strict_ok)
            any_valid_synth += int(any_ok)
            if parsed and parsed.get("action") == case["gt_action"]:
                action_correct += 1
            if case["flaw_type"] == "tilt" and parsed and isinstance(parsed.get("params"), dict):
                deg = parsed["params"].get("degrees")
                if isinstance(deg, (int, float)) and case["gt_degrees"] is not None:
                    tilt_errors.append(abs(float(deg) - case["gt_degrees"]))
        else:
            strict_valid_real += int(strict_ok)
            any_valid_real += int(any_ok)

        records.append({
            "id": case["id"],
            "split": case["split"],
            "flaw_type": case["flaw_type"],
            "source_file": case["source_file"],
            "prompt_text": case["prompt_text"],
            "raw_output": gen_text,
            "strict_json_valid": strict_ok,
            "any_json_valid": any_ok,
            "parsed": parsed,
            "latency_ms": round(elapsed_ms, 1),
        })
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(cases)}...")

    n_synth = sum(1 for c in cases if c["split"] == "synthetic_flaw")
    n_real = sum(1 for c in cases if c["split"] == "real_aadb")
    lat_sorted = sorted(latencies_ms)
    summary = {
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
        "n_synthetic": n_synth,
        "n_real_aadb": n_real,
        "synth_strict_json_pct": round(100.0 * strict_valid_synth / n_synth, 1),
        "synth_any_json_pct": round(100.0 * any_valid_synth / n_synth, 1),
        "synth_action_accuracy_pct": round(100.0 * action_correct / n_synth, 1),
        "tilt_mae_deg": round(sum(tilt_errors) / len(tilt_errors), 2) if tilt_errors else None,
        "tilt_parsed_count": len(tilt_errors),
        "real_strict_json_pct": round(100.0 * strict_valid_real / n_real, 1),
        "real_any_json_pct": round(100.0 * any_valid_real / n_real, 1),
        "latency_p50_ms": round(lat_sorted[len(lat_sorted) // 2], 1),
        "latency_p95_ms": round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 1),
    }
    print("Summary:", json.dumps(summary, indent=2))

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return {"summary": summary, "records": records}


def main():
    base_dir = Path(__file__).resolve().parent
    cases = build_held_out_dataset(base_dir / "data", n_synth=20, n_real=10, seed=2026)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    base_res = evaluate_model("google/gemma-4-E2B-it", cases, device)
    ft_res = evaluate_model("VenuArvind/lenscoach-gemma4-e2b", cases, device)

    out_path = base_dir / "real_gemma4_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base": base_res, "fine_tuned": ft_res}, f, indent=2)
    print(f"\nSaved full real evaluation results to {out_path}")


if __name__ == "__main__":
    main()
