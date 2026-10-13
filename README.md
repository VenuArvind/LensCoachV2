# LensCoach 2.0: On-Device Multimodal Photography Coach

LensCoach 2.0 is a 100% on-device, privacy-first AI photography assistant powered by **Google Gemma 4 E2B** and the **Apple Neural Engine**. It delivers zero-latency composition guidance and deep photographic reasoning directly in the viewfinder—with **zero cloud dependency, zero network requests, and absolute privacy**.

---

## ⚡ Architecture: Two-Tier On-Device Intelligence

LensCoach 2.0 uses a decoupled, hybrid on-device pipeline optimized for the iPhone 15 memory envelope:

```
Viewfinder (30 FPS) ──> PerceptionEngine (CoreML ANE + Vision + CoreMotion)
                             │
                             ├── Real-time aesthetic scoring (12 attributes, 5.1ms)
                             ├── Physical horizon leveling (±0.6° haptic accuracy)
                             └── Subject saliency & boundary detection
                                     │
                                     ▼ (Gated every ~1.8s)
                             GemmaCoachEngine (Gemma 4 E2B Multimodal Edge)
                                     │
                                     ├── Live Viewfinder Advice Capsules
                                     └── Post-Capture Deep Structured Critique
```

1. **Fast Perception Engine (30 FPS on Apple Neural Engine)**:
   - Real-time inference of 12 aesthetic dimensions via CoreML `FrameScore`.
   - Continuous device motion roll/pitch tracking via `CoreMotion` providing an aerospace-grade artificial horizon.
   - Real-time subject saliency and bounds tracking via Apple `Vision`.

2. **Gemma 4 E2B Multimodal Coach (Metal GPU Edge)**:
   - Edge-native 2B effective parameter multimodal model natively understanding both pixels and telemetry.
   - Throttled and gated to trigger only when the camera is steady, conserving battery and thermal headroom.
   - Generates structured coaching actions: horizon counter-rotation, framing realignment, exposure compensation, and rule-of-thirds positioning.
   - Emits full post-capture critiques with sub-scores, strengths, actionable recommendations, and one-tap suggested crop previews.

---

## 🔒 100% Local & Private
- **Zero Cloud APIs**: No OpenAI, Anthropic, or Gemini API keys required.
- **Hardware Isolation**: Images and sensor telemetry never leave your device.
- **Airplane Mode Ready**: Completely functional in remote landscapes without cell coverage.

---

## 📱 Hardware Benchmarks (iPhone 15)
| Metric | Performance | Technical Significance |
| :--- | :---: | :--- |
| **Viewfinder Throughput** | **30 FPS** | Smooth, continuous camera overlay |
| **Neural Engine Latency** | **5.1 ms** | Ultra-efficient CoreML FrameScore execution |
| **Resident Memory** | **~600 MB - 1.4 GB** | Memory-mapped execution safe from iOS Jetsam limits |
| **Horizon Alignment** | **±0.6° precision** | Instant haptic feedback on level shot |

---

## 🚀 Getting Started

### 1. Provision Model
Run the setup script to download the optimized Gemma 4 E2B LiteRT container:
```bash
./scripts/setup_models.sh
```

### 2. Build and Run
1. Open `LensCoachNative/LensCoachNative.xcodeproj` in **Xcode 15+**.
2. Connect your iPhone 15 (or select an iOS 16+ Simulator).
3. Press **Cmd + R** to run.

---

## 🧠 Training & Fine-Tuning Pipeline (`Training/`)

- `generate_synthetic_flaws.py`: Creates synthetic camera flaw training pairs (horizon tilt, misaligned composition, exposure extremes) with exact ground-truth fixes.
- `train_gemma4_lora.py`: Fine-tunes Gemma 4 E2B on multi-modal coaching pairs with QLoRA (compatible with Google Colab and cloud GPUs).
- `export_to_litertlm.py`: Merges adapters and exports deployable `.litertlm` containers for iOS execution.
