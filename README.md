# LensCoach 2.0: On-Device Multimodal Photography Coach

LensCoach 2.0 is an on-device, privacy-first AI photography coach designed around **Google Gemma 4 E2B** and the **Apple Neural Engine**. It provides composition guidance directly in your camera viewfinder and produces structured photographic critiques upon capture—with **zero cloud dependencies, zero API keys, and complete data privacy**.

---

## ⚡ Architecture Flow

```mermaid
flowchart TD
    subgraph INPUT["1. Camera & Sensor Inputs"]
        CAM["AVCaptureSession<br/>(Live Video Sample Buffer)"]
        MOT["CoreMotion Sensors<br/>(Device Gravity & Acceleration)"]
        THERMAL["ProcessInfo<br/>(Device Thermal State)"]
    end

    subgraph FAST["2. Fast Perception Engine (Apple Neural Engine & Vision)"]
        FS["CoreML FrameScore<br/>(12-Attribute Aesthetic Scoring)"]
        FACES["Apple Vision Face Detection<br/>(Portrait Headroom & Eye-line Anchoring)"]
        SAL["Apple Vision Saliency<br/>(Dominant Subject Bounding Box)"]
        TILT["Artificial Horizon Calculator<br/>(CoreMotion Roll Tilt with EMA Smoothing)"]
        STEADY{"Stability &<br/>Thermal Gate"}
    end

    subgraph TELEMETRY["3. SceneTelemetry State Bundle"]
        STATE["SceneTelemetry:<br/>• aestheticScore: Float<br/>• attributes: 12-dim Map<br/>• horizonTiltDegrees: Double<br/>• subjectRect: CGRect<br/>• detectedFacesCount: Int<br/>• isCameraSteady: Bool<br/>• thermalState: ThermalState"]
    end

    subgraph COACH["4. Gemma Coach Intelligence Layer (Local GPU)"]
        RUNTIME["CoachRuntime Protocol<br/>(LiteRTRuntime / Gemma 4 E2B Multimodal)"]
        LIVETIP["Live Viewfinder Tip<br/>(Rate-Limited & Stability-Gated)<br/>Schema: tip, action, params, priority"]
        CRITIQUE["Deep Post-Capture Critique<br/>(Full Photo + Telemetry Grounding)<br/>Schema: overall, scores, strengths, issues, crop, edits, next_shot"]
        CI["CoreImageEditService<br/>(CIExposureAdjust, CIColorControls, CITemperatureAndTint)"]
    end

    subgraph UI["5. Native SwiftUI User Experience"]
        HUD["CameraView Viewfinder<br/>• Aerospace Horizon Reticle<br/>• Dynamic Subject Bounding Box<br/>• Rule of Thirds Grid<br/>• Floating Coach Capsule"]
        REVIEW["CritiqueView Modal<br/>• Overall Score Ring<br/>• 5 Sub-Score Dimensions<br/>• Region-Annotated Issues<br/>• Suggested Crop Preview<br/>• CoreImage Before/After Toggle<br/>• Next Shot Recommendation"]
        GALLERY["GalleryView & History<br/>• Local Photo & Critique Storage<br/>• SwiftCharts Aesthetic Trends<br/>• Interactive Review Sheets"]
    end

    %% Wiring connections
    CAM --> FS
    CAM --> FACES
    CAM --> SAL
    MOT --> TILT
    MOT --> STEADY
    THERMAL --> STEADY

    FS --> STATE
    FACES --> STATE
    SAL --> STATE
    TILT --> STATE
    STEADY --> STATE

    STATE -->|"Steady & Cadence Gated"| LIVETIP
    LIVETIP --> HUD
    STATE --> HUD

    CAM -.->|"Shutter Pressed"| CRITIQUE
    STATE -.->|"Telemetry Grounding"| CRITIQUE
    CRITIQUE --> REVIEW
    CRITIQUE --> CI
    CI --> REVIEW
    REVIEW --> GALLERY
```

---

## 🏗️ Core Architectural Loops

### 1. Viewfinder Fast Perception Loop (Apple Neural Engine)
- **`PerceptionEngine.swift`**: Processes live camera frames through the CoreML `FrameScore` model prioritized on the Apple Neural Engine.
- **Aesthetic Attribute Analysis**: Continuously estimates 12 photographic attributes (*Score, Balance, Color, Content, DoF, Light, Blur, Object, Repetition, Rule of Thirds, Symmetry, Vividness*) smoothed via an exponential moving average (EMA) filter.
- **Vision Face & Subject Tracking**: Uses `VNDetectFaceRectanglesRequest` to anchor framing on human subjects for portrait headroom and thirds alignment, falling back to `VNGenerateAttentionBasedSaliencyImageRequest` for scenes without detected faces.
- **Horizon Level Tracking**: Samples `CoreMotion` gravity vectors to compute camera roll tilt angles relative to portrait and landscape orientations.

### 2. Live Coach Guidance Loop (Stability & Cadence Gated)
- **`GemmaCoachEngine.swift`**: Decoupled behind a polymorphic `CoachRuntime` protocol to allow swapping inference engines (LiteRT-LM / MLX Swift) without UI changes.
- **Battery & Thermal Throttling**: Checks `telemetry.isCameraSteady` and `telemetry.isThermalThrottled` (`ProcessInfo.processInfo.thermalState`) to adapt coaching cadence, protecting device thermals and battery life.
- **Structured Action Schema**: Emits actionable instructions formatted as:
  ```json
  {
    "tip": "Tilt clockwise to level the horizon",
    "action": "rotate",
    "params": {"degrees": 4},
    "priority": "high"
  }
  ```
  *(Supported actions: `rotate`, `move`, `step_closer`, `step_back`, `lower_angle`, `raise_angle`, `exposure`, `refocus`, `reframe`, `hold_steady`, `good_to_shoot`)*

### 3. Post-Capture Deep Critique Loop
- Upon photo capture, processes the high-resolution frame alongside the captured scene telemetry to generate a comprehensive critique.
- **Structured Output Schema**:
  ```json
  {
    "overall": 7.2,
    "summary": "Strong visual composition with confident lighting and deliberate focal hierarchy.",
    "scores": {
      "composition": 7,
      "light": 7,
      "color": 8,
      "focus": 8,
      "subject": 7
    },
    "strengths": ["Harmonious weight distribution across visual planes"],
    "issues": [
      {
        "aspect": "composition",
        "problem": "Slight edge clutter drawing eye away from subject.",
        "fix": "Crop peripheral edges and align subject along upper-left third.",
        "region": [0.08, 0.05, 0.92, 0.95]
      }
    ],
    "crop": [0.08, 0.05, 0.92, 0.95],
    "edits": {
      "exposure": 0.35,
      "contrast": 0.12,
      "temperature": -180
    },
    "next_shot": "Move in closer for an eye-level portrait with natural side-window illumination."
  }
  ```
- **Real CoreImage GPU Filter Pipeline**: `CoreImageEditService` converts `edits` into GPU-accelerated `CIExposureAdjust`, `CIColorControls`, and `CITemperatureAndTint` filters for an instant before/after toggle.

### 4. Progress, History & Portfolio Loop
- **`GalleryManager.swift`**: Persists captured photos along with their complete `ShotCritique` metadata and aesthetic scores.
- **`GalleryView.swift`**: Displays aesthetic trends over time using `SwiftCharts` and allows tapping any photo in your portfolio to inspect its original critique and recommended edits.

---

## 🔒 100% Local & Private
- **Zero Cloud APIs**: No OpenAI, Anthropic, or external cloud keys required.
- **Hardware Isolation**: Photos, video frames, and sensor telemetry never leave your device.
- **Offline Capable**: Fully functional without cellular connectivity or Wi-Fi.

---

## 📊 Instrumentation & Diagnostics

Real-world metrics are captured live on-device rather than estimated:
- **`DiagnosticsManager.swift`**: Logs real-time FPS, inference latency, resident memory footprint (`mach_task_basic_info`), device thermal states, and battery drain to a local CSV file during live sessions for profiling on physical hardware.

---

## 🚀 Getting Started

### 1. Provision Model Weights
Download the pre-packaged Gemma 4 E2B LiteRT container:
```bash
./scripts/setup_models.sh
```

### 2. Build and Run in Xcode
1. Open `LensCoachNative/LensCoachNative.xcodeproj` in **Xcode 15+**.
2. Select your connected iPhone (or iOS Simulator).
3. Press **Cmd + R** to run.

---

## 🧠 Training Suite (`Training/`)

- **`generate_synthetic_flaws.py`**: Generates synthetic photography flaws (horizon tilt, misaligned composition, underexposed/blown highlights) with inverse ground-truth fixes for both live tips and critiques.
- **`train_gemma4_lora.py`**: Parameter-efficient QLoRA fine-tuning for Gemma 4 E2B Multimodal with Unsloth.
- **`export_to_litertlm.py`**: Merges LoRA adapters and exports `.litertlm` containers for Google AI Edge on iOS.
- **`train_framescore.py`**: EfficientNet-B0 multi-attribute aesthetic regressor with 80/20 train/val split and Spearman Rank Correlation Coefficient (SRCC) metric evaluation.
- **`LensCoach_Gemma4_Training.ipynb`**: 1-click Google Colab training notebook ready for T4, L4, or A100 GPUs.
