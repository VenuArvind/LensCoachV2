#!/bin/bash
set -e

# LensCoach 2.0 - Gemma 4 E2B On-Device Model Setup Script
# Downloads the Google AI Edge LiteRT container for Gemma 4 E2B

RESOURCES_DIR="LensCoachApp/Sources/Resources"
MODEL_FILENAME="gemma-4-E2B-it-gpu.litertlm"
MODEL_URL="https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/${MODEL_FILENAME}"

echo "🎯 Starting LensCoach 2.0 Model Setup (Gemma 4 E2B)..."

mkdir -p "$RESOURCES_DIR"

if [ -f "$RESOURCES_DIR/$MODEL_FILENAME" ]; then
    echo "✅ Gemma 4 E2B model already exists at $RESOURCES_DIR/$MODEL_FILENAME"
else
    echo "⬇️ Downloading Gemma 4 E2B GPU model (~1.9 GB)..."
    echo "Target: $RESOURCES_DIR/$MODEL_FILENAME"
    curl -L "$MODEL_URL" -o "$RESOURCES_DIR/$MODEL_FILENAME"
    echo "✅ Download complete."
fi

echo "✨ Ready for on-device compilation and execution!"
