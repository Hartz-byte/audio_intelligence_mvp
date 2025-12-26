import os
from pathlib import Path

# Model configurations
DEMUCS_MODEL = "htdemucs"
WHISPER_MODEL = "openai/whisper-base"
DEVICE = "cuda"  # or "cpu"

# Audio settings
SAMPLE_RATE = 16000
MONO = True
NORMALIZE = True

# Segmentation
CONFIDENCE_THRESHOLD = 0.65
SEGMENT_MIN_DURATION = 0.5  # seconds

# Paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
