"""Configuracion centralizada del proyecto de fine-tuning."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Rutas del proyecto ---
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "finetuned-model"
GENERATED_DIR = PROJECT_ROOT / "generated"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

# --- HuggingFace ---
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")

# --- Modelo base ---
PRETRAINED_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# --- Dataset ---
DATASET_NAME = "gigant/oldbookillustrations"
IMAGE_COLUMN = "1600px"
CAPTION_COLUMN = "info_alt"
MAX_TRAIN_SAMPLES = None  # None = usar todo el dataset

# --- Hiperparametros ---
RESOLUTION = 512
BATCH_SIZE = 6
NUM_EPOCHS = 2
LEARNING_RATE = 1e-5

# --- Prompt para comparacion antes/despues ---
EVAL_PROMPT = "an illustration of a ship sailing through a stormy sea"
