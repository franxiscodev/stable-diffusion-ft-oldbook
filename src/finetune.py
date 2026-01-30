"""
Fine-tuning de Stable Diffusion v1-4 con Old Book Illustrations.

Sigue el patron del notebook del docente (2.finetuning_stable_diffusion.ipynb)
con 3 adaptaciones:
  1. Columna de imagen: "1600px" (en vez de "image")
  2. Columna de caption: "info_alt" (en vez de "text")
  3. Transforms: Resize + CenterCrop (en vez de Resize((512,512)))

Uso:
  python -m src.finetune
"""

# --- Imports ---
from src.config import (
    PRETRAINED_MODEL_NAME,
    DATASET_NAME,
    IMAGE_COLUMN,
    CAPTION_COLUMN,
    MAX_TRAIN_SAMPLES,
    RESOLUTION,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    GENERATED_DIR,
    EVAL_PROMPT,
    PROJECT_ROOT,
)
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from datasets import load_dataset
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
import os

import time
from datetime import datetime, timedelta

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Log file ---
LOG_DIR = PROJECT_ROOT / "outputs"
LOG_FILE = LOG_DIR / "training_log.txt"


def log(msg, log_file=None):
    """Print y escribe a fichero de log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def format_eta(seconds):
    """Formatea segundos a string legible (ej: 1h 42m)."""
    if seconds < 0:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {int(seconds % 60):02d}s"


def main():
    # --- Device setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Preparar log ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOG_FILE, "w", encoding="utf-8")

    log(f"Device: {device}", log_file)
    log(f"Config: RESOLUTION={RESOLUTION}, BATCH_SIZE={BATCH_SIZE}, "
        f"MAX_TRAIN_SAMPLES={MAX_TRAIN_SAMPLES}, "
        f"NUM_EPOCHS={NUM_EPOCHS}, LR={LEARNING_RATE}", log_file)

    # --- Generar imagen ANTES del fine-tuning ---
    log("--- Generando imagen ANTES del fine-tuning ---", log_file)
    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_NAME,
    ).to(device)

    image_before = pipe(
        EVAL_PROMPT, height=RESOLUTION, width=RESOLUTION
    ).images[0]
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    image_before.save(GENERATED_DIR / "before_finetuning.png")
    log(f"Imagen guardada en {GENERATED_DIR / 'before_finetuning.png'}", log_file)

    # Liberar memoria
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Cargar dataset ---
    log("--- Cargando dataset ---", log_file)
    dataset = load_dataset(DATASET_NAME, split="train")
    if MAX_TRAIN_SAMPLES:
        dataset = dataset.select(range(MAX_TRAIN_SAMPLES))
    log(f"Muestras de entrenamiento: {len(dataset)}", log_file)

    # Comprobar tamano de imagen
    size = dataset[0][IMAGE_COLUMN].size
    log(f"Tamano de las imagenes del dataset: {size}", log_file)

    # --- Definir transforms ---
    # Resize + CenterCrop en vez de Resize((R,R))
    # porque las imagenes del dataset no son cuadradas
    image_transforms = transforms.Compose([
        transforms.Resize(RESOLUTION),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # --- Cargar componentes individuales ---
    log("--- Cargando componentes del modelo ---", log_file)

    # Tokenizador
    tokenizer = CLIPTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME, subfolder="tokenizer"
    )

    # Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        PRETRAINED_MODEL_NAME, subfolder="scheduler"
    )

    # Text Encoder (CLIP)
    text_encoder = CLIPTextModel.from_pretrained(
        PRETRAINED_MODEL_NAME, subfolder="text_encoder"
    ).to(device)

    # VAE: Autoencoder
    vae = AutoencoderKL.from_pretrained(
        PRETRAINED_MODEL_NAME, subfolder="vae"
    ).to(device)

    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODEL_NAME, subfolder="unet"
    ).to(device)

    # --- Dataset class ---
    class Text2ImageDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            example = self.dataset[idx]
            # CAMBIO 1: "1600px" en vez de "image"
            image = image_transforms(example[IMAGE_COLUMN].convert("RGB"))
            # CAMBIO 2: "info_alt" en vez de "text"
            token = tokenizer(
                example[CAPTION_COLUMN],
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            return {
                "pixel_values": image,
                "input_ids": token.input_ids.squeeze(0),
                "attention_mask": token.attention_mask.squeeze(0),
            }

    train_dataset = Text2ImageDataset(dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # --- Congelar VAE y text_encoder ---
    vae.eval()
    text_encoder.eval()

    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False

    # --- Optimizer y Accelerator ---
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

    accelerator = Accelerator()
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    log(f"Accelerator device: {accelerator.device}", log_file)

    # --- Training loop ---
    batches_per_epoch = len(train_dataloader)
    total_batches = NUM_EPOCHS * batches_per_epoch
    log(f"--- Iniciando entrenamiento: {NUM_EPOCHS} epochs, "
        f"{batches_per_epoch} batches/epoch, {total_batches} batches total ---",
        log_file)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    global_step = 0
    start_time = time.time()
    last_checkpoint_pct = 0  # ultimo % en que se guardo checkpoint

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        epoch_losses = []
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        for batch in progress_bar:
            batch_start = time.time()

            # Pasar pixeles al espacio latente con el encoder del VAE
            with torch.no_grad():
                latents = vae.encode(
                    batch["pixel_values"].to(accelerator.device)
                ).latent_dist.sample()
                latents = latents * 0.18215

            # Proceso de difusion hacia delante
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(
                latents, noise, timesteps)

            # Codificar texto
            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(accelerator.device)
            )[0]

            # Prediccion de ruido
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

            # Loss y backpropagation
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            global_step += 1
            batch_time = time.time() - batch_start

            # Progreso global
            pct = global_step / total_batches * 100
            elapsed = time.time() - start_time
            avg_time_per_batch = elapsed / global_step
            eta_seconds = avg_time_per_batch * (total_batches - global_step)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                pct=f"{pct:.0f}%",
                eta=format_eta(eta_seconds),
            )

            # Log detallado cada 10 batches
            if global_step % 10 == 0 or global_step == 1:
                log(f"[Batch {global_step}/{total_batches}] "
                    f"[Global {pct:.0f}%] "
                    f"Loss: {loss.item():.4f} | "
                    f"{batch_time:.1f}s/batch | "
                    f"ETA: {format_eta(eta_seconds)}", log_file)

            # Checkpoint cada 10% de progreso
            current_pct_bucket = int(pct // 10) * 10
            if current_pct_bucket > last_checkpoint_pct and current_pct_bucket > 0:
                last_checkpoint_pct = current_pct_bucket
                ckpt_path = CHECKPOINT_DIR / f"checkpoint-pct-{current_pct_bucket}"
                unet.save_pretrained(ckpt_path)
                log(f"--- Checkpoint {current_pct_bucket}% guardado en "
                    f"{ckpt_path} ---", log_file)

        # Fin de epoch
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        log(f"Epoch {epoch + 1}/{NUM_EPOCHS} completado - "
            f"Loss promedio: {avg_loss:.6f} - "
            f"Tiempo: {format_eta(epoch_time)}", log_file)

        # Checkpoint al final de cada epoch
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint-epoch-{epoch + 1}"
        unet.save_pretrained(checkpoint_path)
        log(f"Checkpoint epoch guardado en {checkpoint_path}", log_file)

    # --- Guardar modelo final ---
    total_time = time.time() - start_time
    log(f"--- Guardando modelo final (tiempo total: {format_eta(total_time)}) ---",
        log_file)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log(f"Modelo guardado en {OUTPUT_DIR}", log_file)

    log("Entrenamiento completado.", log_file)
    log_file.close()


if __name__ == "__main__":
    main()
