"""
Fine-tuning de Stable Diffusion v1-4 con Old Book Illustrations.

Sigue el patron del notebook del docente (2.finetuning_stable_diffusion.ipynb)
con 3 adaptaciones:
  1. Columna de imagen: "1600px" (en vez de "image")
  2. Columna de caption: "info_alt" (en vez de "text")
  3. Transforms: Resize(512) + CenterCrop(512) (en vez de Resize(512,512))

Uso:
  python src/finetune.py
"""

# --- Imports ---
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

ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm

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
)


def main():
    # --- Device setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Generar imagen ANTES del fine-tuning ---
    print("\n--- Generando imagen ANTES del fine-tuning ---")
    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_NAME,
    ).to(device)

    image_before = pipe(EVAL_PROMPT).images[0]
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    image_before.save(GENERATED_DIR / "before_finetuning.png")
    print(f"Imagen guardada en {GENERATED_DIR / 'before_finetuning.png'}")

    # Liberar memoria
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Cargar dataset ---
    print("\n--- Cargando dataset ---")
    dataset = load_dataset(DATASET_NAME, split="train")
    if MAX_TRAIN_SAMPLES:
        dataset = dataset.select(range(MAX_TRAIN_SAMPLES))
    print(f"Muestras de entrenamiento: {len(dataset)}")

    # Comprobar tamano de imagen
    size = dataset[0][IMAGE_COLUMN].size
    print(f"Tamano de las imagenes del dataset: {size}")

    # --- Definir transforms ---
    # CAMBIO 3: Resize(512) + CenterCrop(512) en vez de Resize((512,512))
    # porque las imagenes del dataset no son cuadradas
    image_transforms = transforms.Compose([
        transforms.Resize(RESOLUTION),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # --- Cargar componentes individuales ---
    print("\n--- Cargando componentes del modelo ---")

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
    print(f"Accelerator device: {accelerator.device}")

    # --- Training loop ---
    print(f"\n--- Iniciando entrenamiento: {NUM_EPOCHS} epochs ---")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        for batch in progress_bar:
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
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

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
            progress_bar.set_postfix(loss=loss.item())

        # Logging de loss promedio por epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss promedio: {avg_loss:.6f}")

        # Checkpoint cada epoch
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint-epoch-{epoch + 1}"
        unet.save_pretrained(checkpoint_path)
        print(f"Checkpoint guardado en {checkpoint_path}")

    # --- Guardar modelo final ---
    print(f"\n--- Guardando modelo final ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Modelo guardado en {OUTPUT_DIR}")

    print("\nEntrenamiento completado.")


if __name__ == "__main__":
    main()
