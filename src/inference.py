"""
Inferencia con el modelo fine-tuneado de Stable Diffusion.

Genera imagenes antes y despues del fine-tuning y crea una comparacion side-by-side.

Uso:
  python src/inference.py
  python src/inference.py --prompt "a castle on a hill"
  python src/inference.py --after-only  # solo genera con el modelo fine-tuneado
"""

import argparse
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from PIL import Image
import matplotlib.pyplot as plt
import torch

from src.config import (
    PRETRAINED_MODEL_NAME,
    OUTPUT_DIR,
    GENERATED_DIR,
    EVAL_PROMPT,
)


def generate_before(prompt, device):
    """Genera imagen con el modelo base (antes del fine-tuning)."""
    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_NAME,
    ).to(device)

    image = pipe(prompt).images[0]

    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    return image


def generate_after(prompt, device):
    """Genera imagen con el modelo fine-tuneado."""
    finetuned_unet = UNet2DConditionModel.from_pretrained(str(OUTPUT_DIR))
    finetuned_unet.to(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        PRETRAINED_MODEL_NAME,
        unet=finetuned_unet,
    ).to(device)

    image = pipe(prompt).images[0]

    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()

    return image


def create_comparison(image_before, image_after, prompt, save_path):
    """Crea y guarda una comparacion side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image_before)
    axes[0].set_title("ANTES del fine-tuning", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(image_after)
    axes[1].set_title("DESPUES del fine-tuning", fontsize=14)
    axes[1].axis("off")

    plt.suptitle(f'Prompt: "{prompt}"', fontsize=12, style="italic")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparacion guardada en {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Inferencia Stable Diffusion")
    parser.add_argument("--prompt", type=str, default=EVAL_PROMPT)
    parser.add_argument(
        "--after-only",
        action="store_true",
        help="Solo genera con el modelo fine-tuneado",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    if args.after_only:
        print(f"\nGenerando imagen DESPUES del fine-tuning...")
        image_after = generate_after(args.prompt, device)
        image_after.save(GENERATED_DIR / "after_finetuning.png")
        print(f"Guardada en {GENERATED_DIR / 'after_finetuning.png'}")
    else:
        # Generar antes
        print(f"\nGenerando imagen ANTES del fine-tuning...")
        image_before = generate_before(args.prompt, device)
        image_before.save(GENERATED_DIR / "before_finetuning.png")
        print(f"Guardada en {GENERATED_DIR / 'before_finetuning.png'}")

        # Generar despues
        print(f"\nGenerando imagen DESPUES del fine-tuning...")
        image_after = generate_after(args.prompt, device)
        image_after.save(GENERATED_DIR / "after_finetuning.png")
        print(f"Guardada en {GENERATED_DIR / 'after_finetuning.png'}")

        # Comparacion
        create_comparison(
            image_before,
            image_after,
            args.prompt,
            GENERATED_DIR / "comparison.png",
        )

    print("\nInferencia completada.")


if __name__ == "__main__":
    main()
