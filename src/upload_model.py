"""
Subida del modelo fine-tuneado a HuggingFace Hub.

Uso:
  python src/upload_model.py

Requiere .env con HF_TOKEN y HF_REPO_ID configurados.
"""

from huggingface_hub import HfApi, create_repo
from diffusers import UNet2DConditionModel
from pathlib import Path

from src.config import HF_TOKEN, HF_REPO_ID, OUTPUT_DIR, PROJECT_ROOT


def main():
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN no encontrado. Crea un archivo .env con tu token de HuggingFace."
        )
    if not HF_REPO_ID:
        raise ValueError(
            "HF_REPO_ID no encontrado. Crea un archivo .env con tu repo_id (usuario/nombre-repo)."
        )

    model_path = OUTPUT_DIR
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo en {model_path}. Ejecuta el entrenamiento primero."
        )

    print(f"Subiendo modelo desde {model_path} a {HF_REPO_ID}...")

    # Crear repositorio en HuggingFace
    api = HfApi(token=HF_TOKEN)
    create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, exist_ok=True)
    print(f"Repositorio creado/verificado: {HF_REPO_ID}")

    # Subir carpeta del modelo
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=HF_REPO_ID,
        commit_message="Upload fine-tuned UNet for old book illustrations style",
    )
    print(f"Modelo subido a https://huggingface.co/{HF_REPO_ID}")

    # Verificar que se puede cargar desde HuggingFace
    print("\nVerificando carga desde HuggingFace...")
    unet = UNet2DConditionModel.from_pretrained(HF_REPO_ID)
    print(f"Modelo cargado correctamente desde {HF_REPO_ID}")

    # Guardar link en entrega
    entrega_dir = PROJECT_ROOT / "entrega"
    entrega_dir.mkdir(parents=True, exist_ok=True)
    link_file = entrega_dir / "huggingface_link.txt"
    link_file.write_text(f"https://huggingface.co/{HF_REPO_ID}\n")
    print(f"Link guardado en {link_file}")

    print("\nSubida completada.")


if __name__ == "__main__":
    main()
