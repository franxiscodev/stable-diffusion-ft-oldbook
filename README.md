# Stable Diffusion - Old Book Illustrations Fine-tuning

Fine-tuning de **Stable Diffusion v1-4** (`CompVis/stable-diffusion-v1-4`) con el dataset
[gigant/oldbookillustrations](https://huggingface.co/datasets/gigant/oldbookillustrations)
para generar imagenes con estilo de ilustraciones de libros antiguos.

## Estructura del proyecto

```
.
├── src/
│   ├── config.py           # Configuracion centralizada
│   ├── finetune.py         # Script de entrenamiento
│   ├── inference.py        # Script de inferencia
│   └── upload_model.py     # Subida a HuggingFace
├── notebooks/
│   ├── 01_dataset_exploration.ipynb    # Exploracion del dataset
│   ├── 02_finetune_stable_diffusion.ipynb  # Fine-tuning (entregable)
│   └── 03_inference_comparison.ipynb   # Comparacion antes/despues
├── outputs/
│   ├── finetuned-model/    # Modelo fine-tuneado (UNet + tokenizer)
│   └── checkpoints/        # Checkpoints por epoch
├── generated/              # Imagenes generadas
├── entrega/                # Entregables del proyecto
├── requirements.txt
└── .env.example
```

## Setup

```bash
# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# Instalar dependencias
uv pip install -r requirements.txt

# Configurar credenciales
cp .env.example .env
# Editar .env con tu HF_TOKEN y HF_REPO_ID
```

## Uso

### 1. Explorar el dataset
```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### 2. Entrenar el modelo
```bash
# Como script (recomendado para ejecucion larga):
python -m src.finetune

# O como notebook:
jupyter notebook notebooks/02_finetune_stable_diffusion.ipynb
```

### 3. Generar imagenes
```bash
python -m src.inference
python -m src.inference --prompt "a castle on a hill"
python -m src.inference --after-only
```

### 4. Subir modelo a HuggingFace
```bash
python -m src.upload_model
```

## Adaptaciones respecto al notebook del docente

El codigo sigue el patron exacto de `2.finetuning_stable_diffusion.ipynb` con 3 cambios:

| # | Docente (Pokemon) | Este proyecto (Old Book) | Razon |
|---|---|---|---|
| 1 | `example["image"]` | `example["1600px"]` | Columna de imagen del dataset |
| 2 | `example["text"]` | `example["info_alt"]` | Columna de caption del dataset |
| 3 | `Resize((512,512))` | `Resize(512) + CenterCrop(512)` | Imagenes no cuadradas |

## Configuracion

| Parametro | Valor |
|---|---|
| Modelo base | CompVis/stable-diffusion-v1-4 |
| Dataset | gigant/oldbookillustrations |
| Device | CPU (16GB RAM) |
| Batch size | 6 |
| Epochs | 2 |
| Learning rate | 1e-5 |
| Resolucion | 512x512 |
| Optimizer | AdamW |

## Entregables

1. `notebooks/02_finetune_stable_diffusion.ipynb` - Notebook de fine-tuning
2. `entrega/huggingface_link.txt` - Link al modelo en HuggingFace Hub
3. `generated/before_finetuning.png` - Imagen antes del fine-tuning
4. `generated/after_finetuning.png` - Imagen despues del fine-tuning
5. `generated/comparison.png` - Comparativa side-by-side
