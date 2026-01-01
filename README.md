# EmpiricalCircumplexModel

Note: This repository contains experimental code for a pre-publication research paper. The license, datasets, and code are subject to change without prior notice.

This repository provides the experimental implementation for:

Are Emotions Arranged in a Circle? Geometric Analysis of Emotion Representations via Hyperspherical Contrastive Learning

The code trains a model that embeds emotion representations into a circular structure.

## Requirements

- Python 3.12 or 3.13
- CUDA-capable GPU (for flash-attn)
- See `pyproject.toml` for full dependencies

## Installation and Training

### Training Procedure

```bash
# 1. Install basic dependencies (including torch)
uv sync

# 2. Build and install flash-attn using the installed torch (without build isolation)
source .venv/bin/activate
uv pip install flash-attn>=2.8.3 --no-build-isolation

# 3. Run the training script
cd utils
bash train_emolit.sh
```

### Loading Trained Model

```python
from utils.src.model.modeling_encoders import BiEncoderForClassification

model = BiEncoderForClassification.from_pretrained(model_path)
```

## Dataset

This repository provides the preprocessed dataset constructed in "Detecting Fine-Grained Emotions in Literature" (https://www.mdpi.com/2076-3417/13/13/7502) as an experimental dataset. License information is provided in `dataset/license.txt`.
