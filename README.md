# EmpiricalCircumplexModel

🏆 [July 11, 2026] We are happy to announce that our paper received a <a href="https://2026.aclweb.org/program/best_papers/">Best Theme Paper Award</a> at ACL 2026 under the special theme, “Explainability of NLP Models”! 🎉

This repository provides the experimental implementation for:

<a href="https://aclanthology.org/2026.acl-long.772/">Mapping the Circumplex of Affect: Geometric Analysis of Emotion Representations via Hyperspherical Contrastive Learning</a>

The previously released <a href="https://arxiv.org/abs/2601.06575">arXiv version</a> is an earlier version of the paper and differs from the final published version. 
Please refer to the ACL Anthology version linked above for the latest version.

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


## Citation

If you like this paper, please consider citing our work:

```
@inproceedings{yamauchi-aizawa-2026-mapping,
    title = "Mapping the Circumplex of Affect: Geometric Analysis of Emotion Representations via Hyperspherical Contrastive Learning",
    author = "Yamauchi, Yusuke  and
      Aizawa, Akiko",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.772/",
    doi = "10.18653/v1/2026.acl-long.772",
    pages = "16981--17004",
    ISBN = "979-8-89176-390-6",
}
```
