# QDMF-Net

Reference implementation of *Quality-aware disentangled multimodal fusion with cross-modal contrastive learning for esophageal cancer survival prediction*. Whole-slide pathology features are fused with RNA-seq, copy-number and methylation profiles: each modality is split into shared and private subspaces, the shared subspaces are aligned by a quality-weighted contrastive objective, and a quality-aware gate sets patient-level fusion weights at inference.

## Repository structure

```
QDMF-Net/
├── data/
│   ├── dataset.py            # MultiModalDataset: patch bags + omics tensor + survival record
│   ├── split.py              # split construction / loading utilities
│   ├── survival.csv          # sample-level time and event fields
│   └── survival.tsv
├── models/
│   ├── aggregator.py         # gated-attention MIL pathology aggregator
│   ├── omic_encoder.py       # 1D convolutional omics encoder
│   ├── disentangler.py       # shared/private projection heads + orthogonal regularizer
│   ├── contrastive.py        # bidirectional InfoNCE with quality tempering
│   ├── uncertainty.py        # per-modality quality (sigma) estimator
│   ├── adaptive_fusion.py    # quality-aware softmax fusion gate
│   ├── fusion_head.py        # Cox head, auxiliary three-class head, multi-task loss
│   ├── full_model.py         # QDMF-Net (DisentangledModel), the model used in the paper
│   └── model.py              # baseline two-tower fusion model
├── requirements.txt
└── LICENSE
```

## Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch with CUDA is required for training.

## Data

Public TCGA-ESCA data from the Genomic Data Commons (https://gdc.cancer.gov/); raw data are not redistributed here. Expected inputs:

- `path_features/{patient_id}.pt` — UNI patch features, `(N_patches, 1024)`
- `processed/tensor.npy` — `(17472, N_patients, 3)` omics tensor, with `genes.txt` and `patients.txt`
- `data/survival.csv` — columns `sample_id, time, event, patient_id`

`data/survival.csv` and `data/survival.tsv` hold only de-identified sample-level survival fields; no WSIs, molecular matrices or derived features are redistributed.

## Usage

```python
from models.full_model import DisentangledModel

model = DisentangledModel(path_input_dim=1024, omics_input_dim=17472)
risk, risk_class, sigma_p, sigma_o = model(patch_features, omics_tensor)
```

## Citation

Meng X, Qian F, Fang X, Cui T, Li C, Wang Q. *BMC Bioinformatics* (under review).

## Contact

Xiangyang Meng — mengxiangyang0225@gmail.com

## License

MIT. See [LICENSE](LICENSE).
