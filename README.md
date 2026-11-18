# QDMF-Net

Reference implementation of **Quality-aware disentangled multimodal fusion with cross-modal contrastive learning for esophageal cancer survival prediction**.

QDMF-Net fuses whole-slide histopathology features with three molecular modalities (RNA-seq, copy-number variation, DNA methylation) for overall-survival prediction. Each modality is decomposed into a shared and a private subspace; the shared subspaces are aligned with a quality-weighted cross-modal contrastive objective; and a quality-aware gate assigns patient-specific fusion weights at inference time.

## Repository structure

```
QDMF-Net/
├── data/
│   ├── dataset.py            # MultiModalDataset: patch bags + omics tensor + survival record
│   ├── split.py              # split construction / loading utilities
│   ├── survival.csv          # sample-level time and event fields (see Data)
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

PyTorch with CUDA is required for training. Mixed precision is used for the convolutional omics encoder.

## Data

Raw data are public and were obtained from the Genomic Data Commons (https://gdc.cancer.gov/); they are **not** redistributed here. To rebuild the inputs:

1. **Pathology features** — tile WSIs at 20x into 256x256 patches, keep tissue patches, and extract 1024-d features with the frozen UNI foundation model. Store one file per patient: `path_features/{patient_id}.pt`, tensor of shape `(N_patches, 1024)`.
2. **Omics tensor** — align RNA-seq (log2(TPM+1)), gene-level CNV and gene-level methylation beta values on a common gene axis: `processed/tensor.npy` of shape `(17472, N_patients, 3)`, plus `processed/genes.txt` and `processed/patients.txt`.
3. **Survival** — `data/survival.csv` with columns `sample_id, time, event, patient_id` (`event = 1` death, `0` censored; time in months).

Gene filtering (missingness, variance, univariate Cox screen to the top 1,000 genes) and all distribution-dependent preprocessing are fitted inside each training fold only.

`data/survival.csv` and `data/survival.tsv` contain only the de-identified sample-level survival fields needed to index the public cohort; no WSIs, molecular matrices, derived features or fold-level predictions are redistributed.

## Using the model

The core model is `models.full_model.DisentangledModel`. It takes a patch-feature bag and an omics tensor and returns the Cox risk score, the auxiliary three-class logit and the per-modality quality indicators:

```python
import torch
from models.full_model import DisentangledModel

model = DisentangledModel(
    path_input_dim=1024,
    omics_input_dim=17472,
    hidden_dim=512,
    disentangle_dim=128,
    num_classes=3,
)
risk, risk_class, sigma_p, sigma_o = model(patch_features, omics_tensor)
```

## Model outline

1. **Pathology aggregation** — gated-attention multiple-instance learning maps a variable-length bag of 1024-d patch features to a 512-d slide embedding.
2. **Omics encoding** — 1D convolution (kernel 7, three blocks, batch norm, global average pooling) maps the `17472 x 3` tensor to a 512-d embedding.
3. **Shared-private disentanglement** — each modality is projected into a 128-d shared and a 128-d private subspace; the absolute cosine similarity between the two is minimized. Private representations act as a training-time regularizer and do not enter prediction.
4. **Quality-weighted cross-modal contrastive alignment** — bidirectional InfoNCE (temperature 0.07) between the shared representations of the same patient, with each pair weighted by the estimated modality quality.
5. **Quality-aware adaptive fusion** — a learnable logit per modality is adjusted by `softplus(kappa) * log(sigma)`; a softmax over the two logits gives patient-specific weights, and the fused shared representation feeds the Cox head.

## Citation

If you use this code, please cite:

> Meng X, Qian F, Fang X, Cui T, Li C, Wang Q. Quality-aware disentangled multimodal fusion with cross-modal contrastive learning for esophageal cancer survival prediction. *BMC Bioinformatics* (under review).

## Contact

Xiangyang Meng — mengxiangyang0225@gmail.com

## License

MIT. See [LICENSE](LICENSE).
