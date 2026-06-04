# EDRS: Extremity-Density Representative Selection for Semi-Supervised Learning

A representative sample selection method for semi-supervised learning (SSL) on imbalanced tabular data. EDRS selects the most informative unlabeled points by combining **density estimation** and **extremity scoring** in a reduced PCA space, after removing outliers via Local Outlier Factor (LOF). Selected representatives are pseudolabeled and merged with the original labeled set to train a downstream classifier.

## Overview

Semi-supervised learning leverages unlabeled data to improve classifiers when labeled samples are scarce. However, naively incorporating all unlabeled data can degrade performance — especially under class imbalance, where the majority class dominates pseudolabel assignments. EDRS addresses this by selecting a small, high-quality subset of unlabeled points that are both **structurally representative** (high density) and **informative** (high extremity), ensuring coverage of minority-class boundaries.

<p align="center">
  <img src="paper/figures/algorithm_overview.png" width="700"/>
</p>

## Method

The EDRS algorithm operates in six stages:

1. **Outlier Removal** — Local Outlier Factor (LOF) removes anomalous points that would distort density and extremity estimates.
2. **Dimensionality Reduction** — PCA projects the data to a lower-dimensional space. PCA is preferred over t-SNE/MDS because the underlying relationships in tabular data are predominantly linear.
3. **Density Scoring** — For each point, compute the median distance to its *k* nearest neighbors. Local density is defined as the inverse of this median. Points below a quantile threshold are assigned zero density.
4. **Extremity Scoring** — The first principal component (explaining the most variance) is normalized to [0, 1]. Extremity is computed as the absolute distance from the midpoint (0.5), so boundary/tail points score highest.
5. **Clustering** — KMeans++ partitions the data into *N* clusters.
6. **Representative Selection** — Within each cluster, points are ranked by a combined density-extremity score. Non-extreme points are swapped for more extreme alternatives from the same cluster.

## Results

Performance comparison on six UCI benchmark datasets (F1-score, macro-averaged):

| Dataset | Supervised | Self-Training | Co-Training | Label Propagation | **EDRS (Ours)** |
|---|---|---|---|---|---|
| Bank | 0.571 | 0.573 | 0.560 | 0.574 | **0.612** |
| Cancer | 0.943 | 0.936 | 0.927 | 0.937 | **0.953** |
| Thyroid | 0.829 | 0.843 | 0.832 | 0.817 | **0.868** |
| Sonar | 0.712 | 0.744 | 0.729 | 0.720 | **0.769** |
| Seismic | 0.568 | 0.583 | 0.567 | 0.556 | **0.605** |
| Wilt | 0.894 | 0.905 | 0.889 | 0.893 | **0.917** |

## Project Structure

```
EDRS/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── config.yaml          # Hyperparameters matching the paper
├── src/
│   ├── __init__.py
│   ├── edrs.py              # Core EDRS algorithm
│   ├── ssl_pipeline.py      # Semi-supervised training pipeline
│   ├── data_loader.py       # UCI dataset loader (external links)
│   └── utils.py             # Seed, metrics, helpers
├── paper/
│   ├── main.tex             # LaTeX manuscript
│   ├── references.bib
│   └── figures/
└── scripts/
    └── run_experiment.sh    # Reproduce paper experiments
```

## Datasets

All datasets are loaded from external sources. No raw data files are included in this repository.

| Dataset | Source | Instances | Features | Imbalance Ratio |
|---|---|---|---|---|
| Bank Marketing | [UCI](https://archive.ics.uci.edu/ml/datasets/bank+marketing) | 45,211 | 16 | 88:12 |
| Breast Cancer | [UCI](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) | 569 | 30 | 63:37 |
| Thyroid | [UCI](https://archive.ics.uci.edu/ml/datasets/thyroid+disease) | 7,200 | 21 | 92:8 |
| Sonar | [UCI](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)) | 208 | 60 | 53:47 |
| Seismic Bumps | [UCI](https://archive.ics.uci.edu/ml/datasets/seismic-bumps) | 2,584 | 18 | 93:7 |
| Wilt | [UCI](https://archive.ics.uci.edu/ml/datasets/wilt) | 4,839 | 5 | 95:5 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment (all datasets)
bash scripts/run_experiment.sh

# Run single dataset
python -m src.ssl_pipeline --config configs/config.yaml --dataset bank
```

## Citation

```bibtex
@article{duranlopez2025edrs,
  title={EDRS: Extremity-Density Representative Selection for Semi-Supervised Learning on Imbalanced Tabular Data},
  author={Dur{\'a}n-L{\'o}pez, Alberto and Bola{\~n}os-Mart{\'i}nez, Daniel and Berm{\'u}dez-Edo, Mar{\'i}a and Garc{\'i}a-Nieto, Jos{\'e}},
  journal={<Journal Name>},
  year={2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
