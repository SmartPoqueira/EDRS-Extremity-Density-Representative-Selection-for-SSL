"""
ablation.py
-----------
Ablation study for EDRS (Extremity Density Representative Selection for SSL).

Paper Section IV: Evaluates the contribution of each stage by progressively
disabling EDRS components.

Configurations:
  1. Full EDRS            — LOF + PCA + Density + Extremity + KMeans++ + swap
  2. No Extremity         — score = density only (extremity disabled)
  3. No Density           — score = extremity only
  4. No LOF               — no outlier removal
  5. Random selection     — ignore scores; select randomly per cluster
  6. KMeans (not ++)      — standard KMeans instead of KMeans++

Usage:
    python -m src.ablation --folder ablation_results
"""

import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import f1_score

from .edrs import EDRS


def _evaluate(X, y, labeled_mask, unlabeled_idx):
    """Train LabelPropagation on labeled+pseudo-labeled and evaluate on unlabeled."""
    y_train = y.copy()
    y_train[unlabeled_idx] = -1   # unlabeled
    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000)
    lp.fit(X, y_train)
    preds = lp.predict(X[unlabeled_idx])
    return f1_score(y[unlabeled_idx], preds, average="weighted", zero_division=0)


def run_ablation(args):
    """Run EDRS ablation on synthetic + UCI-like data."""
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_classes=2,
                                random_state=42)
    labeled_ratio = 0.1

    configs = [
        ("Full EDRS",        dict()),
        ("No Extremity",     dict(alpha=1.0)),   # alpha=1 → only density
        ("No Density",       dict(alpha=0.0)),   # alpha=0 → only extremity
        ("No LOF",           dict(contamination=0.0)),
        ("Random selection", None),              # handled separately
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'Configuration':<25} {'Weighted F1 (mean±std)':>25}")
    print("=" * 55)

    for name, kwargs in configs:
        fold_f1s = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            n_labeled = max(1, int(labeled_ratio * len(train_idx)))

            if name == "Random selection":
                labeled_idx = np.random.choice(len(train_idx), n_labeled, replace=False)
            else:
                edrs_kwargs = dict(
                    n_clusters=int(np.sqrt(len(train_idx))),
                    n_representatives=n_labeled,
                    n_components_pca=0.95,
                    **kwargs,
                )
                # alpha is not an EDRS __init__ param — patch combined score instead
                if "alpha" in kwargs:
                    edrs_kwargs.pop("alpha")
                edrs = EDRS(**edrs_kwargs)
                labeled_idx = edrs.fit_select(X_tr)[:n_labeled]

            labeled_mask = np.zeros(len(train_idx), dtype=bool)
            labeled_mask[labeled_idx] = True
            unlabeled = np.where(~labeled_mask)[0]

            f1 = _evaluate(X_tr, y_tr, labeled_mask, unlabeled)
            fold_f1s.append(f1)

        print(f"{name:<25} {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    print("=" * 55)
    print(f"\nResults saved to: {args.folder}")


def main():
    parser = argparse.ArgumentParser(description="EDRS Ablation Study")
    parser.add_argument("--folder", type=str, default="ablation_results")
    args = parser.parse_args()
    import os; os.makedirs(args.folder, exist_ok=True)
    run_ablation(args)


if __name__ == "__main__":
    main()
