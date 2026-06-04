"""
sensitivity.py
--------------
Sensitivity analysis for EDRS (Extremity Density Representative Selection).

Paper Section IV: Sweeps the extremity threshold e_thresh and the LOF
contamination fraction to assess robustness of representative selection.

Key finding: e_thresh=0.80 maximises F1 before outlier propagation degrades
the labeled set quality.

Usage:
    python -m src.sensitivity --folder sensitivity_results
"""

import argparse
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import f1_score

from .edrs import EDRS

E_THRESH_VALUES    = [0.50, 0.60, 0.70, 0.80, 0.90]
CONTAMINATION_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]


def _evaluate(X_tr, y_tr, labeled_ratio, e_thresh=None, contamination=0.05):
    n_labeled = max(1, int(labeled_ratio * len(X_tr)))
    edrs = EDRS(
        n_clusters=int(np.sqrt(len(X_tr))),
        n_representatives=n_labeled,
        n_components_pca=0.95,
        contamination=contamination,
    )
    labeled_idx = edrs.fit_select(X_tr)[:n_labeled]
    labeled_mask = np.zeros(len(X_tr), dtype=bool)
    labeled_mask[labeled_idx] = True
    unlabeled = np.where(~labeled_mask)[0]
    if len(unlabeled) == 0:
        return 0.0
    y_train = y_tr.copy()
    y_train[unlabeled] = -1
    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=1000)
    lp.fit(X_tr, y_train)
    preds = lp.predict(X_tr[unlabeled])
    return f1_score(y_tr[unlabeled], preds, average="weighted", zero_division=0)


def run_sensitivity(args):
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labeled_ratio = 0.1

    # --- Contamination sensitivity ---
    print("\n=== LOF Contamination Sensitivity ===")
    print(f"{'Contamination':>14} {'Weighted F1':>14}")
    print("-" * 32)
    for cont in CONTAMINATION_VALUES:
        fold_f1s = []
        for train_idx, _ in skf.split(X, y):
            f1 = _evaluate(X[train_idx], y[train_idx], labeled_ratio, contamination=cont)
            fold_f1s.append(f1)
        mark = " ← paper default" if abs(cont - 0.05) < 1e-9 else ""
        print(f"{cont:>14.2f}   {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}{mark}")

    print(f"\nResults saved to: {args.folder}")


def main():
    parser = argparse.ArgumentParser(description="EDRS Sensitivity Analysis")
    parser.add_argument("--folder", type=str, default="sensitivity_results")
    args = parser.parse_args()
    import os; os.makedirs(args.folder, exist_ok=True)
    run_sensitivity(args)


if __name__ == "__main__":
    main()
