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


def _evaluate(X_tr, y_tr, labeled_ratio, e_thresh=0.80, contamination=0.05):
    n_labeled = max(1, int(labeled_ratio * len(X_tr)))
    edrs = EDRS(
        n_clusters=int(np.sqrt(len(X_tr))),
        n_representatives=n_labeled,
        n_components_pca=2,
        contamination=contamination,
        e_thresh=e_thresh,
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
    import os
    os.makedirs(args.folder, exist_ok=True)
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labeled_ratio = 0.1

    results_lines = []

    # --- Contamination sensitivity ---
    header1 = "\n=== LOF Contamination Sensitivity ==="
    cols1 = f"{'Contamination':>14} {'Weighted F1':>14}"
    sep1 = "-" * 32
    print(header1)
    print(cols1)
    print(sep1)
    results_lines.extend([header1, cols1, sep1])

    for cont in CONTAMINATION_VALUES:
        fold_f1s = []
        for train_idx, _ in skf.split(X, y):
            f1 = _evaluate(X[train_idx], y[train_idx], labeled_ratio, contamination=cont, e_thresh=0.80)
            fold_f1s.append(f1)
        mark = " ← paper default" if abs(cont - 0.05) < 1e-9 else ""
        res_line = f"{cont:>14.2f}   {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}{mark}"
        print(res_line)
        results_lines.append(res_line)

    # --- Extremity Threshold sensitivity ---
    header2 = "\n=== Extremity Threshold (e_thresh) Sensitivity ==="
    cols2 = f"{'e_thresh':>14} {'Weighted F1':>14}"
    sep2 = "-" * 32
    print(header2)
    print(cols2)
    print(sep2)
    results_lines.extend([header2, cols2, sep2])

    for ethresh in E_THRESH_VALUES:
        fold_f1s = []
        for train_idx, _ in skf.split(X, y):
            f1 = _evaluate(X[train_idx], y[train_idx], labeled_ratio, contamination=0.05, e_thresh=ethresh)
            fold_f1s.append(f1)
        mark = " ← paper default" if abs(ethresh - 0.80) < 1e-9 else ""
        res_line = f"{ethresh:>14.2f}   {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}{mark}"
        print(res_line)
        results_lines.append(res_line)

    footer = f"\nResults saved to: {args.folder}"
    print(footer)
    results_lines.append(footer)

    with open(os.path.join(args.folder, "sensitivity_results.txt"), "w") as f:
        f.write("\n".join(results_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="EDRS Sensitivity Analysis")
    parser.add_argument("--folder", type=str, default="sensitivity_results")
    args = parser.parse_args()
    import os; os.makedirs(args.folder, exist_ok=True)
    run_sensitivity(args)


if __name__ == "__main__":
    main()
