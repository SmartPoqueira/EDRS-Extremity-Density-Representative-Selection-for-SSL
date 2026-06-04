"""
Semi-Supervised Learning pipeline using EDRS representative selection.

Workflow:
    1. Split data into labeled and unlabeled partitions.
    2. Run EDRS on the unlabeled set to select representatives.
    3. Pseudolabel selected representatives using a base classifier.
    4. Retrain the classifier on labeled + pseudolabeled data.
    5. Evaluate on the held-out test set.
"""

import argparse
import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .edrs import EDRS
from .data_loader import load_dataset
from .utils import set_seed


def run_ssl_experiment(config):
    """Run the full SSL experiment with EDRS."""
    set_seed(config["seed"])
    dataset_name = config["dataset"]
    X, y = load_dataset(dataset_name)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Samples: {X.shape[0]} | Features: {X.shape[1]}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["seed"])
    f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        n_labeled = max(10, int(len(X_train_full) * config["labeled_ratio"]))
        rng = np.random.RandomState(config["seed"] + fold)
        labeled_mask = np.zeros(len(X_train_full), dtype=bool)

        for cls in np.unique(y_train_full):
            cls_idx = np.where(y_train_full == cls)[0]
            n_cls = max(2, int(n_labeled * len(cls_idx) / len(X_train_full)))
            chosen = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
            labeled_mask[chosen] = True

        X_labeled = X_train_full[labeled_mask]
        y_labeled = y_train_full[labeled_mask]
        X_unlabeled = X_train_full[~labeled_mask]

        scaler = StandardScaler()
        X_labeled_s = scaler.fit_transform(X_labeled)
        X_unlabeled_s = scaler.transform(X_unlabeled)
        X_test_s = scaler.transform(X_test)

        edrs = EDRS(
            n_neighbors_lof=config["edrs"]["n_neighbors_lof"],
            contamination=config["edrs"]["contamination"],
            n_components_pca=config["edrs"]["n_components_pca"],
            k_density=config["edrs"]["k_density"],
            density_quantile=config["edrs"]["density_quantile"],
            n_clusters=config["edrs"]["n_clusters"],
            n_representatives=config["edrs"]["n_representatives"],
            alpha=config["edrs"]["alpha"],
            random_state=config["seed"],
        )
        selected_idx = edrs.fit_select(X_unlabeled_s)

        base_clf = _get_classifier(config["classifier"])
        base_clf.fit(X_labeled_s, y_labeled)
        pseudo_labels = base_clf.predict(X_unlabeled_s[selected_idx])

        X_combined = np.vstack([X_labeled_s, X_unlabeled_s[selected_idx]])
        y_combined = np.concatenate([y_labeled, pseudo_labels])

        final_clf = _get_classifier(config["classifier"])
        final_clf.fit(X_combined, y_combined)

        y_pred = final_clf.predict(X_test_s)
        f1 = f1_score(y_test, y_pred, average="macro")
        f1_scores.append(f1)
        print(f"  Fold {fold}: F1={f1:.4f} | Labeled={len(X_labeled)}, Selected={len(selected_idx)}")

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    print(f"\n>>> {dataset_name} -- Mean F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    return mean_f1, std_f1


def _get_classifier(name):
    classifiers = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "svm": SVC(kernel="rbf", random_state=42, class_weight="balanced"),
    }
    if name not in classifiers:
        raise ValueError(f"Unknown classifier: {name}. Choose from {list(classifiers)}")
    return classifiers[name]


def main():
    parser = argparse.ArgumentParser(description="EDRS SSL Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config["dataset"] = args.dataset

    datasets = config.get("datasets", [config["dataset"]])
    results = {}

    for ds in datasets:
        config["dataset"] = ds
        mean_f1, std_f1 = run_ssl_experiment(config)
        results[ds] = {"mean_f1": mean_f1, "std_f1": std_f1}

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ds, r in results.items():
        print(f"  {ds:20s} -- F1: {r['mean_f1']:.4f} +/- {r['std_f1']:.4f}")


if __name__ == "__main__":
    main()
