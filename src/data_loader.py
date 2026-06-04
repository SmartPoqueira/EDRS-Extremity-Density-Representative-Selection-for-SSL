"""
Data loader for UCI benchmark datasets.
All data is fetched from external sources — no raw files in this repository.
"""

import numpy as np
from sklearn.datasets import fetch_openml


DATASET_REGISTRY = {
    "bank": {"openml_id": 1461, "target": "y"},
    "cancer": {"openml_id": 15, "target": "Class"},
    "thyroid": {"openml_id": 40474, "target": "target"},
    "sonar": {"openml_id": 40, "target": "Class"},
    "seismic": {"openml_id": 1567, "target": "class"},
    "wilt": {"openml_id": 40983, "target": "class"},
}


def load_dataset(name):
    """Load a dataset by name from OpenML.

    Parameters
    ----------
    name : str
        Dataset name (bank, cancer, thyroid, sonar, seismic, wilt).

    Returns
    -------
    X : np.ndarray
    y : np.ndarray (integer-encoded)
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}"
        )

    info = DATASET_REGISTRY[name]
    data = fetch_openml(data_id=info["openml_id"], as_frame=True, parser="auto")
    X = data.data
    y = data.target

    # Encode categorical features
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    cat_cols = X.select_dtypes(include=["category", "object"]).columns
    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    X = X.values.astype(np.float64)
    np.nan_to_num(X, copy=False)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y
