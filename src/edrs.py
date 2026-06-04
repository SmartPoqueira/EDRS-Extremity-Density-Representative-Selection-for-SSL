"""
EDRS: Extremity-Density Representative Selection

Implements the six-stage algorithm for selecting representative unlabeled
points in semi-supervised learning settings with imbalanced tabular data.

Stages:
    1. LOF outlier removal
    2. PCA dimensionality reduction
    3. Density scoring (KNN-based)
    4. Extremity scoring (PC1-based)
    5. KMeans++ clustering
    6. Representative selection with extremity swaps
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class EDRS:
    """Extremity-Density Representative Selection.

    Parameters
    ----------
    n_neighbors_lof : int
        Number of neighbors for LOF outlier detection.
    contamination : float
        Expected proportion of outliers for LOF.
    n_components_pca : int or float
        Number of PCA components (int) or variance ratio (float).
    k_density : int
        Number of neighbors for density estimation.
    density_quantile : float
        Quantile threshold below which density is set to zero.
    n_clusters : int
        Number of KMeans++ clusters for representative grouping.
    n_representatives : int or None
        Total number of representatives to select. If None, selects
        one per cluster.
    alpha : float
        Weight for density in the combined score (1 - alpha for extremity).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_neighbors_lof=20,
        contamination=0.05,
        n_components_pca=0.95,
        k_density=10,
        density_quantile=0.25,
        n_clusters=50,
        n_representatives=None,
        alpha=0.5,
        random_state=42,
    ):
        self.n_neighbors_lof = n_neighbors_lof
        self.contamination = contamination
        self.n_components_pca = n_components_pca
        self.k_density = k_density
        self.density_quantile = density_quantile
        self.n_clusters = n_clusters
        self.n_representatives = n_representatives
        self.alpha = alpha
        self.random_state = random_state

        # Fitted attributes
        self._pca = None
        self._clean_mask = None
        self._density_scores = None
        self._extremity_scores = None
        self._combined_scores = None
        self._cluster_labels = None

    def fit_select(self, X_unlabeled):
        """Run the full EDRS pipeline and return selected indices.

        Parameters
        ----------
        X_unlabeled : np.ndarray of shape (n_samples, n_features)
            Unlabeled data matrix.

        Returns
        -------
        selected_indices : np.ndarray of shape (n_selected,)
            Indices into the original X_unlabeled of selected representatives.
        """
        n_samples = X_unlabeled.shape[0]
        original_indices = np.arange(n_samples)

        # --- Stage 1: LOF Outlier Removal ---
        clean_indices = self._remove_outliers(X_unlabeled, original_indices)
        X_clean = X_unlabeled[clean_indices]

        # --- Stage 2: PCA Dimensionality Reduction ---
        X_pca = self._reduce_dimensions(X_clean)

        # --- Stage 3: Density Scoring ---
        density = self._compute_density(X_pca)

        # --- Stage 4: Extremity Scoring ---
        extremity = self._compute_extremity(X_pca)

        # --- Stage 5: KMeans++ Clustering ---
        cluster_labels = self._cluster(X_pca)

        # --- Stage 6: Representative Selection ---
        combined = self.alpha * density + (1 - self.alpha) * extremity
        self._combined_scores = combined

        selected_local = self._select_representatives(
            combined, extremity, cluster_labels
        )

        # Map back to original indices
        selected_indices = clean_indices[selected_local]
        return selected_indices

    # ------------------------------------------------------------------
    # Stage 1: LOF Outlier Removal
    # ------------------------------------------------------------------
    def _remove_outliers(self, X, indices):
        """Remove outliers using Local Outlier Factor."""
        lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors_lof,
            contamination=self.contamination,
        )
        labels = lof.fit_predict(X)  # 1 = inlier, -1 = outlier
        inlier_mask = labels == 1
        self._clean_mask = inlier_mask
        return indices[inlier_mask]

    # ------------------------------------------------------------------
    # Stage 2: PCA Dimensionality Reduction
    # ------------------------------------------------------------------
    def _reduce_dimensions(self, X):
        """Project data to PCA space."""
        self._pca = PCA(
            n_components=self.n_components_pca,
            random_state=self.random_state,
        )
        return self._pca.fit_transform(X)

    # ------------------------------------------------------------------
    # Stage 3: Density Scoring
    # ------------------------------------------------------------------
    def _compute_density(self, X_pca):
        """Compute local density as inverse median distance to k neighbors."""
        nn = NearestNeighbors(n_neighbors=self.k_density + 1)
        nn.fit(X_pca)
        distances, _ = nn.kneighbors(X_pca)
        # Exclude self-distance (column 0)
        neighbor_dists = distances[:, 1:]
        median_dists = np.median(neighbor_dists, axis=1)

        # Local density = 1 / median distance
        density = 1.0 / (median_dists + 1e-10)

        # Threshold: zero out densities below quantile
        threshold = np.quantile(density, self.density_quantile)
        density[density < threshold] = 0.0

        # Normalize to [0, 1]
        d_max = density.max()
        if d_max > 0:
            density = density / d_max

        self._density_scores = density
        return density

    # ------------------------------------------------------------------
    # Stage 4: Extremity Scoring
    # ------------------------------------------------------------------
    def _compute_extremity(self, X_pca):
        """Compute extremity from first principal component.

        The first PC explains the most variance. Extremity measures how
        far a point is from the center (midpoint = 0.5 after normalization).
        Points at the tails score highest (1.0), central points score lowest (0.0).
        """
        pc1 = X_pca[:, 0]
        scaler = MinMaxScaler(feature_range=(0, 1))
        pc1_norm = scaler.fit_transform(pc1.reshape(-1, 1)).ravel()

        # Extremity = 2 * |pc1_norm - 0.5|, so midpoint → 0, tails → 1
        extremity = 2.0 * np.abs(pc1_norm - 0.5)

        self._extremity_scores = extremity
        return extremity

    # ------------------------------------------------------------------
    # Stage 5: KMeans++ Clustering
    # ------------------------------------------------------------------
    def _cluster(self, X_pca):
        """Partition data into clusters using KMeans++."""
        n_clusters = min(self.n_clusters, X_pca.shape[0])
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(X_pca)
        self._cluster_labels = labels
        return labels

    # ------------------------------------------------------------------
    # Stage 6: Representative Selection
    # ------------------------------------------------------------------
    def _select_representatives(self, combined, extremity, cluster_labels):
        """Select representatives per cluster, swapping non-extreme for extreme.

        For each cluster:
            1. Pick the point with highest combined score.
            2. If that point is not extreme (extremity < median),
               swap it with the most extreme point in the cluster.
        """
        unique_labels = np.unique(cluster_labels)
        n_reps = self.n_representatives or len(unique_labels)
        reps_per_cluster = max(1, n_reps // len(unique_labels))

        selected = []
        extremity_median = np.median(extremity)

        for label in unique_labels:
            mask = cluster_labels == label
            cluster_idx = np.where(mask)[0]

            if len(cluster_idx) == 0:
                continue

            # Rank by combined score
            scores = combined[cluster_idx]
            ranked = cluster_idx[np.argsort(scores)[::-1]]

            # Select top candidates
            candidates = ranked[:reps_per_cluster].tolist()

            # Swap non-extreme candidates
            for i, cand in enumerate(candidates):
                if extremity[cand] < extremity_median:
                    # Find most extreme point in cluster not already selected
                    ext_scores = extremity[cluster_idx]
                    ext_ranked = cluster_idx[np.argsort(ext_scores)[::-1]]
                    for alt in ext_ranked:
                        if alt not in candidates:
                            candidates[i] = alt
                            break

            selected.extend(candidates)

        return np.array(selected[:n_reps])
