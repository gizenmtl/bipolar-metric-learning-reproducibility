"""
BP-K-means implementation for the reproducibility package of the manuscript:
"Bipolar Metric Spaces for Geometric Regularization in Distance-Based Learning"

This file provides a compact research implementation of the BP-K-means procedure
reported in the manuscript. It is intended for reproducibility of the published
experiments, not as a production-ready clustering library.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode


def _distance_matrix(
    A: np.ndarray,
    B: np.ndarray,
    metric: str = "euclidean",
    minkowski_p: int = 2,
) -> np.ndarray:
    """
    Compute pairwise distance matrix between A and B.
    """
    if metric == "minkowski":
        return cdist(A, B, metric="minkowski", p=minkowski_p)

    if metric in {"euclidean", "cityblock", "manhattan", "chebyshev"}:
        scipy_metric = "cityblock" if metric == "manhattan" else metric
        return cdist(A, B, metric=scipy_metric)

    raise ValueError(
        "Unsupported metric. Choose from: "
        "'euclidean', 'manhattan', 'chebyshev', or 'minkowski'."
    )


@dataclass
class BPKMeansDiagnostics:
    """
    Diagnostic information collected during BP-K-means fitting.
    """
    n_iter: int
    inertia: float
    excluded_assignments: int
    total_assignments: int


class BipolarKMeans:
    """
    Bipolar K-means with a BMS-inspired assignment feasibility filter.

    For each data point and centroid, the direct point-centroid assignment is
    evaluated together with an indirect bipolar reference path. Assignments that
    violate the structural feasibility condition are penalized during the
    assignment step.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        metric: str = "euclidean",
        minkowski_p: int = 2,
        data_anchor_offset: int = 2,
        centroid_anchor_offset: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.minkowski_p = minkowski_p
        self.data_anchor_offset = data_anchor_offset
        self.centroid_anchor_offset = centroid_anchor_offset
        self.random_state = random_state

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.diagnostics_: Optional[BPKMeansDiagnostics] = None

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Select initial centroids from the data points.
        """
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices].copy()

    def _assign_clusters(self, X: np.ndarray) -> Tuple[np.ndarray, int, int, float]:
        """
        Assign each point to a centroid using the bipolar structural filter.

        Returns
        -------
        labels : np.ndarray
            Cluster assignment for each sample.
        excluded_assignments : int
            Number of point-centroid assignments excluded by the filter.
        total_assignments : int
            Total number of possible point-centroid assignments.
        inertia : float
            Sum of selected assignment costs.
        """
        n_samples = X.shape[0]

        data_anchor_indices = (
            np.arange(n_samples) + self.data_anchor_offset
        ) % n_samples

        centroid_anchor_indices = (
            np.arange(self.n_clusters) + self.centroid_anchor_offset
        ) % self.n_clusters

        X_anchor = X[data_anchor_indices]
        C = self.centroids_
        C_anchor = C[centroid_anchor_indices]

        # Direct point-centroid distance: d(x_i, c_j)
        direct_cost = _distance_matrix(
            X,
            C,
            metric=self.metric,
            minkowski_p=self.minkowski_p,
        )

        # BMS-inspired indirect path:
        # T_ij = d(x_i, c_l) + d(x_b, c_l) + d(x_b, c_j)
        x_to_secondary_centroid = _distance_matrix(
            X,
            C_anchor,
            metric=self.metric,
            minkowski_p=self.minkowski_p,
        )

        anchor_to_secondary_centroid = _distance_matrix(
            X_anchor,
            C_anchor,
            metric=self.metric,
            minkowski_p=self.minkowski_p,
        )

        anchor_to_centroid = _distance_matrix(
            X_anchor,
            C,
            metric=self.metric,
            minkowski_p=self.minkowski_p,
        )

        threshold = (
            x_to_secondary_centroid
            + anchor_to_secondary_centroid
            + anchor_to_centroid
        )

        feasible = direct_cost <= threshold

        filtered_cost = direct_cost.copy()
        filtered_cost[~feasible] = np.inf

        # Fallback for rows where all assignments are filtered out.
        invalid_rows = np.all(~np.isfinite(filtered_cost), axis=1)
        filtered_cost[invalid_rows] = direct_cost[invalid_rows]

        labels = np.argmin(filtered_cost, axis=1)
        selected_cost = filtered_cost[np.arange(n_samples), labels]

        excluded_assignments = int(np.sum(~feasible))
        total_assignments = int(feasible.size)
        inertia = float(np.sum(selected_cost))

        return labels, excluded_assignments, total_assignments, inertia

    def fit(self, X: np.ndarray) -> "BipolarKMeans":
        """
        Fit the BP-K-means model.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed the number of samples.")

        self.centroids_ = self._initialize_centroids(X)

        last_excluded = 0
        last_total = 0
        last_inertia = np.inf

        for iteration in range(1, self.max_iter + 1):
            labels, excluded, total, inertia = self._assign_clusters(X)

            new_centroids = self.centroids_.copy()

            for cluster_id in range(self.n_clusters):
                cluster_points = X[labels == cluster_id]

                if cluster_points.size > 0:
                    new_centroids[cluster_id] = np.mean(cluster_points, axis=0)
                else:
                    # Empty-cluster fallback: select the point farthest from
                    # its nearest current centroid.
                    distances = _distance_matrix(
                        X,
                        self.centroids_,
                        metric=self.metric,
                        minkowski_p=self.minkowski_p,
                    )
                    farthest_idx = np.argmax(np.min(distances, axis=1))
                    new_centroids[cluster_id] = X[farthest_idx]

            shift = np.linalg.norm(self.centroids_ - new_centroids)

            self.centroids_ = new_centroids
            self.labels_ = labels

            last_excluded = excluded
            last_total = total
            last_inertia = inertia

            if shift <= self.tol:
                break

        self.diagnostics_ = BPKMeansDiagnostics(
            n_iter=iteration,
            inertia=last_inertia,
            excluded_assignments=last_excluded,
            total_assignments=last_total,
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new samples to the nearest learned centroid using direct distance.
        """
        if self.centroids_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        distances = _distance_matrix(
            np.asarray(X, dtype=float),
            self.centroids_,
            metric=self.metric,
            minkowski_p=self.minkowski_p,
        )

        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and return cluster assignments.
        """
        self.fit(X)
        return self.labels_


def map_clusters_to_labels(
    cluster_assignments: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Map cluster IDs to class labels using majority voting.

    This utility is used only for external evaluation when ground-truth labels
    are available.
    """
    cluster_assignments = np.asarray(cluster_assignments)
    y_true = np.asarray(y_true)

    mapping: Dict[int, int] = {}

    for cluster_id in np.unique(cluster_assignments):
        labels_in_cluster = y_true[cluster_assignments == cluster_id]

        if labels_in_cluster.size > 0:
            mapping[int(cluster_id)] = int(mode(labels_in_cluster, keepdims=False).mode)

    y_pred = np.asarray(
        [mapping.get(int(cluster_id), -1) for cluster_id in cluster_assignments],
        dtype=int,
    )

    return y_pred, mapping