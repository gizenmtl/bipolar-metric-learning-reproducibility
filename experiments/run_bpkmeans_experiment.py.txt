"""
Run Standard K-means and BP-K-means comparison.

This script is part of the reproducibility package for:
"Bipolar Metric Spaces for Geometric Regularization in Distance-Based Learning"

Examples:

    python experiments/run_bpkmeans_experiment.py --dataset iris --n-clusters 3
    python experiments/run_bpkmeans_experiment.py --dataset wine --n-clusters 3
    python experiments/run_bpkmeans_experiment.py --dataset breast_cancer --n-clusters 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.bpkmeans import BipolarKMeans, map_clusters_to_labels
from src.data_utils import load_sklearn_dataset, load_wdbc_from_csv
from src.evaluation import classification_metrics, measure_runtime_and_memory


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="breast_cancer",
        choices=["iris", "wine", "breast_cancer", "wdbc"],
        help="Dataset to use.",
    )

    parser.add_argument(
        "--wdbc-path",
        type=str,
        default=None,
        help="Path to original WDBC file if dataset='wdbc'.",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=2,
        help="Number of clusters.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum number of iterations.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "chebyshev", "minkowski"],
        help="Distance metric for BP-K-means.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/bpkmeans_comparison.csv",
        help="Output CSV path.",
    )

    return parser.parse_args()


def load_dataset(dataset_name: str, wdbc_path: str = None):
    if dataset_name == "wdbc":
        if wdbc_path is None:
            raise ValueError("Please provide --wdbc-path for the WDBC dataset.")
        return load_wdbc_from_csv(wdbc_path)

    return load_sklearn_dataset(dataset_name)


def fit_standard_kmeans(X, n_clusters, max_iter, random_state):
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=10,
        random_state=random_state,
    )
    labels = model.fit_predict(X)
    return model, labels


def fit_bpkmeans(X, n_clusters, max_iter, metric, random_state):
    model = BipolarKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        metric=metric,
        random_state=random_state,
    )
    labels = model.fit_predict(X)
    return model, labels


def main():
    args = parse_args()

    X, y = load_dataset(args.dataset, args.wdbc_path)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rows = []

    # Standard K-means
    (std_model, std_clusters), std_time, std_ram = measure_runtime_and_memory(
        fit_standard_kmeans,
        X,
        args.n_clusters,
        args.max_iter,
        args.random_state,
    )

    std_pred, std_mapping = map_clusters_to_labels(std_clusters, y)
    std_metrics = classification_metrics(y, std_pred)

    rows.append(
        {
            "dataset": args.dataset,
            "method": "Standard K-means",
            "metric": "euclidean",
            "n_clusters": args.n_clusters,
            "runtime_seconds": std_time,
            "ram_delta_mb": std_ram,
            "n_iter": int(std_model.n_iter_),
            "assignment_exclusion_rate": 0.0,
            **std_metrics,
        }
    )

    # BP-K-means
    (bp_model, bp_clusters), bp_time, bp_ram = measure_runtime_and_memory(
        fit_bpkmeans,
        X,
        args.n_clusters,
        args.max_iter,
        args.metric,
        args.random_state,
    )

    bp_pred, bp_mapping = map_clusters_to_labels(bp_clusters, y)
    bp_metrics = classification_metrics(y, bp_pred)

    if bp_model.diagnostics_ is not None:
        exclusion_rate = (
            bp_model.diagnostics_.excluded_assignments
            / bp_model.diagnostics_.total_assignments
        ) * 100.0
        n_iter = bp_model.diagnostics_.n_iter
    else:
        exclusion_rate = np.nan
        n_iter = np.nan

    rows.append(
        {
            "dataset": args.dataset,
            "method": "BP-K-means",
            "metric": args.metric,
            "n_clusters": args.n_clusters,
            "runtime_seconds": bp_time,
            "ram_delta_mb": bp_ram,
            "n_iter": int(n_iter),
            "assignment_exclusion_rate": float(exclusion_rate),
            **bp_metrics,
        }
    )

    result_df = pd.DataFrame(rows)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print("\nResults:")
    print(result_df)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()