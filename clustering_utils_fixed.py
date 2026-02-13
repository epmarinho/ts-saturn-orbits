"""clustering_utils_fixed.py

Utilities to run grid searches over clustering hyperparameters.

This module matches the API expected by `saturn-orbits-time-series-clustering_fixed.py`:

    best_params, silhouette, db_index, ch_index = grid_search_kmeans(X, param_grid, seed)

Where `param_grid` is a dict mapping parameter name -> list of candidate values.

Main fix (for your error):
- Avoid passing duplicate `random_state` to scikit-learn estimators.
  We never do `Estimator(random_state=..., **params)`.
  Instead, we ensure `params['random_state']` exists (inject/overwrite) and call
  `Estimator(**params)`.

Robustness:
- Catches failures for a given parameter set and continues.
- Returns metrics for the best silhouette score.

Author: generated for Eraldo Pereira Marinho
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Tuple, Any, Optional

import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def _iter_param_dicts(param_grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """Yield dictionaries for every combination in a param grid.

    param_grid: {'a':[1,2], 'b':[3,4]} -> yields 4 dicts.
    """
    if not param_grid:
        yield {}
        return

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _score_clustering(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """Compute (silhouette, davies-bouldin, calinski-harabasz).

    Notes:
    - silhouette_score requires >= 2 clusters and < n_samples clusters.
    """
    # Ensure labels are 1D ints
    labels = np.asarray(labels)

    # If only one cluster (or all unique), silhouette is undefined.
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or n_clusters >= len(labels):
        raise ValueError(f"Invalid number of clusters for silhouette: {n_clusters}")

    s = float(silhouette_score(X, labels))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))
    return s, db, ch


def grid_search_kmeans(
    X: np.ndarray,
    param_grid: Dict[str, List[Any]],
    random_seed: int = 42,
) -> Tuple[Dict[str, Any], float, float, float]:
    """Grid search for KMeans.

    Returns:
      best_params, best_silhouette, best_db, best_ch
    """
    best_params: Optional[Dict[str, Any]] = None
    best_s = -float("inf")
    best_db = float("inf")
    best_ch = -float("inf")

    for params in _iter_param_dicts(param_grid):
        params = dict(params)  # copy

        # FIX: ensure random_state appears exactly once
        params["random_state"] = params.get("random_state", random_seed)

        try:
            model = KMeans(**params)
            labels = model.fit_predict(X)
            s, db, ch = _score_clustering(X, labels)
        except Exception:
            # Skip invalid combinations (rare but possible)
            continue

        if s > best_s:
            best_s = s
            best_db = db
            best_ch = ch
            best_params = params

    if best_params is None:
        raise RuntimeError("KMeans grid search found no valid configuration.")

    return best_params, float(best_s), float(best_db), float(best_ch)


def grid_search_agglomerative(
    X: np.ndarray,
    param_grid: Dict[str, List[Any]],
    random_seed: int = 42,
) -> Tuple[Dict[str, Any], float, float, float]:
    """Grid search for AgglomerativeClustering.

    random_seed is unused (kept for uniform API).
    """
    _ = random_seed

    best_params: Optional[Dict[str, Any]] = None
    best_s = -float("inf")
    best_db = float("inf")
    best_ch = -float("inf")

    for params in _iter_param_dicts(param_grid):
        params = dict(params)
        try:
            model = AgglomerativeClustering(**params)
            labels = model.fit_predict(X)
            s, db, ch = _score_clustering(X, labels)
        except Exception:
            continue

        if s > best_s:
            best_s = s
            best_db = db
            best_ch = ch
            best_params = params

    if best_params is None:
        raise RuntimeError("Agglomerative grid search found no valid configuration.")

    return best_params, float(best_s), float(best_db), float(best_ch)


def grid_search_gmm(
    X: np.ndarray,
    param_grid: Dict[str, List[Any]],
    random_seed: int = 42,
) -> Tuple[Dict[str, Any], float, float, float]:
    """Grid search for GaussianMixture.

    We score using labels from model.predict(X).
    """
    best_params: Optional[Dict[str, Any]] = None
    best_s = -float("inf")
    best_db = float("inf")
    best_ch = -float("inf")

    for params in _iter_param_dicts(param_grid):
        params = dict(params)

        # FIX: ensure random_state appears exactly once
        params["random_state"] = params.get("random_state", random_seed)

        try:
            model = GaussianMixture(**params)
            model.fit(X)
            labels = model.predict(X)
            s, db, ch = _score_clustering(X, labels)
        except Exception:
            continue

        if s > best_s:
            best_s = s
            best_db = db
            best_ch = ch
            best_params = params

    if best_params is None:
        raise RuntimeError("GMM grid search found no valid configuration.")

    return best_params, float(best_s), float(best_db), float(best_ch)
