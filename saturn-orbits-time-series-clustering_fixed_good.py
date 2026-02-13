# saturn-orbits-time-series-clustering_fixed_good.py
# "good" version: ablation-ready, frozen DR by default, and (important) avoids
# caching/storing raw batches to prevent numba/sktime segfaults on some setups.
#
# Key outputs:
#   - ablation_<phi>.csv  (one row per hkey, best config for that hkey)
#   - <best_method>_dynamic_map_<phi>_hkeyXX.csv and labels subfolder for best global
#
# Run example:
#   python saturn-orbits-time-series-clustering_fixed_good.py --phi phi1 --hkeys 10,2,7,8

from __future__ import annotations

import argparse
import os
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from io_utils import configure_logging, print_and_log, load_batch, create_dynamic_map
from features import preprocess_batch
from clustering_utils_fixed import grid_search_agglomerative, grid_search_gmm, grid_search_kmeans
from plots import plot_clusters, plot_sample_series


def parse_hkeys(spec: str) -> List[int]:
    """Parse --hkeys specification.

    Accepted formats:
      - "4"            -> [4]
      - "1-15"         -> [1,2,...,15]
      - "4,6,8"        -> [4,6,8]
      - "1-3,7,9-10"   -> [1,2,3,7,9,10]
    """
    spec = (spec or "").strip()
    if not spec:
        return [4]

    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i, b_i = int(a.strip()), int(b.strip())
            step = 1 if b_i >= a_i else -1
            out.extend(range(a_i, b_i + step, step))
        else:
            out.append(int(part))

    # de-dup but preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _list_orbit_files(directory_path: str) -> List[str]:
    files = sorted(
        [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".csv")
        ]
    )
    return files


def _load_original_data(files: List[str], col_idx: int, batch_size: int) -> np.ndarray:
    """Load raw time-series for plotting only (done once, after best global is known)."""
    batches: List[np.ndarray] = []
    for start in range(0, len(files), batch_size):
        batch_files = files[start:start + batch_size]
        batch_data = load_batch(batch_files, col_idx=col_idx)
        batches.append(batch_data)
    return np.vstack(batches)


def main(
    phi: str,
    directory_path: str,
    labels_folder: str,
    hkeys_to_test: Sequence[int],
    batch_size: int = 22288,
    subset_size: int = 22288,
    random_seed: int = 42,
    # Frozen DR defaults (ablation-friendly). You can unfreeze by passing multiple values.
    umap_components: Sequence[int] = (30,),
    umap_neighbors: Sequence[int] = (70,),
    umap_min_dist: Sequence[float] = (0.00125,),
    pca_components: Sequence[int] = (3,),
) -> None:
    assert phi in ("phi1", "phi2")
    angle_name = phi
    col_idx = 0 if phi == "phi1" else 1

    configure_logging(angle_name)
    print_and_log(f"\nReading data from {directory_path}\n")

    files = _list_orbit_files(directory_path)
    total_files = len(files)
    print_and_log(f"Number of input files: {total_files}")

    if total_files == 0:
        raise RuntimeError(f"No .csv files found in: {directory_path}")

    # Optional subsampling
    if total_files > subset_size:
        np.random.seed(random_seed)
        idx = np.random.choice(len(files), size=subset_size, replace=False)
        files = [files[i] for i in idx]
        print_and_log(f"Subsampled to {len(files)} files (subset_size={subset_size})")

    # DR grids
    umap_grid = product(list(umap_components), list(umap_neighbors), list(umap_min_dist))
    n_pca_values = list(pca_components)

    # Clustering setup
    n_clusters = 4
    print_and_log(f"Number of clusters: {n_clusters}")

    kmeans_param_grid = {
        "n_clusters": [n_clusters],
        "init": ["k-means++", "random"],
        "n_init": [6],
        "max_iter": [300],
        "random_state": [random_seed],
    }
    agglomerative_param_grid = {
        "n_clusters": [n_clusters],
        "linkage": ["ward", "complete", "average", "single"],
    }
    gmm_param_grid = {
        "n_components": [n_clusters],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "random_state": [random_seed],
    }

    clustering_methods = {
        "K-Means": (grid_search_kmeans, kmeans_param_grid),
        "Agglomerative": (grid_search_agglomerative, agglomerative_param_grid),
        "GMM": (grid_search_gmm, gmm_param_grid),
    }

    # Best global trackers (across all tested hkeys)
    best_s_global = -float("inf")
    best_global: Dict[str, Any] = {}

    # Collect per-hkey results for ablation table
    ablation_rows: List[Dict[str, Any]] = []

    # -------------------------------
    # Main loop over hkeys
    # -------------------------------
    for hkey in hkeys_to_test:
        print_and_log(f"\n=== Testing hkey={hkey} ===")

        # Best trackers within this hkey
        best_s_h = -float("inf")
        best_db_h = float("inf")
        best_ch_h: Optional[float] = None
        best_method_h: Optional[str] = None
        best_model_params_h: Optional[Dict[str, Any]] = None
        best_umap_params_h: Optional[Tuple[int, int, float]] = None
        best_n_pca_h: Optional[int] = None

        # Build features for this hkey WITHOUT caching raw batches
        feat_batches: List[np.ndarray] = []
        for start in range(0, len(files), batch_size):
            batch_files = files[start:start + batch_size]
            batch_data = load_batch(batch_files, col_idx=col_idx)

            print_and_log(
                f"Batch {start // batch_size + 1}/"
                f"{(len(files) + batch_size - 1) // batch_size}: "
                f"loading {len(batch_files)} files"
            )

            # IMPORTANT: preprocess immediately after loading (reduces segfault risk)
            batch_features = preprocess_batch(batch_data, hkey=hkey)
            print_and_log(f"Batch features shape: {batch_features.shape}")
            feat_batches.append(batch_features)

            # Explicitly drop references ASAP (helps some environments)
            del batch_data
            del batch_features

        all_features = np.vstack(feat_batches)
        del feat_batches
        print_and_log(f"All features (hkey={hkey}): {all_features.shape}")

        # For each DR configuration + clustering method
        for n_components, n_neighbors, min_dist in umap_grid:
            print_and_log(
                f"UMAP: components={n_components}, neighbors={n_neighbors}, min_dist={min_dist}"
            )

            reducer = umap.UMAP(
                n_components=int(n_components),
                n_neighbors=int(n_neighbors),
                min_dist=float(min_dist),
                metric="euclidean",
                # random_state=random_seed,  # keep commented unless you want deterministic UMAP
            )
            reduced_umap = reducer.fit_transform(all_features)

            for n_pca in n_pca_values:
                pca = PCA(n_components=int(n_pca))
                reduced_data = pca.fit_transform(reduced_umap)

                for method_name, (grid_search_fn, param_grid) in clustering_methods.items():
                    best_params, silhouette, db_index, ch_index = grid_search_fn(
                        reduced_data, param_grid, random_seed
                    )

                    # Best for THIS hkey
                    if silhouette > best_s_h:
                        best_s_h = silhouette
                        best_db_h = db_index
                        best_ch_h = ch_index
                        best_method_h = method_name
                        best_model_params_h = best_params
                        best_umap_params_h = (int(n_components), int(n_neighbors), float(min_dist))
                        best_n_pca_h = int(n_pca)

                    # Best GLOBAL across hkeys
                    if silhouette > best_s_global:
                        best_s_global = silhouette
                        best_global = {
                            "hkey": int(hkey),
                            "method": method_name,
                            "model_params": best_params,
                            "umap_params": (int(n_components), int(n_neighbors), float(min_dist)),
                            "pca_n": int(n_pca),
                            "silhouette": float(silhouette),
                            "davies_bouldin": float(db_index),
                            "calinski_harabasz": float(ch_index),
                            # Save reduced_data for final model/plots
                            "reduced_data": reduced_data,
                        }

        # Record hkey result (use variables so VSCode won't fade them)
        if best_method_h is None or best_model_params_h is None or best_umap_params_h is None or best_n_pca_h is None:
            raise RuntimeError(f"No valid configuration found for hkey={hkey}. Check grids/inputs.")

        row = {
            "phi": phi,
            "hkey": int(hkey),
            "best_method": best_method_h,
            "silhouette": float(best_s_h),
            "davies_bouldin": float(best_db_h),
            "calinski_harabasz": float(best_ch_h) if best_ch_h is not None else None,
            "umap_n_components": best_umap_params_h[0],
            "umap_n_neighbors": best_umap_params_h[1],
            "umap_min_dist": best_umap_params_h[2],
            "pca_n": int(best_n_pca_h),
            "model_params": str(best_model_params_h),
        }
        ablation_rows.append(row)

        print_and_log(
            f"[HKEY RESULT] hkey={hkey} method={best_method_h} "
            f"s={best_s_h:.4f} DB={best_db_h:.4f} CH={float(best_ch_h):.4f}"
        )

        # Reduce peak memory between hkeys
        del all_features

    # -------------------------------
    # Save ablation table
    # -------------------------------
    ablation_df = pd.DataFrame(ablation_rows).sort_values(["phi", "hkey"])
    ablation_csv = f"ablation_{phi}.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    print_and_log(f"\nSaved ablation table to {ablation_csv}\n")
    print(ablation_df[["hkey", "best_method", "silhouette", "davies_bouldin", "calinski_harabasz",
                      "pca_n", "umap_n_components", "umap_n_neighbors", "umap_min_dist"]])

    # -------------------------------
    # Final best global configuration
    # -------------------------------
    print_and_log("\nFinal best configuration (GLOBAL):")
    print_and_log(f"Best hkey: {best_global.get('hkey')}")
    print_and_log(f"Best method: {best_global.get('method')}")
    print_and_log(f"Best model params: {best_global.get('model_params')}")
    print_and_log(f"Best UMAP params: {best_global.get('umap_params')}")
    print_and_log(f"Best PCA components: {best_global.get('pca_n')}")
    print_and_log(f"Silhouette: {best_global.get('silhouette'):.4f}")
    print_and_log(f"Davies-Bouldin: {best_global.get('davies_bouldin'):.4f}")
    print_and_log(f"Calinski-Harabasz: {best_global.get('calinski_harabasz'):.4f}")

    if not best_global:
        raise RuntimeError("No valid configuration found globally; check grids and inputs.")

    reduced_best = best_global["reduced_data"]
    method_best = best_global["method"]
    params_best = best_global["model_params"]
    hkey_best = best_global["hkey"]
    pca_best = best_global["pca_n"]

    # Fit final model to get clusters
    model_cls = {
        "K-Means": KMeans,
        "Agglomerative": AgglomerativeClustering,
        "GMM": GaussianMixture,
    }[method_best]

    final_model = model_cls(**params_best)
    best_clusters = final_model.fit_predict(reduced_best)

    # Load original time-series only once (for plots)
    original_data = _load_original_data(files, col_idx=col_idx, batch_size=batch_size)

    angle_limits = (-180, 180) if phi == "phi1" else (0, 360)
    plot_clusters(reduced_best, best_clusters, method_best, angle_name, angle_limits, n_pca=pca_best)
    plot_sample_series(original_data, best_clusters, files, method_best, angle_limits)

    # Output map/labels per best global hkey
    labels_folder_hkey = os.path.join(labels_folder, f"hkey_{hkey_best:02d}")
    os.makedirs(labels_folder_hkey, exist_ok=True)

    output_file = f"{method_best}_dynamic_map_{phi}_hkey{hkey_best:02d}.csv"
    create_dynamic_map(directory_path, output_file, labels_folder_hkey, best_clusters, files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi", choices=["phi1", "phi2"], required=True)
    parser.add_argument("--data-dir", default="./orbits-dataset")
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument(
        "--hkeys",
        default="4",
        help="hkeys to test. Examples: '4', '1-15', '4,6,8', '1-3,7,9-10'.",
    )
    parser.add_argument("--batch-size", type=int, default=22288)
    parser.add_argument("--subset-size", type=int, default=22288)
    parser.add_argument("--seed", type=int, default=42)

    # Optional: unfreeze DR from CLI if you want
    parser.add_argument("--umap-components", default="30", help="Comma/range list, e.g. '20,30' or '30'")
    parser.add_argument("--umap-neighbors", default="70", help="Comma/range list, e.g. '60,70' or '70'")
    parser.add_argument("--umap-min-dist", default="0.00125", help="Comma list of floats, e.g. '0.000625,0.00125'")
    parser.add_argument("--pca-components", default="3", help="Comma/range list, e.g. '2,3,4' or '3'")

    args = parser.parse_args()

    labels_dir = args.labels_dir or os.path.join(args.data_dir, f"labels_{args.phi}")
    hkeys_to_test = parse_hkeys(args.hkeys)

    def _parse_int_list(s: str) -> List[int]:
        s = (s or "").strip()
        if not s:
            return []
        out: List[int] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                a_i, b_i = int(a.strip()), int(b.strip())
                step = 1 if b_i >= a_i else -1
                out.extend(range(a_i, b_i + step, step))
            else:
                out.append(int(part))
        # de-dup
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _parse_float_list(s: str) -> List[float]:
        s = (s or "").strip()
        if not s:
            return []
        out: List[float] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
        # de-dup
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    main(
        phi=args.phi,
        directory_path=args.data_dir,
        labels_folder=labels_dir,
        hkeys_to_test=hkeys_to_test,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        random_seed=args.seed,
        umap_components=tuple(_parse_int_list(args.umap_components)),
        umap_neighbors=tuple(_parse_int_list(args.umap_neighbors)),
        umap_min_dist=tuple(_parse_float_list(args.umap_min_dist)),
        pca_components=tuple(_parse_int_list(args.pca_components)),
    )
