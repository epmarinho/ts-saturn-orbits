# saturn-orbits-time-series-clustering_fixed.py
import argparse
import os
from itertools import product
from typing import List, Sequence, Tuple, Optional, Dict

import numpy as np
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
            out.extend(list(range(a_i, b_i + step, step)))
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


def main(
    phi: str,
    directory_path: str,
    labels_folder: str,
    hkeys_to_test: Sequence[int],
    batch_size: int = 22288,
    subset_size: int = 22288,
    random_seed: int = 42,
) -> None:
    assert phi in ("phi1", "phi2")
    angle_name = phi
    col_idx = 0 if phi == "phi1" else 1

    configure_logging(angle_name)
    print_and_log(f"\nReading data from {directory_path}\n")

    files = sorted(
        [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".csv")
        ]
    )
    total_files = len(files)
    print_and_log(f"Number of input files: {total_files}")

    if total_files > subset_size:
        np.random.seed(random_seed)
        idx = np.random.choice(len(files), size=subset_size, replace=False)
        files = [files[i] for i in idx]

    # ------------------------------------------------------------
    # Load raw data once (independent of hkey) to avoid I/O duplication
    # ------------------------------------------------------------
    original_batches: List[np.ndarray] = []
    num_files = len(files)
    for start in range(0, num_files, batch_size):
        end = min(start + batch_size, num_files)
        batch_files = files[start:end]
        batch_data = load_batch(batch_files, col_idx=col_idx)

        print_and_log(
            f"Loading batch {start // batch_size + 1}/"
            f"{(num_files + batch_size - 1) // batch_size}"
        )
        original_batches.append(batch_data)

    original_data = np.vstack(original_batches)
    print_and_log(f"Original data shape: {original_data.shape}")

    # ------------------------------------------------------------
    # Global best trackers (across all tested hkeys)
    # ------------------------------------------------------------
    best_silhouette = -float("inf")
    best_db = float("inf")
    best_ch: Optional[float] = None
    best_n_pca: Optional[int] = None
    best_reduced_data: Optional[np.ndarray] = None
    best_clusters: Optional[np.ndarray] = None
    best_method: Optional[str] = None
    best_umap_params: Optional[Tuple[int, int, float]] = None
    best_model_params: Optional[Dict] = None
    best_hkey: Optional[int] = None

    # UMAP / PCA grids (preserve your commented ranges in the full scripts;
    # this simplified driver keeps your current active grid)
    umap_cache: Dict[Tuple[int, int, int, float], np.ndarray] = {}
    umap_grid = product(
        [20, 30, 40],
        [60, 70, 80],
        [0.000625, 0.00125],
    )
    n_pca_values = [2, 3, 4]

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

    # ------------------------------------------------------------
    # hkey loop: compare any subset/range of pipeline presets
    # ------------------------------------------------------------
    for hkey in hkeys_to_test:
        print_and_log(f"\n=== Testing hkey={hkey} ===")

        # Build features for this hkey (batch-wise, to preserve your original structure)
        feat_batches: List[np.ndarray] = []
        for bidx, batch_data in enumerate(original_batches, start=1):
            print_and_log(
                f"Extracting features (hkey={hkey}) for batch {bidx}/{len(original_batches)}"
            )
            batch_features = preprocess_batch(batch_data, hkey=hkey)
            print_and_log(f"Batch features shape: {batch_features.shape}")
            feat_batches.append(batch_features)

        all_features = np.vstack(feat_batches)
        print_and_log(f"All features (hkey={hkey}): {all_features.shape}")

        # Grid search DR + clustering
        for n_components, n_neighbors, min_dist in umap_grid:
            print_and_log(
                f"\nUMAP: components={n_components}, neighbors={n_neighbors}, min_dist={min_dist}"
            )

            umap_key = (hkey, n_components, n_neighbors, float(min_dist))
            if umap_key not in umap_cache:
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric="euclidean",
                    # random_state=random_seed,
                )
                umap_cache[umap_key] = reducer.fit_transform(all_features)

            reduced_umap = umap_cache[umap_key]

            for n_pca in n_pca_values:
                pca = PCA(n_components=n_pca)
                reduced_data = pca.fit_transform(reduced_umap)
                print_and_log(f"Number of PCA components: {n_pca}")

                for method_name, (grid_search_fn, param_grid) in clustering_methods.items():
                    print_and_log(f"Running grid search for {method_name}")
                    best_params, silhouette, db_index, ch_index = grid_search_fn(
                        reduced_data, param_grid, random_seed
                    )

                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_db = db_index
                        best_ch = ch_index
                        best_n_pca = n_pca
                        best_reduced_data = reduced_data
                        best_method = method_name
                        best_umap_params = (n_components, n_neighbors, min_dist)
                        best_model_params = best_params
                        best_hkey = hkey

                        model_cls = {
                            "K-Means": KMeans,
                            "Agglomerative": AgglomerativeClustering,
                            "GMM": GaussianMixture,
                        }[method_name]

                        # For scikit-learn models, some params are lists in the grid; grid_search_* returns scalars.
                        final_model = model_cls(**best_params)
                        best_clusters = final_model.fit_predict(reduced_data)

                print_and_log(
                    f"Current best: hkey={best_hkey}, method={best_method}, "
                    f"s={best_silhouette:.4f}, DB={best_db:.4f}, CH={float(best_ch):.4f}"
                )

    # ------------------------------------------------------------
    # Final best configuration (FULL pipeline, incl. best_hkey + model params)
    # ------------------------------------------------------------
    print_and_log("\nFinal best configuration:")
    print_and_log(f"Best hkey: {best_hkey}")
    print_and_log(f"Best method: {best_method}")
    print_and_log(f"Best model params: {best_model_params}")
    print_and_log(f"Best UMAP params: {best_umap_params}")
    print_and_log(f"Best PCA components: {best_n_pca}")
    print_and_log(f"Silhouette: {best_silhouette:.4f}")
    print_and_log(f"Davies-Bouldin: {best_db:.4f}")
    print_and_log(f"Calinski-Harabasz: {float(best_ch):.4f}")

    # Also ensure stdout shows it even if logging is redirected
    print("\nFinal best configuration:")
    print("Best hkey:", best_hkey)
    print("Best method:", best_method)
    print("Best model params:", best_model_params)
    print("Best UMAP params:", best_umap_params)
    print("Best PCA components:", best_n_pca)
    print(f"Silhouette: {best_silhouette:.4f}")
    print(f"Davies-Bouldin: {best_db:.4f}")
    print(f"Calinski-Harabasz: {float(best_ch):.4f}")

    if best_reduced_data is None or best_clusters is None or best_method is None:
        raise RuntimeError("No valid clustering configuration found; check grids and inputs.")

    angle_limits = (-180, 180) if phi == "phi1" else (0, 360)
    plot_clusters(
        best_reduced_data,
        best_clusters,
        best_method,
        angle_name,
        angle_limits,
        n_pca=best_n_pca,
    )
    plot_sample_series(original_data, best_clusters, files, best_method, angle_limits)

    # # Include hkey in outputs to avoid overwriting across runs
    # output_file = f"{best_method}_dynamic_map_{phi}_h{best_hkey}.csv"
    # create_dynamic_map(directory_path, output_file, labels_folder, best_clusters, files)

    # Subdir de labels específico do melhor hkey
    labels_folder_hkey = os.path.join(labels_folder, f"hkey_{best_hkey:02d}")
    os.makedirs(labels_folder_hkey, exist_ok=True)

    # Nome do mapa dinâmico com sufixo do melhor hkey
    output_file = f"{best_method}_dynamic_map_{phi}_hkey{best_hkey:02d}.csv"

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
    args = parser.parse_args()

    labels_dir = args.labels_dir or os.path.join(args.data_dir, f"labels_{args.phi}")
    hkeys_to_test = parse_hkeys(args.hkeys)

    main(
        phi=args.phi,
        directory_path=args.data_dir,
        labels_folder=labels_dir,
        hkeys_to_test=hkeys_to_test,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        random_seed=args.seed,
    )
