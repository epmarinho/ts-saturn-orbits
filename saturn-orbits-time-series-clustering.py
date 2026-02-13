# saturn-orbits-time-series-clustering.py
import argparse
import os
import numpy as np

from itertools import product
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import umap

from io_utils import (
    configure_logging,
    print_and_log,
    load_batch,
    create_dynamic_map,
)
from features import preprocess_batch
from clustering_utils_fixed import (
    grid_search_kmeans,
    grid_search_agglomerative,
    grid_search_gmm,
)
from plots import plot_clusters, plot_sample_series


def main(
    phi: str,
    directory_path: str,
    labels_folder: str,
    batch_size: int = 22288,
    subset_size: int = 22288,
    random_seed: int = 42,
):

    assert phi in ("phi1", "phi2")
    angle_name = phi
    col_idx = 0 if phi == "phi1" else 1

    configure_logging(angle_name)
    print_and_log(f"\nReading data from {directory_path}\n")

    files = sorted(
        [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith('.csv')
        ]
    )
    total_files = len(files)
    print_and_log(f"Number of input files: {total_files}")

    if total_files > subset_size:
        np.random.seed(random_seed)
        idx = np.random.choice(len(files), size=subset_size, replace=False)
        files = [files[i] for i in idx]

    all_features = []
    original_data = []
    num_files = len(files)

    for start in range(0, num_files, batch_size):
        end = min(start + batch_size, num_files)
        batch_files = files[start:end]
        batch_data = load_batch(batch_files, col_idx=col_idx)

        print_and_log(
            f"Processing batch {start // batch_size + 1}/"
            f"{(num_files + batch_size - 1) // batch_size}"
        )

        batch_features = preprocess_batch(batch_data, hkey=4)
        print_and_log(f"Batch features shape: {batch_features.shape}")

        all_features.append(batch_features)
        original_data.append(batch_data)

    all_features = np.vstack(all_features)
    original_data = np.vstack(original_data)
    print_and_log(f"All features: {all_features.shape}")
    print_and_log(f"Original data features: {original_data.shape}")

    best_silhouette = -float('inf')
    best_db = float('inf')
    best_ch = None
    best_n_pca = None
    best_reduced_data = None
    best_clusters = None
    best_method = None
    best_umap_params = None
    best_hkey = None

    umap_cache = {}
    umap_grid = product(
        [20, 30, 40],
        [60, 70, 80],
        [0.000625, 0.00125]
    )
    n_pca_values = [2, 3, 4]

    n_clusters = 4
    print_and_log(f"Number of clusters: {n_clusters}")

    kmeans_param_grid = {
        'n_clusters': [n_clusters],
        'init': ['k-means++', 'random'],
        'n_init': [6],
        'max_iter': [300],
    }

    agglomerative_param_grid = {
        'n_clusters': [n_clusters],
        'linkage': ['ward', 'complete', 'average', 'single'],
    }

    gmm_param_grid = {
        'n_components': [n_clusters],
        'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    }

    clustering_methods = {
        'K-Means': (grid_search_kmeans, kmeans_param_grid),
        'Agglomerative': (grid_search_agglomerative, agglomerative_param_grid),
        'GMM': (grid_search_gmm, gmm_param_grid),
    }

    for n_components, n_neighbors, min_dist in umap_grid:
        print_and_log(
            f"\nUMAP: components={n_components}, "
            f"neighbors={n_neighbors}, min_dist={min_dist}"
        )

        umap_key = (n_components, n_neighbors, min_dist)
        if umap_key not in umap_cache:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='euclidean',
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

                    final_model = {
                        'K-Means': KMeans,
                        'Agglomerative': AgglomerativeClustering,
                        'GMM': GaussianMixture,
                    }[method_name](**best_params)
                    best_clusters = final_model.fit_predict(reduced_data)

            print_and_log(
                f"Current best: method={best_method}, "
                f"s={best_silhouette:.4f}, DB={best_db:.4f}, CH={best_ch:.4f}"
            )

    print_and_log("\nFinal best configuration:")
    print_and_log(f"Best feature preset (hkey): {best_hkey}")
    print_and_log(f"Best method: {best_method}")
    print_and_log(f"Best UMAP params: {best_umap_params}")
    print_and_log(f"Best PCA components: {best_n_pca}")
    print_and_log(f"Silhouette: {best_silhouette:.4f}")
    print_and_log(f"Davies-Bouldin: {best_db:.4f}")
    print_and_log(f"Calinski-Harabasz: {best_ch:.4f}")

    angle_limits = (-180, 180) if phi == "phi1" else (0, 360)
    plot_clusters(best_reduced_data, best_clusters, best_method, angle_name, angle_limits, n_pca=best_n_pca)
    plot_sample_series(original_data, best_clusters, files, best_method, angle_limits)

    # output_file = f"{best_method}_dynamic_map_{phi}.csv"
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
    args = parser.parse_args()

    labels_dir = args.labels_dir or os.path.join(
        args.data_dir, f"labels_{args.phi}"
    )

    main(
        phi=args.phi,
        directory_path=args.data_dir,
        labels_folder=labels_dir,
    )
