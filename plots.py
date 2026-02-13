# plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
import random


def plot_clusters(data, clusters, method_name, angle_name, angle_limits, n_pca=3):
    """
    Plota clusters em 2D ou 3D, salva como <method_name>_clusters_<phi>.png.
    """
    if n_pca > 2 and data.shape[1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            cluster_data = data[clusters == cluster]
            ax.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                cluster_data[:, 2],
                s=10,
                label=f'Cluster {cluster}'
            )

        ax.set_title(f'Clusters found by {method_name}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend()
    else:
        fig, ax = plt.subplots()
        unique_clusters = np.unique(clusters)

        for cluster in unique_clusters:
            cluster_data = data[clusters == cluster]
            ax.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                s=10,
                label=f'Cluster {cluster}'
            )

        ax.set_title(f'Clusters found by {method_name}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()

    plt.tight_layout()
    plt.savefig(f'{method_name}_clusters_{angle_name}.png')
    plt.close()


def plot_sample_series(data, clusters, files, method_name, angle_limits):
    """
    Plota amostras de séries por cluster, com linhas horizontais em angle_limits.
    angle_limits = (-180, 180) para phi1, (0, 360) para phi2, etc.
    """
    ymin, ymax = angle_limits
    unique_clusters = np.unique(clusters)

    for cluster in unique_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) == 0:
            continue

        sample_indices = random.sample(
            list(cluster_indices),
            min(20, len(cluster_indices))
        )

        num_pages = (len(sample_indices) + 4) // 5

        for page in range(num_pages):
            fig, axs = plt.subplots(5, 1, figsize=(10, 15))

            for i in range(5):
                idx = page * 5 + i
                if idx >= len(sample_indices):
                    break

                sample_idx = sample_indices[idx]
                axs[i].plot(
                    data[sample_idx],
                    linewidth=0.5,
                    color='blue'
                )
                filename = os.path.basename(files[sample_idx])
                axs[i].set_title(f'{method_name} - Cluster {cluster} - {filename}')
                axs[i].set_xlabel('Time')
                axs[i].set_ylabel('Angle (degrees)')

                axs[i].axhline(y=ymin, color='grey', linestyle='--')
                axs[i].axhline(y=ymax, color='grey', linestyle='--')

            plt.tight_layout()
            plt.savefig(f'{method_name}_cluster_{cluster}_page_{page+1}.png')
            plt.close()
