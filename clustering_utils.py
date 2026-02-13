# clustering_utils.py
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.model_selection import ParameterGrid


def grid_search_kmeans(data, param_grid, random_seed):
    from sklearn.cluster import KMeans

    best_score = -1.0
    best_db = float('inf')
    best_params = {}
    best_ch_index = None

    for params in ParameterGrid(param_grid):
        model = KMeans(random_state=random_seed, **params)
        clusters = model.fit_predict(data)

        silhouette_avg = silhouette_score(data, clusters)
        db_index = davies_bouldin_score(data, clusters)
        ch_index = calinski_harabasz_score(data, clusters)

        if db_index < best_db:
            best_db = db_index
            best_score = silhouette_avg
            best_ch_index = ch_index
            best_params = params

    return best_params, best_score, best_db, best_ch_index


def grid_search_agglomerative(data, param_grid, random_seed):
    from sklearn.cluster import AgglomerativeClustering

    best_db = float('inf')
    best_score = -1.0
    best_params = {}
    best_ch_index = None

    for params in ParameterGrid(param_grid):
        model = AgglomerativeClustering(**params)
        clusters = model.fit_predict(data)

        silhouette_avg = silhouette_score(data, clusters)
        db_index = davies_bouldin_score(data, clusters)
        ch_index = calinski_harabasz_score(data, clusters)

        if db_index < best_db:
            best_db = db_index
            best_score = silhouette_avg
            best_ch_index = ch_index
            best_params = params

    return best_params, best_score, best_db, best_ch_index


def grid_search_gmm(data, param_grid, random_seed):
    from sklearn.mixture import GaussianMixture

    best_db = float('inf')
    best_score = -1.0
    best_params = {}
    best_ch_index = None

    for params in ParameterGrid(param_grid):
        model = GaussianMixture(random_state=random_seed, **params)
        clusters = model.fit_predict(data)

        silhouette_avg = silhouette_score(data, clusters)
        db_index = davies_bouldin_score(data, clusters)
        ch_index = calinski_harabasz_score(data, clusters)

        if db_index < best_db:
            best_db = db_index
            best_score = silhouette_avg
            best_ch_index = ch_index
            best_params = params

    return best_params, best_score, best_db, best_ch_index
