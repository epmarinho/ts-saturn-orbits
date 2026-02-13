import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import ParameterGrid
import hdbscan
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters, EfficientFCParameters
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter
import pywt
import re
import umap
# from utils import Visualizer
# from visdom import Visdom
from itertools import product
import logging
from sktime.transformations.panel.rocket import MiniRocket

def apply_minirocket(data):
    print("Applying MiniRocket")
    minirocket = MiniRocket()  # Initialize MINIROCKET
    minirocket.fit(data)  # Fit to data
    return minirocket.transform(data)  # Transform data to features

# Configure logging
logging.basicConfig(filename='phi2.log', level=logging.INFO)

def print_and_log(message):
    print(message)
    logging.info(message)

import stumpy
print(stumpy.__version__)

# Set logging level to WARNING
logging.getLogger('tornado.access').setLevel(logging.WARNING)

# # Initialize Visdom
# viz = Visualizer.Visualizer('Time-series Clustering', use_incoming_socket=False)

# Set the random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Define a TemporalBlock class for the Temporal Convolutional Network
# class TemporalBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         padding = (kernel_size - 1) * dilation
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)
#         self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
#         self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.net(x)
#         if self.downsample is not None:
#             res = self.downsample(x)
#         else:
#             res = x
#         out = out[:, :, :res.shape[2]]  # Ensure the dimensions match
#         return self.relu(out + res)
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.batch_norm1 = nn.BatchNorm1d(output_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.batch_norm2 = nn.BatchNorm1d(output_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                        dilation=dilation_size, padding=padding, dropout=dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# # Example usage:
# if __name__ == "__main__":
#     # Input dimensions: (batch_size, input_dim, sequence_length)
#     batch_size = 32
#     input_dim = 1  # Single channel (univariate time series)
#     sequence_length = 400

#     # Define TCN
#     num_channels = [32, 64, 128]  # Number of channels for each layer
#     kernel_size = 3
#     dropout = 0.2

#     model = TCN(input_dim=input_dim, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
#     print(model)

#     # Dummy input
#     x = torch.randn(batch_size, input_dim, sequence_length)
#     output = model(x)

#     # Output will have shape (batch_size, num_channels[-1], sequence_length)
#     print(f"Output shape: {output.shape}")

class TCNEncoderDecoder(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, latent_dim, dropout=0.2):
        super(TCNEncoderDecoder, self).__init__()
        self.encoder = TCN(input_dim, num_channels, kernel_size, dropout)
        self.latent = nn.Linear(num_channels[-1], latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, num_channels[-1])
        self.decoder_tcn = TCN(num_channels[-1], list(reversed(num_channels)), kernel_size, dropout)

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        encoded = torch.mean(encoded, dim=2)  # Global average pooling
        latent = self.latent(encoded)

        # Decode
        decoded = self.decoder_fc(latent)
        decoded = decoded.unsqueeze(-1).repeat(1, 1, x.size(2))  # Expand to match time dimension
        reconstructed = self.decoder_tcn(decoded)
        return latent, reconstructed

# # Loss function
# loss_fn = nn.MSELoss()

# Function to read a file with proper encoding handling
def read_file(file_path):
    try:
        return pd.read_csv(file_path, sep=r'\s+', header=None, usecols=[1], skiprows=1, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, sep=r'\s+', header=None, usecols=[1], skiprows=1, encoding='latin1')

# Function to load a batch of CSV files from a directory and return them as a numpy array
def load_batch(files):
    data_list = []
    for file_path in files:
        data = read_file(file_path)
        data_list.append(data.values.flatten())
    return np.array(data_list)

# Function to apply FFT transformation to a series
def apply_fft(series):
    return np.abs(np.fft.rfft(series))

# Function to apply wavelet transformation to a series
# def apply_wavelet(series):
#     coeffs = pywt.wavedec(series, 'db1', level=5)
#     return np.concatenate(coeffs)
def apply_wavelet(series, wavelet='db1', level=5, output_length=None):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    coeffs_concat = np.concatenate(coeffs)
    
    if output_length is None:
        return coeffs_concat
    
    if len(coeffs_concat) < output_length:
        # Pad the coefficients to the desired length
        coeffs_concat = np.pad(coeffs_concat, (0, output_length - len(coeffs_concat)), 'constant')
    elif len(coeffs_concat) > output_length:
        # Truncate the coefficients to the desired length
        coeffs_concat = coeffs_concat[:output_length]
    
    return coeffs_concat

# Function to extract relevant features using tsfresh
# def extract_relevant_tsfresh_features(data):
#     df = pd.DataFrame(data)
#     df_long = df.stack().reset_index()
#     df_long.columns = ['id', 'time', 'value']

#     # fc_parameters = MinimalFCParameters()
#     # fc_parameters = ComprehensiveFCParameters()
#     fc_parameters = EfficientFCParameters()
#     relevant_features = extract_features(df_long, column_id='id', column_sort='time', default_fc_parameters=fc_parameters)
#     relevant_features = relevant_features.fillna(0)
#     relevant_features = relevant_features.loc[:, (relevant_features != 0).any(axis=0)]
#     return relevant_features.values
def extract_relevant_tsfresh_features(data, predefined_columns=None):
    df = pd.DataFrame(data)
    df_long = df.stack().reset_index()
    df_long.columns = ['id', 'time', 'value']

    # fc_parameters = MinimalFCParameters()
    # print_and_log("MinimalFCParameters")
    # fc_parameters = ComprehensiveFCParameters()
    # print_and_log("ComprehensiveFCParameters")
    fc_parameters = EfficientFCParameters()
    print_and_log("EfficientFCParameters")
    relevant_features = extract_features(df_long, column_id='id', column_sort='time', default_fc_parameters=fc_parameters)
    relevant_features = relevant_features.fillna(0)
    
    if predefined_columns is not None:
        # Add missing columns with zeros
        for col in predefined_columns:
            if col not in relevant_features.columns:
                relevant_features[col] = 0
        relevant_features = relevant_features[predefined_columns]
    else:
        global common_columns
        common_columns = relevant_features.columns

    return relevant_features.values

def preprocess_batch(batch_data, hkey=1):
    # print_and_log("MinMax scaling")
    # scaler = MinMaxScaler()
    print_and_log("StandardScaler")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(batch_data)

    # Convert `batch_data` to a DataFrame where each row is a time series
    flattened_phi2 = pd.DataFrame(batch_data)  # `batch_data` loaded from your source
    # flattened_phi2 = pd.DataFrame(data_normalized)
    print("Flattened phi2 shape:", flattened_phi2.shape)

    # Create a MultiIndex for (series, time) to treat each row as an independent series
    n_samples, n_timepoints = flattened_phi2.shape
    multiindex = pd.MultiIndex.from_product([range(n_samples), range(n_timepoints)], names=["series", "time"])

    # Flatten data and create the new DataFrame with MultiIndex
    new_df = pd.DataFrame(flattened_phi2.values.flatten(), index=multiindex, columns=['phi2'])
    print("New DF with MultiIndex:", new_df.shape)

    # Initialize and apply MiniRocket on `new_df`
    minirocket = MiniRocket()
    minirocket.fit(new_df)  # Fit on the structured data
    data_transform = minirocket.transform(new_df)  # Transform to features

    # Convert to numpy array if necessary and check the shape
    minirocket_features = scaler.fit_transform(np.array(data_transform))
    print("MiniROCKET features shape:", minirocket_features.shape)

    # reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=0.1, metric='euclidean')
    
    # # print("Applying TCN model")
    # # num_channels = [32, 64, 128] # Original setting
    # num_channels = [64, 128, 256]
    # # num_channels = [32, 64, 128, 256] # Try with care regarding memory consumption
    # # num_channels = [8, 16, 32, 64, 128]
    # print_and_log(f"Number of TCN channels: {num_channels}")
    # tcn_model = TemporalConvNet(num_inputs=1, num_channels=num_channels, kernel_size=2, dropout=0.2)
    # tcn_model = tcn_model.float()

    # print("Adam optimizing and loss criterion")
    # optimizer = optim.Adam(tcn_model.parameters(), lr=0.001, weight_decay=1e-4)
    # # criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss(beta=1)
    
    # # # Set up the learning rate scheduler
    # # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # # Refined Cyclic Learning Rate Scheduler
    # lr_scheduler = CyclicLR(optimizer, base_lr=1.0e-6, max_lr=1.0e-2, step_size_up=10, mode='triangular2')
    
    # # Early stopping parameters
    # early_stopping_patience = 10
    # early_stopping_counter = 0
    # best_loss = float('inf')

    # num_epochs = 100
    # losses = []
    # train_data = torch.tensor(data_normalized[:, np.newaxis, :], dtype=torch.float32)
    
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     output = tcn_model(train_data)
    #     loss = criterion(output, train_data)
    #     loss.backward()

    #     # for name, param in tcn_model.named_parameters():
    #     #     if param.grad is not None:
    #     #         print(f'{name}: {param.grad.norm()}')

    #     optimizer.step()
        
    #     # Store the loss and print for each epoch
    #     losses.append(loss.item())
    #     current_lr = lr_scheduler.optimizer.param_groups[0]['lr']
    #     print_and_log(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, Learning Rate: {current_lr:.6e}")
    #     viz.plot_lines('Batch Loss', loss.item())

    #     # Step the LR scheduler
    #     # lr_scheduler.step(loss.item())
    #     lr_scheduler.step()
        
    #     # Early stopping check
    #     if loss.item() < best_loss:
    #         best_loss = loss.item()
    #         # print(f"Best loss by now: {best_loss}")
    #         early_stopping_counter += 1  # Reset counter if we find a new best
    #     else:
    #         early_stopping_counter = 0
        
    #     if early_stopping_counter >= early_stopping_patience:
    #         print_and_log(f"Early stopping triggered at epoch {epoch+1}")
    #         break

    # viz.reset_x_axis('Batch Loss')

    # print("Extracting TCN features")
    # with torch.no_grad():
    #     tcn_features = tcn_model(train_data).numpy()

    # # Flatten TCN features
    # print("Flattening TCN features")
    # tcn_features = tcn_features.reshape(tcn_features.shape[0], -1)
    # tcn_features_norm = scaler.fit_transform(tcn_features)

    print("FFT, Wavelet and TSFresh feature extraction")
    
    data_fft = np.array(Parallel(n_jobs=-1)(delayed(apply_fft)(series) for series in data_normalized))
    # data_fft = np.array(Parallel(n_jobs=-1)(delayed(apply_fft)(series) for series in tcn_features_norm))
    fft_norm = scaler.fit_transform(data_fft)
    # fft_umap = scaler.fit_transform(reducer.fit_transform(fft_norm))
    
    data_wavelet = np.array(Parallel(n_jobs=-1)(delayed(apply_wavelet)(series) for series in data_normalized))
    # data_wavelet = np.array(Parallel(n_jobs=-1)(delayed(apply_wavelet)(series, output_length=200) for series in tcn_features_norm))
    wavelet_norm = scaler.fit_transform(data_wavelet)
    # wavelet_umap = scaler.fit_transform(reducer.fit_transform(wavelet_norm))

    tsfresh_features = extract_relevant_tsfresh_features(data_normalized)
    # tsfresh_features = extract_relevant_tsfresh_features(tcn_features_norm)
    tsfresh_norm = scaler.fit_transform(tsfresh_features)

    # tcn_umap = scaler.fit_transform(reducer.fit_transform(tcn_features_norm))

    # data_umap = scaler.fit_transform(reducer.fit_transform(data_normalized))

    print_and_log(f"MiniROCKET features shape: {minirocket_features.shape}")
    print_and_log(f"FFT features shape:, {fft_norm.shape}")
    print_and_log(f"Wavelet features shape:, {wavelet_norm.shape}")
    print_and_log(f"TSFresh features shape:, {tsfresh_norm.shape}")

    # # Ensure all arrays have compatible shapes for hstack
    # print(f"Shapes before hstack - FFT: {data_fft.shape}, Wavelet: {data_wavelet.shape}, TSFresh: {tsfresh_features.shape}, TCN: {tcn_features.shape}, Data UMAP-reduced shape: {data_umap.shape}")
    
    # Combine features using hstack directly -- try one of the combinations (benchmarks performed for phi_2)
    # combined_features = np.hstack([fft_norm, wavelet_norm, tsfresh_norm, tcn_features_norm, data_umap]) # Silhouette Score: 0.6349
    # combined_features = np.hstack([fft_norm, wavelet_norm, tsfresh_norm, tcn_umap, data_umap]) # Silhouette Score: 0.5978. Bad dyn map
    # combined_features = np.hstack([fft_norm, wavelet_norm, tsfresh_norm, tcn_umap]) # Silhouette Score: 0.7323 (best by now)
    # combined_features = np.hstack([minirocket_features, fft_norm, wavelet_norm, tsfresh_norm, tcn_umap])
    # combined_features = np.hstack([minirocket_features, fft_norm, wavelet_norm, tsfresh_norm])
    # combined_features = np.hstack([minirocket_features, fft_norm, wavelet_norm])
    # combined_features = np.hstack([minirocket_features, wavelet_norm])
    # combined_features = np.hstack([minirocket_features, fft_norm]) # best after including mini rocket
    # combined_features = np.hstack([minirocket_features, fft_norm, wavelet_norm, tsfresh_norm])
    # combined_features = np.hstack([minirocket_features, fft_norm, tsfresh_norm])
    # combined_features = np.hstack([minirocket_features, tsfresh_norm]) # Previous working setting
    # combined_features = np.hstack([minirocket_features])
    # combined_features = np.hstack([fft_norm, wavelet_norm, tsfresh_norm, tcn_features_norm]) # Silhouette Score: 0.5436. Best map
    # combined_features = np.hstack([fft_norm, wavelet_norm, tsfresh_norm, data_umap]) # Silhouette Score: 0.7155, but terrible dynamic map
    # combined_features = np.hstack([fft_norm, wavelet_norm, tsfresh_norm]) # Silhouette Score: 0.6027. The best dynamic map
    # combined_features = np.hstack([fft_norm, wavelet_norm, data_umap]) # Silhouette index: 0.7146. However, dynamic map remains shit
    # combined_features = np.hstack([fft_norm, wavelet_norm, data_umap, data_normalized]) # Silhouette Score: 0.6885. Dynamic map is very weird
    # combined_features = np.hstack([tsfresh_norm, fft_norm, wavelet_norm, data_umap]) # Silhouette Score: 0.6989. Inaccurate dynamic map
    # combined_features = np.hstack([tsfresh_norm, fft_norm, wavelet_norm, tcn_umap, data_umap])
    # combined_features = np.hstack([tsfresh_norm, fft_norm, wavelet_umap, tcn_umap, data_umap])
    # combined_features = np.hstack([tsfresh_norm, fft_umap, wavelet_umap, tcn_umap, minirocket_features])
    # The following ones are the smallest feature representation: # none of them was good!
    # combined_features = data_umap
    # combined_features = tcn_features
    # combined_features = tsfresh_features # The worst representation
    # combined_features = data_fft
    # combined_features = data_wavelet

    # Dictionary of combinations with numerical keys
    combined_hstack = {
        1: np.hstack([fft_norm, wavelet_norm, tsfresh_norm]),
        2: np.hstack([minirocket_features, fft_norm]), # Best by now
        3: np.hstack([minirocket_features, wavelet_norm]),
        4: np.hstack([minirocket_features, tsfresh_norm]),
        5: np.hstack([minirocket_features, fft_norm, tsfresh_norm]),
        6: np.hstack([minirocket_features, wavelet_norm, tsfresh_norm]),
        7: np.hstack([minirocket_features, fft_norm, wavelet_norm]),
        8: np.hstack([minirocket_features, fft_norm, wavelet_norm, tsfresh_norm]),
        9: np.hstack([fft_norm, wavelet_norm, data_normalized]),
        10: minirocket_features,
        11: tsfresh_features,
        12: data_fft,
        13: data_wavelet,
    }
    combined_features = combined_hstack[hkey]

    # Check the combined shape to ensure consistency
    print(f"Combined features shape: {combined_features.shape}")

    scaler = StandardScaler()
    return scaler.fit_transform(combined_features)

# Function for dimensionality reduction using Incremental PCA
def incremental_pca(data, n_components=3, batch_size=500):
    print_and_log(f"Dimensionality reduction with Incremental PCA using {n_components} components")
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # First fit the IPCA on the data in batches
    for batch in np.array_split(data, len(data) // batch_size):
        ipca.partial_fit(batch)
    
    # Then transform the data in batches and collect the results
    reduced_data = []
    for batch in np.array_split(data, len(data) // batch_size):
        reduced_data.append(ipca.transform(batch))
    
    return np.vstack(reduced_data)

# Function to plot and save 2D or 3D cluster results
def plot_clusters(data, clusters, method_name, n_pca=3):
    if n_pca > 2 and data.shape[1] > 2:  # Ensure enough dimensions for 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            cluster_data = data[clusters == cluster]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], s=10, label=f'Cluster {cluster}')
        ax.set_title(f'Clusters found by {method_name}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.legend()
    else:  # For 2D data
        fig, ax = plt.subplots()
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            cluster_data = data[clusters == cluster]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], s=10, label=f'Cluster {cluster}')
        ax.set_title(f'Clusters found by {method_name}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        
    plt.savefig(f'{method_name}_clusters_phi_2.png')
    plt.close()

# # Function to plot and save 3D cluster results
# def plot_clusters(data, clusters, method_name):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     unique_clusters = np.unique(clusters)
#     for cluster in unique_clusters:
#         ax.scatter(data[clusters == cluster, 0], data[clusters == cluster, 1], data[clusters == cluster, 2], s=10, label=f'Cluster {cluster}')
#     ax.set_title(f'Clusters found by {method_name}')
#     ax.set_xlabel('PCA Component 1')
#     ax.set_ylabel('PCA Component 2')
#     ax.set_zlabel('PCA Component 3')
#     ax.legend()
#     plt.savefig(f'{method_name}_clusters.png')
#     plt.close()

# Function to plot and save sample series for each cluster
def plot_sample_series(data, clusters, files, method_name):
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        sample_indices = random.sample(list(cluster_indices), min(20, len(cluster_indices)))  # Choose up to 15 random samples
        num_pages = (len(sample_indices) + 4) // 5
        for page in range(num_pages):
            fig, axs = plt.subplots(5, 1, figsize=(10, 15))
            for i in range(5):
                idx = page * 5 + i
                if idx < len(sample_indices):
                    sample_idx = sample_indices[idx]
                    # print(f"Summary statistics for {files[sample_idx]}:")
                    # print(f"  Mean: {np.mean(data[sample_idx])}")
                    # print(f"  Std: {np.std(data[sample_idx])}")
                    # print(f"  Min: {np.min(data[sample_idx])}")
                    # print(f"  Max: {np.max(data[sample_idx])}")
                    
                    axs[i].plot(data[sample_idx], linewidth=0.5, color='blue')  # Use same linewidth and color as earlier version
                    filename = os.path.basename(files[sample_idx])
                    axs[i].set_title(f'{method_name} - Cluster {cluster} - {filename}')
                    axs[i].set_xlabel('Time')
                    axs[i].set_ylabel('Angle (degrees)')
                    # Add horizontal grey lines at 0 and +360 degrees
                    axs[i].axhline(y=0, color='grey', linestyle='--')
                    axs[i].axhline(y=360, color='grey', linestyle='--')

            plt.tight_layout()
            plt.savefig(f'{method_name}_cluster_{cluster}_page_{page+1}.png')
            plt.close()

# Function to evaluate clustering using various metrics
def evaluate_clustering(data, clusters):
    silhouette_avg = silhouette_score(data, clusters)  # Silhouette score
    db_index = davies_bouldin_score(data, clusters)  # Davies-Bouldin index
    ch_index = calinski_harabasz_score(data, clusters)  # Calinski-Harabasz index
    return silhouette_avg, db_index, ch_index

# Function to perform grid search for K-Means with verbose output
def grid_search_kmeans(data, param_grid):
    best_score = -1
    best_db = float('infinity')
    best_params = {}
    for params in ParameterGrid(param_grid):
        # print("KMeans param: ", params)
        model = KMeans(random_state=random_seed, **params)
        clusters = model.fit_predict(data)
        silhouette_avg = silhouette_score(data, clusters)
        # print("Silhoueatte: ", silhouette_avg)
        db_index = davies_bouldin_score(data, clusters)  # Davies-Bouldin index
        # print("Davies-Bouldin: ", db_index)
        ch_index = calinski_harabasz_score(data, clusters)  # Calinski-Harabasz index
        # print("Chalinski-Harabaz: ", ch_index)
        
        # if silhouette_avg > best_score:
        if db_index < best_db:
            best_db = db_index
            best_score = silhouette_avg
            best_ch_index = ch_index
            best_params = params
            # print_and_log(f"New best score: {best_score}, {db_index}, {ch_index} with params: {best_params}")
    
    return best_params, best_score, best_db, best_ch_index

# # Function to perform grid search for K-Means using DB index minimization
# def grid_search_kmeans(data, param_grid):
#     best_db_index = float('inf')  # Initialize with a large value
#     best_params = {}
#     for params in ParameterGrid(param_grid):
#         model = KMeans(random_state=random_seed, **params)
#         clusters = model.fit_predict(data)
        
#         # Calculate the Davies-Bouldin index
#         db_index = davies_bouldin_score(data, clusters)
#         ch_index = calinski_harabasz_score(data, clusters)
#         silhouette = silhouette_score(data, clusters)
        
#         # Check if the current DB index is lower than the best found so far
#         if db_index < best_db_index:
#             best_db_index = db_index
#             best_params = params
#             print(f"New best DB index: {best_db_index} with silhouette = {silhouette}, ch_index, and params: {best_params}")
    
#     return best_params, silhouette, best_db_index, ch_index

# Function to perform grid search for DBSCAN
def grid_search_dbscan(data, param_grid):
    best_db = float('infinity')
    best_score = -1
    best_params = {}
    for params in ParameterGrid(param_grid):
        model = DBSCAN(**params)
        clusters = model.fit_predict(data)
        if len(set(clusters)) <= 1:  # Avoid cases where only one cluster is found
            continue
        silhouette_avg = silhouette_score(data, clusters)
        # print("Silhoueatte: ", silhouette_avg)
        db_index = davies_bouldin_score(data, clusters)  # Davies-Bouldin index
        # print("Davies-Bouldin: ", db_index)
        ch_index = calinski_harabasz_score(data, clusters)  # Calinski-Harabasz index
        # print("Chalinski-Harabaz: ", ch_index)
        # if silhouette_avg > best_score:
        if db_index < best_db:
            best_db = db_index
            best_score = silhouette_avg
            best_ch_index = ch_index
            best_params = params
            # print_and_log(f"New best score: {best_score}, {db_index}, {ch_index} with params: {best_params}")            
            # print_and_log(f"New best score: {best_score} with params: {best_params}")
    return best_params, silhouette_avg, best_db, best_ch_index

# Function to perform grid search for Agglomerative Clustering
def grid_search_agglomerative(data, param_grid):
    best_db = float('infinity')
    best_score = -1
    best_params = {}
    for params in ParameterGrid(param_grid):
        model = AgglomerativeClustering(**params)
        clusters = model.fit_predict(data)
        silhouette_avg = silhouette_score(data, clusters)
        # print("Silhoueatte: ", silhouette_avg)
        db_index = davies_bouldin_score(data, clusters)  # Davies-Bouldin index
        # print("Davies-Bouldin: ", db_index)
        ch_index = calinski_harabasz_score(data, clusters)  # Calinski-Harabasz index
        # print("Chalinski-Harabaz: ", ch_index)
        # if silhouette_avg > best_score:
        if db_index < best_db:
            best_db = db_index
            best_score = silhouette_avg
            best_ch_index = ch_index
            best_params = params
            # print_and_log(f"New best score: {best_score}, {db_index}, {ch_index} with params: {best_params}")
    return best_params, best_score, best_db, best_ch_index

# Function to perform grid search for Gaussian Mixture Models
def grid_search_gmm(data, param_grid):
    best_db = float('infinity')
    best_score = -1
    best_params = {}
    for params in ParameterGrid(param_grid):
        model = GaussianMixture(random_state=random_seed, **params)
        clusters = model.fit_predict(data)
        silhouette_avg = silhouette_score(data, clusters)
        db_index = davies_bouldin_score(data, clusters)  # Davies-Bouldin index
        # print("Davies-Bouldin: ", db_index)
        ch_index = calinski_harabasz_score(data, clusters)  # Calinski-Harabasz index
        # print("Chalinski-Harabaz: ", ch_index)
        # if silhouette_avg > best_score:
        if db_index < best_db:
            best_db = db_index
            best_score = silhouette_avg
            best_ch_index = ch_index
            best_params = params
            # print_and_log(f"New best score: {best_score}, {db_index}, {ch_index} with params: {best_params}")
    return best_params, best_score, best_db, best_ch_index

# Function to perform grid search for HDBSCAN
def grid_search_hdbscan(data, param_grid):
    best_db = float('infinity')
    best_score = -1
    best_params = {}
    for params in ParameterGrid(param_grid):
        model = hdbscan.HDBSCAN(**params)
        clusters = model.fit_predict(data)
        if len(set(clusters)) <= 1:  # Avoid cases where only one cluster is found
            continue
        silhouette_avg = silhouette_score(data, clusters)
        # print("Silhoueatte: ", silhouette_avg)
        db_index = davies_bouldin_score(data, clusters)  # Davies-Bouldin index
        # print("Davies-Bouldin: ", db_index)
        ch_index = calinski_harabasz_score(data, clusters)  # Calinski-Harabasz index
        # print("Chalinski-Harabaz: ", ch_index)
        # if silhouette_avg > best_score:
        if db_index < best_db:
            best_db = db_index
            best_score = silhouette_avg
            best_ch_index = ch_index
            best_params = params
            # print_and_log(f"New best score: {best_score}, {db_index}, {ch_index} with params: {best_params}")
    return best_params, best_score, best_db, best_ch_index

# Function to create dynamic map
def create_dynamic_map(data_path, output_file, labels_folder, cluster_labels, files):
    print(f"Creating dynamic map with data path: {data_path}, output file: {output_file}, labels folder: {labels_folder}")

    # Ensure the labels_folder exists
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
        print(f"Created labels folder: {labels_folder}")
    else:
        print(f"Labels folder already exists: {labels_folder}")
    
    # Create an empty DataFrame to store the results
    dynamic_map_list = []
    
    # Collect the first line data and iterate over each file
    for i, file in enumerate(files):
        with open(file, 'r') as f:
            first_line = f.readline().strip()
            columns = re.split(r'\s+', first_line)
            a, e = map(float, columns[:2])
        
            # Get the cluster index
            cluster_index = cluster_labels[i]
        
            # Add cluster index to the first line
            columns.append(str(cluster_index))
            new_first_line = ','.join(columns)
        
            # Read the rest of the file
            rest_of_file = f.read()
        
        # Write the new first line and the rest of the file to the label file
        label_file = os.path.join(labels_folder, os.path.basename(file))
        try:
            with open(label_file, 'w') as lf:
                lf.write(new_first_line + '\n')
                lf.write(rest_of_file)
        except IOError as e:
            print(f"Failed to write label file {label_file}: {e}")
        
        # Append the data to the list for DataFrame creation
        dynamic_map_list.append({
            'semimajor_axis': a,
            'eccentricity': e,
            'file_name': os.path.basename(file),
            'cluster_index': cluster_index
        })
    
    # Create the DataFrame from the list
    dynamic_map = pd.DataFrame(dynamic_map_list)
    
    # Save the DataFrame to a CSV file using explicit file handling with extended precision
    try:
        with open(output_file, 'w') as f:
            dynamic_map.to_csv(f, index=False, float_format='%.15e')
            print(f"Dynamic map saved to {output_file}")
        print(dynamic_map)  # Print the dynamic map to verify content
    except IOError as e:
        print(f"Failed to save dynamic map to {output_file}: {e}")

# Main function to execute the entire pipeline
def main(directory_path, labels_folder, batch_size=22288, subset_size=22288):
    print_and_log(f"\nReading data from {directory_path}\n")
    output_file = 'dynamic_map_phi_2.csv'
    files = sorted([os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')])

    total_files = len(files)
    print_and_log(f"Number of input files: {total_files}")
    if total_files > subset_size:
        # Randomly select a subset of files
        np.random.seed(random_seed)
        selected_indices = np.random.choice(len(files), size=subset_size, replace=False)
        files = [files[i] for i in selected_indices]

    combined_features = {
        1: "np.hstack([fft_norm, wavelet_norm, tsfresh_norm])",
        2: "np.hstack([minirocket_features, fft_norm])", 
        3: "np.hstack([minirocket_features, wavelet_norm])",
        4: "np.hstack([minirocket_features, tsfresh_norm])", # Best by now
        5: "np.hstack([minirocket_features, fft_norm, tsfresh_norm])",
        6: "np.hstack([minirocket_features, wavelet_norm, tsfresh_norm])",
        7: "np.hstack([minirocket_features, fft_norm, wavelet_norm])",
        8: "np.hstack([minirocket_features, fft_norm, wavelet_norm, tsfresh_norm])",
        9: "np.hstack([fft_norm, wavelet_norm, data_normalized])",
        10: "minirocket_features",
        11: "tsfresh_features",
        12: "fft_norm",
        13: "wavelet_norm",
    }
    # print("Combined features dictionary: ", combined_features)

    best_n_pca = None
    best_silhouette_avg = -float('inf')
    best_db_index = float('inf')
    best_reduced_data = None
    best_clusters = None
    best_method = None
    best_umap_components = None
    best_n_neighbs = None
    best_min_dist = None
    best_best_params = None
    best_hkey = None

    not_yet = True
    for hkey in range(1, 13+1): # range(1, len(combined_features)+1):
        print_and_log(f"\nCurrent combined features: {hkey}: {combined_features[hkey]}")

        # This portion is used to extract all feature vectors to be ulteriorly used for clustering
        if not_yet:
            all_features = []
            original_data = []
            num_files = len(files)
            for start in range(0, num_files, batch_size):
                end = min(start + batch_size, num_files)
                batch_files = files[start:end]
                batch_data = load_batch(batch_files)
                print_and_log(f"Processing batch {start // batch_size + 1}/{(num_files + batch_size - 1) // batch_size}")
                batch_features = preprocess_batch(batch_data, hkey=hkey)
                
                # Print shape for debugging
                print_and_log(f"Batch features shape: {batch_features.shape}")
                
                all_features.append(batch_features)
                original_data.append(batch_data)

            all_features = np.vstack(all_features)
            print_and_log(f"All features: {all_features.shape}")
            original_data = np.vstack(original_data)  # Stack all batches to get the complete original dataset
            print_and_log(f"Original data features: {original_data.shape}")
            not_yet = False

        # for umap_components, n_neighbors, min_dist in product([100, 304, 400], [100, 206, 400], [0.0025, 0.005, 0.01]):
        # for umap_components, n_neighbors, min_dist in product(range(100, 210, 10), range(100, 210, 10), [0.0025, 0.005, 0.01]):
        # for umap_components, n_neighbors, min_dist in product(range(100, 140, 20), range(50, 80, 10), [0.0025, 0.005]):
        # for umap_components, n_neighbors, min_dist in product([45], [61], [0.0025]):
        # for umap_components, n_neighbors, min_dist in product([40], [120], [0.0025]):
        # for umap_components, n_neighbors, min_dist in product([20, 30, 40], [60, 70, 80], [0.000625, 0.00125]):
        for umap_components, n_neighbors, min_dist in product(range(50, 160, 10), range(10, 80, 10), [0.000625]):
            print_and_log(f'\numap_components: {umap_components}, n_neighbors: {n_neighbors}, min_dist: {min_dist}')

            n_pca_values = [2, 3]
            for n_pca in n_pca_values:
                reducer = umap.UMAP(n_components=umap_components, n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean')
                # reducer = umap.UMAP(n_components=umap_components, n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine')

                reduced_data = reducer.fit_transform(all_features)
                # reduced_data = incremental_pca(all_features, n_components=n_pca)  # Reduce dimensionality using Incremental PCA
                
                pca = PCA(n_components=n_pca)
                # reduced_data = pca.fit_transform(all_features)
                reduced_data = pca.fit_transform(reduced_data)
                # print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()}")
                # # Cumulative explained variance
                # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

                # # Find the number of components for 90% explained variance
                # n_pca = np.argmax(cumulative_variance >= 0.9) + 1
                # print(f"Number of components for 90% explained variance: {n_pca}")
                print_and_log(f"Number of PCA components: {n_pca}")

                n_clusters = 4

                # Grid search parameters for each method
                print_and_log(f"Number of clusters: {n_clusters}")
                kmeans_param_grid = {
                    'n_clusters': [n_clusters],
                    'init': ['k-means++', 'random'],
                    'n_init': [6],
                    'max_iter': [3]
                }

                agglomerative_param_grid = {
                    'n_clusters': [n_clusters],
                    'linkage': ['ward', 'complete', 'average', 'single']
                }

                gmm_param_grid = {
                    'n_components': [n_clusters],
                    'covariance_type': ['full', 'tied', 'diag', 'spherical']
                }

                dbscan_param_grid = {
                    'eps': np.arange(0.1, 4, 0.1), # fine-grained search for eps
                    'min_samples': range(2, 10), # typical values for min_samples
                    'metric': ['euclidean', 'manhattan'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'] # ensure all algorithm options are included but 'brute'
                }

                hdbscan_param_grid = {
                    'min_cluster_size': range(4, 20, 4), # typical range for min_cluster_size
                    'min_samples': range(10, 50, 10), # range for min_samples
                    'metric': ['manhattan', 'euclidean', 'braycurtis', 'correlation'] # appropriate metrics for HDBSCAN
                }

                clustering_methods = {
                    'K-Means': (KMeans, kmeans_param_grid, grid_search_kmeans),
                    'Agglomerative': (AgglomerativeClustering, agglomerative_param_grid, grid_search_agglomerative),
                    'GMM': (GaussianMixture, gmm_param_grid, grid_search_gmm),
                    'DBSCAN': (DBSCAN, dbscan_param_grid, grid_search_dbscan),
                    'HDBSCAN': (hdbscan.HDBSCAN, hdbscan_param_grid, grid_search_hdbscan)
                }

                for method_name, (method, param_grid, grid_search) in clustering_methods.items():
                    print_and_log(f"Running grid search for {method_name}")
                    best_params, silhouette, db_index, ch_index = grid_search(reduced_data, param_grid)

                    # print_and_log(f"Best parameters for {method_name} with {best_params}")
                    model = method(**best_params)
                    clusters = model.fit_predict(reduced_data)
                    # silhouette_avg, db_index, ch_index = evaluate_clustering(reduced_data, clusters)
                    # print(f"{method_name} with n_pca={n_pca} -> Silhouette Score: {silhouette_avg:.4f}, Davies-Bouldin Index: {db_index:.4f}, Calinski-Harabasz Index: {ch_index:.4f}")
                    # print_and_log(f"{method_name} with n_pca={n_pca} -> Silhouette Score: {silhouette:.4f}, Davies-Bouldin Index: {db_index:.4f}, Calinski-Harabasz Index: {ch_index:.4f}")

                    if silhouette > best_silhouette_avg:
                    # if db_index < best_db_index:
                        best_silhouette_avg = silhouette
                        best_db_index = db_index
                        best_ch_index = ch_index
                        best_n_pca = n_pca
                        best_reduced_data = reduced_data
                        best_clusters = clusters
                        best_method = method_name
                        best_umap_components = umap_components
                        best_n_neighbs = n_neighbors
                        best_min_dist = min_dist
                        best_best_params = best_params
                        best_hkey = hkey

                        print_and_log("\nCurrent results:")
                        print_and_log(f"Best UMAP Params: components={best_umap_components}, neighbors={best_n_neighbs}, min dist={best_min_dist},")
                        print_and_log(f"Best PCA components: {best_n_pca}")
                        print_and_log(f"Best Metrics: Silhouette={best_silhouette_avg}, DB={best_db_index}, CH={best_ch_index}")
                        print_and_log(f"Best Feature Combination: {hkey}: {combined_features[best_hkey]}")
                        print_and_log(f"Best Clustering: {best_method}\n")

    # Plotting and saving the results for the best method
    plot_clusters(best_reduced_data, best_clusters, best_method, n_pca=n_pca)  # Plot and save clusters
    # plot_clusters(best_reduced_data, best_clusters, best_method, n_pca=best_umap_components)  # Plot and save clusters
    plot_sample_series(original_data, best_clusters, files, best_method)  # Plot and save sample series

    # Perform clustering and create the dynamic map
    output_file = f"{best_method}_dynamic_map_phi_2.csv"
    create_dynamic_map(directory_path, output_file, labels_folder, best_clusters, files)

    # # === SAVE: EMBEDDING ÓTIMO DO PIPELINE (φ2) PARA ANÁLISE ORM ===
    # best_embedding_path = os.path.join(labels_folder, "phi2_best_embedding.npy")
    # best_filenames_path = os.path.join(labels_folder, "phi2_best_filenames.npy")

    # np.save(best_embedding_path, best_reduced_data)
    # np.save(best_filenames_path, np.array([os.path.basename(f) for f in files]))

    # print_and_log(f"[SAVE] Best embedding saved to: {best_embedding_path}")
    # print_and_log(f"[SAVE] Filenames saved to: {best_filenames_path}")

    # Evaluate and print clustering results for each method
    # silhouette_avg, db_index, ch_index = evaluate_clustering(best_reduced_data, best_clusters)  # Evaluate clustering
    # best_silhouette_avg, db_index, ch_index = evaluate_clustering(best_reduced_data, best_clusters)
    unique, counts = np.unique(best_clusters, return_counts=True)  # Get cluster counts
    cluster_counts = dict(zip(unique, counts))  # Create dictionary of cluster counts

    # total_samples = sum(counts)  # Total number of samples
    print_and_log("\n")
    print_and_log(f"Best H-Stack combination: {best_hkey}: {combined_features[best_hkey]}")
    print_and_log(f"Best method: {best_method}")
    print_and_log(f"Best parameters {best_best_params}")
    print_and_log(f"Best number of UMAP components: {best_umap_components}")
    print_and_log(f"Best number of neighbors: {best_n_neighbs}")
    print_and_log(f"Best minimum distance: {best_min_dist}")
    print_and_log(f"Best number of PCA components: {best_n_pca}")
    print_and_log(f"Silhouette Score: {best_silhouette_avg:.4f}")
    print_and_log(f"Davies-Bouldin Index: {best_db_index:.4f}")
    print_and_log(f"Calinski-Harabasz Index: {best_ch_index:.4f}")
    print_and_log(f"Cluster Counts: {cluster_counts}\n")
    # print_and_log(f"Checksum: {total_samples} (Expected: {len(best_reduced_data)})\n")

# Run the main function
if __name__ == "__main__":
    # main("./saturn-orbits/origin", "./saturn-orbits/labels_phi_2")
    main("./orbits-dataset", "./orbits-dataset/labels_phi_2")
