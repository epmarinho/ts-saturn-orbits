# features.py
import numpy as np
import pandas as pd
import pywt

from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

from sktime.transformations.panel.rocket import MiniRocket

from io_utils import print_and_log


def apply_fft(series):
    return np.abs(np.fft.rfft(series))


def apply_wavelet(series, wavelet='db1', level=5, output_length=None):
    coeffs = pywt.wavedec(series, wavelet, level=level)
    coeffs_concat = np.concatenate(coeffs)

    if output_length is None:
        return coeffs_concat

    if len(coeffs_concat) < output_length:
        coeffs_concat = np.pad(
            coeffs_concat,
            (0, output_length - len(coeffs_concat)),
            'constant'
        )
    elif len(coeffs_concat) > output_length:
        coeffs_concat = coeffs_concat[:output_length]

    return coeffs_concat


def extract_relevant_tsfresh_features(data, predefined_columns=None):
    df = pd.DataFrame(data)
    df_long = df.stack().reset_index()
    df_long.columns = ['id', 'time', 'value']

    fc_parameters = EfficientFCParameters()
    print_and_log("EfficientFCParameters")

    relevant_features = extract_features(
        df_long,
        column_id='id',
        column_sort='time',
        default_fc_parameters=fc_parameters,
    )

    relevant_features = relevant_features.fillna(0)

    if predefined_columns is not None:
        for col in predefined_columns:
            if col not in relevant_features.columns:
                relevant_features[col] = 0
        relevant_features = relevant_features[predefined_columns]
    else:
        global common_columns
        common_columns = relevant_features.columns

    return relevant_features.values


def preprocess_batch(batch_data, hkey=4):

    print_and_log("StandardScaler")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(batch_data)

    # -------------------------
    # MiniRocket (sempre necessário para hkeys >=2 exceto 1,9,12,13,14,15)
    # -------------------------
    minirocket_features = None
    if hkey in (2,3,4,5,6,7,8,10):
        flattened = pd.DataFrame(batch_data)
        n_samples, n_timepoints = flattened.shape
        multiindex = pd.MultiIndex.from_product(
            [range(n_samples), range(n_timepoints)],
            names=["series", "time"]
        )
        new_df = pd.DataFrame(
            flattened.values.flatten(),
            index=multiindex,
            columns=['series_val']
        )

        minirocket = MiniRocket()
        minirocket.fit(new_df)
        data_transform = minirocket.transform(new_df)
        minirocket_features = scaler.fit_transform(np.array(data_transform))
        print_and_log(f"MiniROCKET features shape: {minirocket_features.shape}")

    # -------------------------
    # FFT
    # -------------------------
    fft_norm = None
    if hkey in (1,2,5,7,8,9,12,15):
        data_fft = np.array(Parallel(n_jobs=8)(
            delayed(apply_fft)(series) for series in data_normalized
        ))
        fft_norm = scaler.fit_transform(data_fft)
        print_and_log(f"FFT features shape: {fft_norm.shape}")

    # -------------------------
    # Wavelet
    # -------------------------
    wavelet_norm = None
    if hkey in (1,3,6,7,8,9,13,15):
        data_wavelet = np.array(Parallel(n_jobs=8)(
            delayed(apply_wavelet)(series) for series in data_normalized
        ))
        wavelet_norm = scaler.fit_transform(data_wavelet)
        print_and_log(f"Wavelet features shape: {wavelet_norm.shape}")

    # -------------------------
    # TSFresh
    # -------------------------
    tsfresh_norm = None
    if hkey in (1,4,5,6,8,11):
        tsfresh_features = extract_relevant_tsfresh_features(data_normalized)
        tsfresh_norm = scaler.fit_transform(tsfresh_features)
        print_and_log(f"TSFresh features shape: {tsfresh_norm.shape}")

    # -------------------------
    # Combine
    # -------------------------
    combined_hstack = {
        1: np.hstack([fft_norm, wavelet_norm, tsfresh_norm]),
        2: np.hstack([minirocket_features, fft_norm]),
        3: np.hstack([minirocket_features, wavelet_norm]),
        4: np.hstack([minirocket_features, tsfresh_norm]),
        5: np.hstack([minirocket_features, fft_norm, tsfresh_norm]),
        6: np.hstack([minirocket_features, wavelet_norm, tsfresh_norm]),
        7: np.hstack([minirocket_features, fft_norm, wavelet_norm]),
        8: np.hstack([minirocket_features, fft_norm, wavelet_norm, tsfresh_norm]),
        9: np.hstack([fft_norm, wavelet_norm, data_normalized]),
        10: minirocket_features,
        11: tsfresh_norm,
        12: fft_norm,
        13: wavelet_norm,
        14: data_normalized,
        15: np.hstack([fft_norm, wavelet_norm])
    }

    combined_features = combined_hstack[hkey]
    print(f"Combined features shape: {combined_features.shape}")

    return StandardScaler().fit_transform(combined_features)
