# io_utils.py
import os
import logging
import re
import pandas as pd


def configure_logging(angle_name: str):
    logging.basicConfig(
        filename=f'{angle_name}.log',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )


def print_and_log(msg: str):
    print(msg)
    logging.info(msg)


def read_file(file_path: str, col_idx: int):
    """
    Lê uma coluna específica (0 = phi1, 1 = phi2) de um arquivo de órbita.
    """
    try:
        return pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            usecols=[col_idx],
            skiprows=1,
            encoding='utf-8'
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            usecols=[col_idx],
            skiprows=1,
            encoding='latin1'
        )


def load_batch(files, col_idx: int):
    """
    Carrega um lote de arquivos CSV e retorna como array 2D (amostras x tempo).
    """
    import numpy as np

    data_list = []
    for file_path in files:
        data = read_file(file_path, col_idx)
        data_list.append(data.values.flatten())
    return np.array(data_list)


def create_dynamic_map(data_path, output_file, labels_folder, cluster_labels, files):
    """
    Cria o CSV de dynamic map e os arquivos de labels_* adicionando o índice de cluster
    na primeira linha de cada arquivo original.
    """
    import numpy as np

    print(f"Creating dynamic map with data path: {data_path}, output file: {output_file}, labels folder: {labels_folder}")

    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)
        print(f"Created labels folder: {labels_folder}")
    else:
        print(f"Labels folder already exists: {labels_folder}")

    dynamic_map_list = []

    for i, file in enumerate(files):
        with open(file, 'r') as f:
            first_line = f.readline().strip()
            columns = re.split(r'\s+', first_line)
            a, e = map(float, columns[:2])

            cluster_index = int(cluster_labels[i])

            columns.append(str(cluster_index))
            new_first_line = ','.join(columns)

            rest_of_file = f.read()

        label_file = os.path.join(labels_folder, os.path.basename(file))
        try:
            with open(label_file, 'w') as lf:
                lf.write(new_first_line + '\n')
                lf.write(rest_of_file)
        except IOError as err:
            print(f"Failed to write label file {label_file}: {err}")

        dynamic_map_list.append({
            'semimajor_axis': a,
            'eccentricity': e,
            'file_name': os.path.basename(file),
            'cluster_index': cluster_index,
        })

    dynamic_map = pd.DataFrame(dynamic_map_list)

    try:
        with open(output_file, 'w') as f:
            dynamic_map.to_csv(f, index=False, float_format='%.15e')
        print(f"Dynamic map saved to {output_file}")
        print(dynamic_map)
    except IOError as err:
        print(f"Failed to save dynamic map to {output_file}: {err}")
