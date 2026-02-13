# Saturn Orbits Time-Series Clustering (phi1)

This repository contains the exact script and dataset used to generate
Table 2 (phi1) in the manuscript:

"Clustering Astronomical Orbital Synthetic Data Using Advanced Feature Extraction and Dimensionality Reduction Techniques"

---

## Repository Contents

- `dict-tsf-fft-wvlt-pca-plot-lbl-rocket-phi1.py`  
  Main script performing the full benchmark grid search.

- `orbits-dataset/`  
  Contains 22,288 synthetic Saturn satellite orbital time series.

- `requirements.txt`  
  Python dependencies required to run the experiment.

- `environment.yml`
  To be used with conda (recomended)

---

## Dataset

Each file in `orbits-dataset/` corresponds to one trajectory and contains:

- First line: initial orbital parameters
- Remaining lines: 400 time-series samples

The dataset is included to ensure full reproducibility.

---

## Installation

We recommend using a virtual environment:

```bash

conda env create -f environment.yml
conda activate new_timeseries_env

