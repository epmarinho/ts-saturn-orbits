# Saturn Orbits Time-Series Clustering

This repository contains the complete code and dataset used to generate the results reported in Table 2 (phi1) of the manuscript:

"Clustering Astronomical Orbital Synthetic Data Using Advanced Feature Extraction and Dimensionality Reduction Techniques"

---

## Repository Structure

ts-saturn-orbits/
├── dict-tsf-fft-wvlt-pca-plot-lbl-rocket-phi1.py
├── features.py
├── io_utils.py
├── clustering_utils.py
├── plots.py
├── orbits-dataset/
└── requirements.txt

---

## Dataset

The `orbits-dataset/` directory contains 22,288 synthetic Saturn satellite orbital time series.

Each file corresponds to one trajectory and contains:
- First line: initial orbital parameters
- Remaining lines: time-series values (400 samples)

The dataset is included in this repository to ensure full reproducibility.

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt

