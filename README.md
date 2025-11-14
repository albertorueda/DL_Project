# DL_Project

## Project Structure

- **data/**  
  - Place here the CSV files from the AIS web and configure the desired filename in `ais_dataset.py`.
  - Download raw AIS data from: http://aisdata.ais.dk/?prefix=

- **modules/**
  - `__init__.py`: Makes this folder a Python module.
  - `ais_preprocessing.py`: Contains full AIS data cleaning pipeline (filtering, segmentation, interpolation).
  - `describe_dataset.py`: Computes descriptive statistics (e.g., speed, gap analysis, segment lengths).
  - `losses.py`: Custom loss functions (to be implemented).
  - `models.py`: GRU, LSTM, and other model architectures.

- **ais_dataset.py**  
  - Main script to load raw AIS CSV, clean and preprocess it, split into train/val/test, and save resulting CSVs.
  - Also triggers dataset metric generation using `describe_dataset.py`.

- **evaluate.py**  
  - Evaluation script to test model performance (e.g., MAE, FDE, Haversine). *(To be implemented)*

- **training.py**  
  - Training loop using PyTorch. Loads train/val sets and runs model training.

- **README.md**  
  - Current documentation file (this one).
