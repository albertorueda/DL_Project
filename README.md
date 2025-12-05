# DL_Project

## Overview
**DL_Project** is a project made in th context of the course 02456 Deep Learning. This github repository is made for preprocessing, training, and evaluating deep learning models (such as GRU and LSTM) on AIS (Automatic Identification System) maritime trajectory data. It loads raw AIS CSV files, cleans and preprocesses them, splits the data into train/validation/test sets, and trains sequence models to predict vessel trajectories.

---

## Project Structure
```
DL_Project/
├── data/ # Raw AIS CSV files (added by the user)
├── datasplits/ # Generated train/val/test splits
├── modules/
│ ├── ais_preprocessing.py # AIS cleaning pipeline
│ ├── ais_dataset.py # Automates preprocessing and splitting
│ ├── dataset.py # PyTorch dataset
│ ├── losses.py # Custom loss functions
│ ├── models.py # GRU/LSTM models in PyTorch
│ └── metrics.py # Metrics for evaluation
├── training/
│ ├── training_parameters.py # Main training loop
│ ├── finetuning_loop_batch_size.py # Finetune batch size
│ ├── finetuning_loop_dropout.py # Finetune dropout rate
│ ├── finetuning_loop_window_size.py # Finetune window size
│ ├── training_final_model.py # Final model training
│ └── training_final_model_more_days.py # Final model on larger dataset
├── evaluation/
│ ├── evaluate.py # Eval with 1 day of data
│ └── evaluate_more_data.py # Eval with 4 days of data
├── results/ # All experiment outputs
├── final-notebook.ipynb # User-friendly notebook
├── jobscript.sh # Cluster batch script
├── requirements.txt # Python dependencies
└── README.md # This file
```
---

## Getting Started
As an user of this github repository you can either: use the user-friendly notebook we made, or follow the following steps: 

### 1. Add AIS Data
Place AIS CSV files inside the `data/` folder.  
Adjust paths/filenames inside processing scripts if needed.

### 2. Preprocess the Dataset
Run the preprocessing pipeline:

`python ais_dataset.py`

This will:
- clean and filter AIS data  
- segment and interpolate trajectories  
- compute dataset statistics  
- create train/validation/test splits in `datasplits/`

### 3. Train a Model
Run:

`python training/training_final_model.py`

This script:
- loads the processed data  
- trains the GRU/LSTM architecture selected in `models.py`  
- saves checkpoints to `results/` using a naming scheme like:

model_loss_model_layers_embed_hidden.pth

A JSON file logging training losses is also created.

### 4. Evaluate a Model (Work in Progress)
Run:

`python evaluation/evaluate.py`

Intended metrics:
- MAE (Mean Absolute Error)  
- Final displacement error  
- Haversine distance  

The evaluation script currently serves as a template.

### 5. Optional: Use a Compute Cluster
Submit the batch script:

`bsub < jobscript.sh`

Checkpoints and logs appear in `results/` after completion.

---

## Requirements
Dependencies are listed in `requirements.txt`.  
This includes:
- Python  
- NumPy  
- Pandas  
- PyTorch  

---


## Authors
Pablo Baurier (s253159), <br>
Cecilia Biondini (s251965), <br>
Layla Doumont (s253007), <br>
Ignacio Ripoll (s242875), <br>
Alberto Rueda (s253057)
 
