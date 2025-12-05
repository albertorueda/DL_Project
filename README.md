# DL_Project

## Overview
**DL_Project** is a project made in th context of the course 02456 Deep Learning. This github repository is made for preprocessing, training, and evaluating deep learning models (such as GRU and LSTM) on AIS (Automatic Identification System) maritime trajectory data. It loads raw AIS CSV files, cleans and preprocesses them, splits the data into train/validation/test sets, and trains sequence models to predict vessel trajectories.

---

## Project Structure

DL_Project/  
  ├── data/                     # Raw AIS CSV files (added by the user)  
  ├── datasplits/               # Generated train/val/test splits  
  ├── modules/  
  │     ├── ais_preprocessing.py  # AIS cleaning pipeline  
  |     ├── ais_dataset.py        # Automates the preprocessing and splitting of AIS datasets
  │     ├── dataset.py            # Pytorch dataset
  │     ├── losses.py             # Custom losses functions
  │     └── models.py             # GRU/LSTM models in PyTorch  
  │     └── metrics.py            # Metrics for the evaluation
  ├── training/  
  │     └── training_big_loop.py           # Main training loop
  │     └── finetuning_loop_batch_size.py  # Main training loop  
  │     └── finetuning_loop_dropout.py     # Main training loop  
  │     └── finetuning_loop_window_size.py # Main training loop  
  │     └── training_final_model.py        # Main training loop  
  │     └── training_more_data.py          # Main training loop  
  ├── evaluation/  
        └── evaluate.py           # Evaluation script (template)  
  ├── results/                    # Model checkpoints + training logs  
  ├── final-notebook.ipynb        # Example end-to-end notebook  
  ├── jobscript.sh                # Cluster batch script  
  ├── requirements.txt            # Python dependencies  
  └── README.md                   # This file  

---

## Getting Started

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

`python training/training.py`

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

## What This Project Provides

- ✔ A full AIS data pipeline: raw CSV → cleaned → split → train → evaluate  
- ✔ Recurrent deep learning models (GRU & LSTM) for trajectory prediction  
- ✔ Automatic saving of model weights and training histories  
- ✔ Dataset statistics for understanding trajectory quality  

---

## Limitations / To-Do
- evaluation script needs completion  
- losses.py contains placeholders  
- no visualization tools (trajectory plots, histograms, etc.)  
- no hyperparameter configuration system  
- no license specified  

---

## Requirements
Dependencies are listed in `requirements.txt`.  
Core components include:
- Python  
- NumPy  
- Pandas  
- PyTorch  

---

## Contributing
Ideas for improvement:
- implement missing loss functions  
- complete the evaluation pipeline  
- add visualization utilities  
- introduce config files / CLI arguments  
- support multiple AIS datasets or automated ingestion  

---

## License
_No license is currently specified in the repository._
 
