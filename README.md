# DL_Project

## Structure of the folders
- data
  - Place here the csv from the AIS web and change the name in the file data_processing.py. We should fix which files to use, how many days, ...
- modules
  - ```__init__.py```: just to allow Python to recognize this folder as module.
  - ```dataset.py```: file where are defined the methods to preprocess the dataset as well as the definition of the custom dataset to use in the DataLoader of PyTorch.
  - ```losses.py```: file where we will implement/add all the loss functions.
  - ```models.py```: file where we will define the models.
- ```data_processing.py```: script that generates three csv for the train/val/test split.
- ```evaluate.py```: we will implement the test loop to obtain the metrics.
- ```training.p```: we will implement here the training loop.
