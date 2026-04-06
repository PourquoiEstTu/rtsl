# TGCN Folder
This folder contains the Temporal Graph Convolutional Network (TGCN) implementation used by the backend for spatiotemporal forecasting on graph-structured data.

## What this folder is for
- Build and run the TGCN model.
- Prepare graph/time-series inputs for training and inference.
- Support training, evaluation, and prediction workflows.

## Files in this folder
Files here are organized as follows:

- **train_model.py**: Contains the python class definitions for the TGCN.
- **tgcn_wlasl.py**: Contains code for preprocessing the videos to be fed into the TGCN pipeline.
- **train_tgcn.py**: Contains the code for training, evaluating, and generating metrics on TGCN model.
- **train_utils.py**: Contains helper functions used in train_tgcn.py.
- **utils.py**: Contains other helper functions used throughout the files. 
- **convert_to_onnx.py**: Contains code to construct onnx files from binary models.  
- **configs.py**: Contains functions to read configs from  
- **pose_extractor.py**: Contains code to format the raw mediapipe keypoints into a well organized format of features
- **formatter.py**: Contains code used in main.py to format the raw mediapipe into our format (based on pose_extractor.py)
- **sign_dataset.py**: Contains code and classes from storing dataset frames and using within training.
