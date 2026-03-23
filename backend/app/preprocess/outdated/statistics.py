import numpy as np
import os

def extract_statistics_features(input_file:str):
    """ Extracts mean, std, min, max, velocity_mean, velocity_std from a features file
    Input: 2D array .npy file of n frames by 126 features
    Output: 1D array of 756 features of mean, std, min, max, velocity, velocity_mean, velocity_std for each coordinate"""

    features = np.load(input_file)
    # if (len(features) == 0):
    #     return -1
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    min = np.min(features, axis=0)
    max = np.max(features, axis=0)

    velocity = np.diff(features, axis=0)
    velocity_mean = np.mean(velocity, axis=0)
    velocity_std = np.std(velocity, axis=0)
    
    # print("mean: ", mean.shape, " ", mean)
    # print("std: ", std.shape, " ", std)
    # print("min: ", min.shape, " ", min)
    # print("max: ", max.shape, " ", max)
    # print("velocity: ", velocity.shape, " ", velocity)
    # print("velocity_mean: ", velocity_mean.shape, " ", velocity_mean)
    # print("velocity_std: ", velocity_std.shape, " ", velocity_std)

    statistics_features = np.concatenate([mean, std, min, max, velocity_mean, velocity_std])
    return statistics_features

def create_new_stat_features(input_dir:str, output_dir:str, overwrite:bool=False):
    """Takes an input of features and an output directory. Populates output directory with normalized features
    where each normalized feature file is a 1D array of 756 features
    input_dir: directory with generated features 
    output_dir: directory where processed files are saved"""
    
    for file in sorted(os.scandir(input_dir), key=lambda e: e.name):
        if file.name == "ordered_labels.npy":
            continue
        if file.is_file() and file.name.endswith(".npy"):
            out_path = os.path.join(output_dir, file.name)
            if os.path.exists(out_path) and not overwrite:
                continue
            
            np.save(out_path, extract_statistics_features(file.path))
