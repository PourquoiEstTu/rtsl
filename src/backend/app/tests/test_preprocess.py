import pytest
import numpy as np

from app.core.preprocess.workflow import extract_features, remove_zero_frames, interpolate_features

# Process a video into keypoints
def test_extract_features_00335():
    data = None
    data = extract_features("app/tests/sample_data/videos/00335.mp4")
    assert len(data) > 0

def test_extract_features_00421():
    data = None
    data = extract_features("app/tests/sample_data/videos/00421.mp4")
    assert len(data) > 0
    
def test_extract_features_00585():
    data = None
    data = extract_features("app/tests/sample_data/videos/00585.mp4")
    assert len(data) > 0

# Removing frames without keypoints 
def test_remove_zero_frames(tmp_path):
    remove_zero_frames("app/tests/sample_data/raw_keypoints", tmp_path)
    for file_name in ["00593.npy", "05728.npy", "43903.npy"]:
       test_file = tmp_path / file_name
       assert(test_file.exists)
       
       # gets before zero frames removed and after removal
       data = np.load("app/tests/sample_data/raw_keypoints/" + file_name)
       data_cleaned = np.load(test_file)
       
       # checks to see if zero frames have been removed and we still have data
       assert len(data_cleaned) < len(data)
       assert len(data_cleaned) > 0
    
def test_remove_zero_frames_nonexistent_file(tmp_path):
    test_file = tmp_path / "nonexistent.npy"
    remove_zero_frames("app/tests/sample_data/raw_keypoints", tmp_path)
    assert not test_file.exists()
    
# Interpolation Tests
def test_interpolate_features_dimension():
    data = [[0, 0, 0], [1, 1, 1]]
    np_data = np.array(data)
    interpolated_data = interpolate_features(np_data, 50)
    assert len(interpolated_data) == 50
    assert len(interpolated_data[0]) == 3
    
def test_interpolate_features_interpolation_middle():
    data = [[0, 0, 0], [1, 1, 1]]
    np_data = np.array(data)
    interpolated_data = interpolate_features(np_data, 3)
    assert interpolated_data[1][0] == 0.5

def test_interpolate_features_interpolation_ends():
    data = [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]
    np_data = np.array(data)
    interpolated_data = interpolate_features(np_data, 2)
    assert interpolated_data[0][0] == 0
    assert interpolated_data[1][0] == 1