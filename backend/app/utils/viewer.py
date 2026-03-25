import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import time
# commenting some of these out can make script run faster if you only want to call
#   specific functions

# print numpy arrays without truncation
# np.set_printoptions(threshold=sys.maxsize)

# global vars
# BASE_DIR = Path(__file__).resolve().parents[3] / "archive"
# DIR = str(BASE_DIR)
DIR = "/windows/Users/thats/Documents/archive"
 # folder where your dataset is
JSON_PATH = f"{DIR}/WLASL_v0.3.json"
VIDEO_DIR = f"{DIR}/videos/"  # folder with your video files
TRAIN_OUTPUT_DIR = f"{DIR}/train_output" # folder to save .npy feature files
TEST_OUTPUT_DIR = f"{DIR}/test_output" # folder to save .npy feature files
VALIDATION_OUTPUT_DIR = f"{DIR}/validation_output" # folder to save .npy feature files
TRAIN_OUTPUT_DIR_CLEANED = f"{DIR}/train_output_cleaned" # folder to save .npy feature files
TEST_OUTPUT_DIR_CLEANED = f"{DIR}/test_output_cleaned" # folder to save .npy feature files
VALIDATION_OUTPUT_DIR_CLEANED = f"{DIR}/validation_output_cleaned" # folder to save .npy feature files
TRAIN_OUTPUT_DIR_NORMALIZED = f"{DIR}/train_output_normalized"
TEST_OUTPUT_DIR_NORMALIZED = f"{DIR}/test_output_normalized"
VALIDATION_OUTPUT_DIR_NORMALIZED = f"{DIR}/validation_output_normalized"

def open_video_frames(file: str) :
    if not os.path.exists(file) :
        raise Exception(f"{file} does not exist")
    video = np.load(file)
    # Ctrl+c into terminal where program is running to kill the entire loop
    for frame in video :
        while 1 :
            cv2.imshow('', frame)
            k = cv2.waitKey(100)
            if k==27:    # Esc key to stop
                break
            elif k==-1:  # normally -1 returned,so don't print it
                continue
# open_video_frames(f"{DIR}/train_output_normalized_video/00947.npy")


def open_video(file: str) :
    if not os.path.exists(file) :
        raise Exception(f"{file} does not exist")
    video = np.load(file)
    # Ctrl+c into terminal where program is running to kill the entire loop
    for frame in video :
        cv2.imshow('', frame)
        k = cv2.waitKey(100)
        time.sleep(0.5)
# open_video(f"{DIR}/train_output_normalized_video/00947.npy")
