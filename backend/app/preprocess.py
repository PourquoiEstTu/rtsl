import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# print numpy arrays without truncation
np.set_printoptions(threshold=sys.maxsize)

# global vars
# BASE_DIR = Path(__file__).resolve().parents[3] / "archive"
DIR = "/u50/quyumr/rtsl-features" #str(BASE_DIR)
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
FLATTENED_OUTPUT_DIR = f"{DIR}/flattened_outputs"

os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR_NORMALIZED, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR_NORMALIZED, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_DIR_NORMALIZED, exist_ok=True)

# INITIALIZE MEDIAPIPE HOLISTIC
# essentially uses the mediapipe holistic model to extract hands and pose features
# comment out if not needed when running this file
# mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic(
#     static_image_mode=False,
#     model_complexity=1,
#     smooth_landmarks=True,
#     enable_segmentation=False, # mediapipe crashes when true? 
#         # someone else run this file with this and refine_face_landmarks=True as well
#     refine_face_landmarks=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# extract features from a video
def extract_features(video_path: str):
    """Output is an array of the 21 landmarks on both hands
       With dimensions a x b x c. a represents the number of 
       frames in the video. b represents the two hands, first 
       one is right hand, second one is left hand. c represents
       the keypoints, each group of 3 represents the x, y, z 
       coordinates of the keypoint in the image."""
    cap = cv2.VideoCapture(video_path)
    sequence = []

    # read frames from video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # convert the BGR image to RGB before processing?
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if we want to de-normalize landmark points later, we can
        #   multiply by x by width and y by height
        # img_height, img_width, _ = frame_rgb.shape
        results = holistic.process(frame_rgb) # process the frame

        # HANDS
        hand_keypoints = []
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                hand_keypoints.extend([lm.x, lm.y, lm.z])
        else:
            # if hand isn't in frame, add 0s as the feature
            # add 21*3 0's since there's 21 landmarks per hand and they
            #   are represented using 3D coordinates
            hand_keypoints.extend([0] * 21 * 3)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                hand_keypoints.extend([lm.x, lm.y, lm.z])
        else:
            hand_keypoints.extend([0] * 21 * 3)

        sequence.append(hand_keypoints)

    cap.release()
    sequence = np.array(sequence)

    return sequence.astype(np.float32)

def gen_videos_features(json_path: str=JSON_PATH, overwrite_prev_files:bool=False) -> None :
    """Generate features for each video and save them to disk."""
    with open(json_path, "r") as f:
        data = json.load(f)

    train_feature_paths = []
    train_labels = []
    test_feature_paths = []
    test_labels = []
    validation_feature_paths = []
    validation_labels = []

    for entry in data:
        gloss = entry["gloss"]

        for instance in entry["instances"]:
            video_file = os.path.join(VIDEO_DIR, f"{instance['video_id']}.mp4")

            if not os.path.exists(video_file):
                print(f"Skipping missing video: {video_file}")
                continue

            if instance["split"] == "train":
                npy_path = os.path.join(TRAIN_OUTPUT_DIR, f"{instance['video_id']}.npy")
            elif instance["split"] == "test" :
                npy_path = os.path.join(TEST_OUTPUT_DIR, f"{instance['video_id']}.npy")
            elif instance["split"] == "val" :
                npy_path = os.path.join(VALIDATION_OUTPUT_DIR, f"{instance['video_id']}.npy")
            if overwrite_prev_files :
                features = extract_features(video_file)
                np.save(npy_path, features)
                print(f"Saved features: {npy_path}")
            else :
                if not os.path.exists(npy_path):
                    # print(video_file)
                    features = extract_features(video_file)
                    np.save(npy_path, features)
                    print(f"Saved features: {npy_path}")
                else :
                    print(f"Features already generated for {npy_path}, skipped...")
# gen_videos_features()

def remove_zero_frames(input_dir: str, output_dir: str, overwrite_prev_file: bool=False) -> None :
    """Frames that have the hands out of view and so don't contribute
       any keypoints (array representing the frame contains all 0s)
       are removed.
       input_dir: directory with features generated from gen_videos_features()
       output_dir: directory where processed files are saved"""
    for file in os.scandir(input_dir) :
        cleaned_features = []
        if file.is_file() : # sanity check
            npy_path = os.path.join(output_dir, f"{file.name}")
            if not overwrite_prev_file :
                if os.path.exists(npy_path) :
                    continue
            features = np.load(f"{input_dir}/{file.name}")
            nframes, nfeatures = features.shape
            for i in range(nframes) :
                if np.any(features[i]) :
                    cleaned_features.append(features[i])
            cleaned_features = np.array(cleaned_features)
            np.save(npy_path, cleaned_features)
            print(f"Saved cleaned features: {npy_path}")

# remove_zero_frames(TRAIN_OUTPUT_DIR, TRAIN_OUTPUT_DIR_CLEANED)

# linear search as of now, maybe add code to order json by gloss or video_id
#   for a faster search?
# should probably be moved into a file in utils/
def find_gloss_by_video_id(video_id: str, json_path: str=JSON_PATH) -> str :
    video_id = video_id.strip(".npy")
    with open(json_path, "r") as f :
        data = json.load(f)
    for entry in data :
        for instance in entry["instances"] :
            if instance["video_id"] == video_id :
                return entry["gloss"]
# print(find_gloss_by_video_id("00421"))

def get_labels_sklearn(features_dir:str, json_path: str=JSON_PATH, overwrite_prev_file:bool=False) -> None :
    """Output corresponding label/gloss for a video in a 1d array
       that a sklearn SVM can use. Implicitly orders the labels
       by which file in features_dir is seen first, so ascending
       numerical order.
       features_dir: directory where features from gen_videos_features()
         are saved."""
    npy_path = os.path.join(features_dir, "ordered_labels.npy")
    if not overwrite_prev_file :
        if os.path.exists(npy_path) :
            print("labels file already exists, set the overwrite_prev_file flag to True to overwrite.")
            return
    with open(json_path, "r") as f :
        data = json.load(f)
    labels = []
    for file in os.scandir(features_dir) :
        if file.is_file() :
            label = find_gloss_by_video_id(f"{file.name.strip('.npy')}")
            if label != None :
                labels.append(label)
                print(f"Added {label} to labels.")
            else :
                print(f"Video {file.name} has no label.")
    np.save(npy_path, np.array(labels))
# get_labels_sklearn(VALIDATION_OUTPUT_DIR_CLEANED, JSON_PATH, True)

def flatten_directory(input_dir : str, output_file_name : str, overwrite_prev_file : bool=False) -> None :
    """ Converts a directory of normalized feature .npy files (2D arrays representing frame x features) into a 2D array representing (word x feature).
        Input: path to a directory of .npy files
        Output: 2D array of features"""
    
    npy_path = os.path.join(FLATTENED_OUTPUT_DIR, output_file_name)
    if not overwrite_prev_file:
        if os.path.exists(npy_path):
            print("Flattened features already exists, set overwrite_prev_file flag to True to overwrite.")
            return
    
    output = []
    
    # sorted scandir to ensure that final order of flattened features is consistent on different machines and labels are properly assigned
    for file in sorted(os.scandir(input_dir), key=lambda e: e.name):
        if file.is_file(): # sanity check
            features = np.load(f"{input_dir}/{file.name}")
        
        if (features.ndim != 2): # sanity check
            print(f"Skipped {file.name}.")
            continue
    
        output.append(np.ndarray.flatten(features))
    
    np.save(npy_path, np.array(output))

def flatten_directory_in_place(input_dir: str) -> list[np_ndarray] :
    """Converts a directory of feature .npy files (2D arrays representing 
       frame x features) into a 2D array representing (word x feature). It
       is recommended to use the non-in-place version of this function 
       (flatten_directory()) instead of this.
       Input: path to a directory of .npy files
       Output: 2D array of features"""

    output = []

    for file in os.scandir(input_dir):
        if file.is_file(): # sanity check
            features = np_load(f"{input_dir}/{file.name}")

        if (features.ndim != 2): # sanity check
            continue

        output.append(np_ndarray.flatten(features))

    # Note that we return a python list of np.ndarrays.
    # It isn't an ndarray itself because the flattened features are 
    #   different lengths
    return output
