import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader

# print numpy arrays without truncation
np.set_printoptions(threshold=sys.maxsize)

# global vars
# DIR = "/windows/Users/thats/Documents/archive"
DIR = "../../../"
# JSON_PATH = f"{DIR}/WLASL_v0.3.json"
JSON_PATH = f"{DIR}/info.json"
VIDEO_DIR = f"{DIR}/videos/"  # folder with your video files
TRAIN_OUTPUT_DIR = f"{DIR}/train_output" # folder to save .npy feature files
TEST_OUTPUT_DIR = f"{DIR}/test_output" # folder to save .npy feature files
VALIDATION_OUTPUT_DIR = f"{DIR}/validation_output" # folder to save .npy feature files
TRAIN_OUTPUT_DIR_CLEANED = f"{DIR}/train_output_cleaned" # folder to save .npy feature files
TEST_OUTPUT_DIR_CLEANED = f"{DIR}/test_output_cleaned" # folder to save .npy feature files
VALIDATION_OUTPUT_DIR_CLEANED = f"{DIR}/validation_output_cleaned" # folder to save .npy feature files
TARGET_LENGTH = 64                   # number of frames per sequence
BATCH_SIZE = 4
CATEGORIES_TO_USE = ["book", "bye", "hello"]  # Only preprocess these glosses

os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR_CLEANED, exist_ok=True)
os.makedirs(VALIDATION_OUTPUT_DIR_CLEANED, exist_ok=True)

# INITIALIZE MEDIAPIPE HOLISTIC
# essentially uses the mediapipe holistic model to extract hands and pose features
# commented out because it makes code run slower when not in use
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
            hand_keypoints.extend([0] * 21 * 3)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                hand_keypoints.extend([lm.x, lm.y, lm.z])
        else:
            hand_keypoints.extend([0] * 21 * 3) # 21 landmarks per hand

        # POSE
        # pose_keypoints = []
        # if results.pose_landmarks:
        #     for lm in results.pose_landmarks.landmark:
        #         pose_keypoints.extend([lm.x, lm.y, lm.z])
        # else:
        #     pose_keypoints.extend([0] * 33 * 3) # 33 pose landmarks
        #
        # frame_features = hand_keypoints# + pose_keypoints
        sequence.append(hand_keypoints)

    cap.release()
    sequence = np.array(sequence)

    # pad/trim sequence to TARGET_LENGTH
    # if len(sequence) > TARGET_LENGTH:
    #     sequence = sequence[:TARGET_LENGTH]
    # elif len(sequence) < TARGET_LENGTH:
    #     pad = np.zeros((TARGET_LENGTH - len(sequence), sequence.shape[1]))
    #     sequence = np.vstack([sequence, pad])

    return sequence.astype(np.float32)

# print(len(extract_features(f"{VIDEO_DIR}/69210.mp4")[0]))
# print(extract_features(f"{VIDEO_DIR}/69210.mp4")[32])

def gen_videos_features() -> None :
    # LOAD JSON AND PROCESS ALL VIDEOS
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    train_feature_paths = []
    train_labels = []
    test_feature_paths = []
    test_labels = []
    validation_feature_paths = []
    validation_labels = []

    for entry in data:
        gloss = entry["gloss"]
        # only use the categories we care about
        # if gloss not in CATEGORIES_TO_USE:
        #     continue  # Skip unwanted categories

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
            if not os.path.exists(npy_path):
                # print(video_file)
                features = extract_features(video_file)
                np.save(npy_path, features)
                print(f"Saved features: {npy_path}")

            # This block is not necessary anymore as we are moving to 
            #   file-based preprocessing
            # Only append if the feature file exists
            # if os.path.exists(npy_path):
            #     if instance["split"] == "train":
            #         train_feature_paths.append(npy_path)
            #         train_labels.append(gloss)
            #     elif instance["split"] == "test":
            #         test_feature_paths.append(npy_path)
            #         test_labels.append(gloss)
            #     elif instance["split"] == "val" :
            #         validation_feature_paths.append(npy_path)
            #         validation_labels.append(gloss)

def remove_zero_frames(input_dir: str, output_dir: str) -> None :
    """Frames that have the hands out of view and so don't contribute 
       any keypoints are removed
       input_dir: directory with features from extract_features
       output_dir: directory where processed files are saved"""
    for file in os.scandir(input_dir) :
        cleaned_features = []
        if file.is_file() : # sanity check
            npy_path = os.path.join(output_dir, f"{file.name}")
            # assume file has already had all zero frames removed if it
            #   already exists in output_dir
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

# remove_zero_frames(VALIDATION_OUTPUT_DIR, VALIDATION_OUTPUT_DIR_CLEANED)

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

def get_labels_pytorch(features_dir:str, json_path: str=JSON_PATH, overwrite_prev_file:bool=False) -> None :
    """Output corresponding label/gloss for a video in a 1d array
       that pytorch code can use. """
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
    np.save(npy_path, labels)

def get_labels_sklearn(features_dir:str, json_path: str=JSON_PATH, overwrite_prev_file:bool=False) -> None :
    """Output corresponding label/gloss for a video in a 1d array
       that a sklearn SVM can use. Implicitly orders the labels
       by which file in features_dir is seen first, so ascending
       numerical order."""
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
    le = LabelEncoder()
    y_numeric = le.fit_transform(labels)
    np.save(npy_path, y_numeric)
# get_labels_sklearn(TRAIN_OUTPUT_DIR_CLEANED)

# TODO: write function to flatten 2d arrays in all feature files into one 
#   large array where the entries are the features from all frames, this is
#   is not meant to be saved as a file, but used in the training_svm.py file

# ENCODE LABELS
# # essentially converts string labels to numeric labels
# le = LabelEncoder()
# y_numeric = le.fit_transform(train_labels)
# print("Classes:", le.classes_)
#
# # create a dataset class to be used with pytorch dataloader
# class JSONASLDataset(Dataset):
#     def __init__(self, features_paths, labels):
#         self.features_paths = features_paths
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.features_paths)
#
#     def __getitem__(self, idx):
#         X = np.load(self.features_paths[idx])
#         y = self.labels[idx]
#         return torch.tensor(X), torch.tensor(y) #tensor of features and label
#
# # CREATE DATASET AND DATALOADER
# train_dataset = JSONASLDataset(train_feature_paths, y_numeric)
# # save numeric labels for training script
# np.save(os.path.join(OUTPUT_DIR, "labels.npy"), y_numeric)
#
# # not sure what this would be used for?
# # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
# print(f"Training dataset ready. Number of samples: {len(train_dataset)}")

# get_labels_pytorch(TRAIN_OUTPUT_DIR_CLEANED, JSON_PATH, overwrite_prev_file=True)