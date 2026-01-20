import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import utils.utils as utils
# commenting some of these out can make script run faster if you only want to call
#   specific functions

# print numpy arrays without truncation
# np.set_printoptions(threshold=sys.maxsize)

# global vars
# BASE_DIR = Path(__file__).resolve().parents[3] / "archive"
# DIR = str(BASE_DIR)
# DIR = "/windows/Users/thats/Documents/archive"
DIR = "/windows/Users/thats/Documents/archive"
 # folder where your dataset is
JSON_PATH = f"{DIR}/WLASL_v0.3.json"
# VIDEO_DIR = f"{DIR}/videos/"  # folder with your video files
VIDEO_DIR = f"{DIR}/videos/"  # folder with your video files
TRAIN_OUTPUT_DIR = f"{DIR}/train_output_json" # folder to save .npy feature files
TEST_OUTPUT_DIR = f"{DIR}/test_output_json" # folder to save .npy feature files
VALIDATION_OUTPUT_DIR = f"{DIR}/validation_output_json" # folder to save .npy feature files
TRAIN_OUTPUT_DIR_CLEANED = f"{DIR}/train_output_cleaned" # folder to save .npy feature files
TEST_OUTPUT_DIR_CLEANED = f"{DIR}/test_output_cleaned" # folder to save .npy feature files
VALIDATION_OUTPUT_DIR_CLEANED = f"{DIR}/validation_output_cleaned" # folder to save .npy feature files
TRAIN_OUTPUT_DIR_NORMALIZED = f"{DIR}/train_output_normalized"
TEST_OUTPUT_DIR_NORMALIZED = f"{DIR}/test_output_normalized"
VALIDATION_OUTPUT_DIR_NORMALIZED = f"{DIR}/validation_output_normalized"

# os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
# os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
# os.makedirs(VALIDATION_OUTPUT_DIR, exist_ok=True)
# os.makedirs(TRAIN_OUTPUT_DIR_CLEANED, exist_ok=True)
# os.makedirs(TEST_OUTPUT_DIR_CLEANED, exist_ok=True)
# os.makedirs(VALIDATION_OUTPUT_DIR_CLEANED, exist_ok=True)
# os.makedirs(TRAIN_OUTPUT_DIR_NORMALIZED, exist_ok=True)
# os.makedirs(TEST_OUTPUT_DIR_NORMALIZED, exist_ok=True)
# os.makedirs(VALIDATION_OUTPUT_DIR_NORMALIZED, exist_ok=True)

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
    hand_sequence = []
    pose_sequence = []
    face_sequence = []

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

        # pose 
        pose_keypoints = []
        if results.pose_landmarks :
            for lm in results.pose_landmarks.landmark :
                # all 3 coords added in due to prev functions assuming 3 coords,
                #   but z coord is currently not usable per mediapipe docs
                pose_keypoints.extend([lm.x, lm.y, lm.z])
        else :
            pose_keypoints.extend([0] * 21 * 3)

        # face
        face_keypoints = []
        if results.face_landmarks :
            for lm in results.face_landmarks.landmark :
                face_keypoints.extend([lm.x, lm.y, lm.z])
        else :
            face_keypoints.extend([0] * 21 * 3)

        hand_sequence.append(hand_keypoints)
        pose_sequence.append(pose_keypoints)
        face_sequence.append(face_keypoints)

    cap.release()

    # ndarray's aren't serializable to json, could use array.to_list() on them
    #   but not doing that because its expensive
    # hand_sequence = np.array(hand_sequence)
    # pose_sequence = np.array(pose_sequence)
    # face_sequence = np.array(face_sequence)

    return {"hands": hand_sequence, "pose": pose_sequence, "face": face_sequence}
# extract_features(f"{DIR}/videos/69547.mp4")
# sys.exit()

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
                npy_path = os.path.join(TRAIN_OUTPUT_DIR, f"{instance['video_id']}.json")
            elif instance["split"] == "test" :
                npy_path = os.path.join(TEST_OUTPUT_DIR, f"{instance['video_id']}.json")
            elif instance["split"] == "val" :
                npy_path = os.path.join(VALIDATION_OUTPUT_DIR, f"{instance['video_id']}.json")
            if overwrite_prev_files :
                # features = extract_features(video_file) #to be changed
                feature_dict = extract_features(video_file)
                with open(npy_path, 'w') as f :
                    json.dump(feature_dict, f)
                print(f"Saved features: {npy_path}")
            else :
                if not os.path.exists(npy_path):
                    # print(video_file)
                    # features = extract_features(video_file) #to be changed
                    # not sure what above comment is talking about so leaving it in
                    feature_dict = extract_features(video_file)
                    with open(npy_path, 'w') as f :
                        json.dump(feature_dict, f)
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

def normalize_sequence_length(input_dir: str, output_dir, overwrite=False):
    """Normalize all feature files to have the same number of frames.
       Pads or truncates all .npy feature arrays in input_dir so they all have
    the same number of frames (rows). Uses the max length found across videos.
       input_dir: directory with features generated from gen_videos_features()
       output_dir: directory where processed files are saved"""

    max_length = 0
    for file in os.scandir(input_dir) :
        if file.name == "ordered_labels.npy":
            continue
        if file.is_file() and file.name.endswith(".npy"):
            arr = np.load(file.path)
            n_frames = arr.shape[0] # number of rows/frames
            max_length = max(max_length, n_frames)
    print(f"[normalize_sequence_length] Max frame length found: {max_length}")

    for file in os.scandir(input_dir) :
        if file.is_file() and file.name.endswith(".npy"):
            out_path = os.path.join(output_dir, file.name)
            if not overwrite and os.path.exists(out_path):
                continue
            features = np.load(file.path)

            if features.size == 0 or file.name == "ordered_labels.npy":
                print(f"[WARNING] Skipping {file.name}: empty or invalid feature array (shape={features.shape}),(size={features.size})")
                continue

            n_frames, n_features = features.shape
            if n_frames < max_length :
                # pad with zeros
                pad_len = max_length - n_frames
                padded = np.vstack([
                    features,
                    np.zeros((pad_len, n_features), dtype=np.float32)
                ])
            elif n_frames > max_length:
                # safety guard, should never enter this branch if data cleaning was done correctly
                raise ValueError(
                    f"[NormalizationError] Video '{file.name}' has {n_frames} frames, "
                    f"which exceeds the expected maximum of {max_length}. "
                    "This indicates that the dataset contains inconsistent feature lengths. "
                    "Recheck your cleaning or max_length computation step."
                )
            else:
                padded = features
            np.save(out_path, padded)
            print(f"Saved normalized features: {out_path}")
        
# normalize_sequence_length(TRAIN_OUTPUT_DIR_CLEANED, TRAIN_OUTPUT_DIR_NORMALIZED, True)
# normalize_sequence_length(VALIDATION_OUTPUT_DIR_CLEANED, VALIDATION_OUTPUT_DIR_NORMALIZED, True)
# normalize_sequence_length(TEST_OUTPUT_DIR_CLEANED, TEST_OUTPUT_DIR_NORMALIZED, True)

def force_equal_dims_features_labels(input_dir: str, label_file: str, json_path: str = JSON_PATH, overwrite: bool = False) :
    """Make sure that X and y have the same dimensions. This function
       implements deleting entries from y to match the number of rows in X 
       (represented by the number of files in input_dir)
       while ensuring that the labels in y still correctly match the feature
       at the same index in X. 

       If this is needed in the reverse direction, implement when needed."""

    out_path = f"{input_dir}/ordered_labels_normalized.npy"
    if not overwrite and os.path.exists(f"{out_path}") :
        print("Normalized labels already exist. Please set overwrite param to True to execute function.")
        return
    labels = np.load(label_file)
    n_labels = labels.shape[0]
    idx = 0
    for file in sorted(os.scandir(input_dir), key=lambda e: e.name) :
        if file.is_file() and file.name.endswith(".npy"):
            if "ordered_labels" in file.name :
                print(f"{file.name} encountered... Skipping.")
                continue

            gloss = find_gloss_by_video_id(file.name, json_path)
            if gloss == None :
                raise Exception(f"{file.name} has no corresponding feature in {json_path}")

            try :
                if labels[idx] != gloss :
                    labels[idx] = gloss
                    print(f"gloss '{gloss}' at {idx} in labels does not match gloss for {file.name}... Value at {idx} has been replaced by matching gloss")
            except IndexError :
                raise IndexError("Number of features is greater than number of labels. Please generate the correct number of labels")
            idx += 1
    # idx + 1 in if stmt not needed b/c final loop increments idx once 
    #   more after all elements in input_dir have been checked
    if idx != n_labels : 
        labels = labels[0:idx]
        print("Labels array has been truncated.")
    np.save(out_path, labels)
    print("Normalized labels have been saved")
# normalize_labels(TRAIN_OUTPUT_DIR_NORMALIZED, 
#       f"{TRAIN_OUTPUT_DIR_CLEANED}/ordered_labels.npy", JSON_PATH, True)
#normalize_labels(TEST_OUTPUT_DIR_NORMALIZED, 
#           f"{TEST_OUTPUT_DIR_CLEANED}/ordered_labels.npy", JSON_PATH, True)
# normalize_labels(VALIDATION_OUTPUT_DIR_NORMALIZED, 
#         f"{VALIDATION_OUTPUT_DIR_CLEANED}/ordered_labels.npy", JSON_PATH, True)

# make another function that takes a dir of keypoint files and converts all
#   later
def hand_keypoint_to_img(keypoint_file: str, img_size: int = 300) :
    """Takes a set of unflattened keypoints (single file) generated by Mediapipe and 
       converts them into a 224x224  image that can be inputted to ResNet
    """
    with open(keypoint_file, 'r') as f :
        data = json.load(f)
    # keypoints are all between 0 and 1, so we un-normalize them
    hand_keypoints = ( np.array(data["hands"][:-1]) * img_size ).astype(np.int64)
    # for frame in data["face"] :
    #     print(len(frame))
    # return
    face_keypoints = ( np.array(data["face"][:-1]) * img_size ).astype(np.int64)
    pose_keypoints = ( np.array(data["pose"][:-1]) * img_size ).astype(np.int64)
    
    keypoints_per_frame_hand = hand_keypoints.shape[1]
    keypoints_per_frame_face = face_keypoints.shape[1]
    keypoints_per_frame_pose = pose_keypoints.shape[1]

    # keypoints are assumed to be (x,y,z) coordinates
    if keypoints_per_frame_hand % 3 != 0 : 
        raise Exception("Number of hand keypoints not divisible by 3")
    if keypoints_per_frame_face % 3 != 0 :
        raise Exception("Number of face keypoints not divisible by 3")
    if keypoints_per_frame_pose % 3 != 0 : 
        raise Exception("Number of pose keypoints not divisible by 3")

    # make sure to stop converting frames when no keypoints exist 
    #   (when coordinates are (0,0))
    patience = 0
    real_frame_encountered = False
    imgs = []
    all_keypoints = [hand_keypoints, face_keypoints, pose_keypoints]
    # all 3 keypoint arrays should have the same first dim and first dim represents
    #   frames
    for frame in range(hand_keypoints.shape[0]) :
        # don't start patience counter until non-zero frame hand frame is encountered
        if np.any(hand_keypoints[frame]) :
            real_frame_encountered = True
        if real_frame_encountered and not np.any(hand_keypoints[frame]) :
            patience += 1
        else :
            patience = 0
        if patience >= 3 :
            break
        img = np.zeros((img_size,img_size,1))
        for keypoint_set in all_keypoints :
            keypoint_frame = keypoint_set[frame]
            for x_coord in range(0, len(keypoint_frame) - 3, 3) :
                y_coord = x_coord+1
                if keypoint_frame[y_coord] >= img_size : 
                    keypoint_frame[y_coord] = img_size - 1
                if keypoint_frame[x_coord] >= img_size : 
                    keypoint_frame[x_coord] = img_size - 1
                img[keypoint_frame[y_coord]][keypoint_frame[x_coord]] = 1
        imgs.append(img)
        # add line between pose keypoints for better visualization
        POSE_CONNECTIONS = [
            (11, 13), (13, 15), # left arm points?
            (12, 14), (14, 16), # right arm points?
            (11, 12) # shoulder points?
        ]
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4), # thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # index finger
            (0, 9), (9,10), (10,11), (11,12), # middle finger
            (0,13), (13,14), (14,15), (15,16), # ring finger
            (0,17), (17,18), (18,19), (19,20), # pinky finger
            # palm connections?
        ]
        for connection in POSE_CONNECTIONS :
            x1 = pose_keypoints[frame][connection[0]*3]
            y1 = pose_keypoints[frame][connection[0]*3 + 1]
            x2 = pose_keypoints[frame][connection[1]*3]
            y2 = pose_keypoints[frame][connection[1]*3 + 1]
            cv2.line(img, (x1, y1), (x2, y2), (1), 1)
        for connection in HAND_CONNECTIONS :
            # right hand
            x1 = hand_keypoints[frame][connection[0]*3]
            y1 = hand_keypoints[frame][connection[0]*3 + 1]
            x2 = hand_keypoints[frame][connection[1]*3]
            y2 = hand_keypoints[frame][connection[1]*3 + 1]
            cv2.line(img, (x1, y1), (x2, y2), (1), 1)
            # left hand
            x1 = hand_keypoints[frame][(connection[0]+21)*3]
            y1 = hand_keypoints[frame][(connection[0]+21)*3 + 1]
            x2 = hand_keypoints[frame][(connection[1]+21)*3]
            y2 = hand_keypoints[frame][(connection[1]+21)*3 + 1]
            cv2.line(img, (x1, y1), (x2, y2), (1), 1)
        # visualizing each frame (hold esc to play it like a video)
        # while 1 :
        #     cv2.imshow('', img)
        #     k = cv2.waitKey(100)
        #     if k==27:    # Esc key to stop
        #         break
        #     elif k==-1:  # normally -1 returned,so don't print it
        #         continue
    return imgs
# hand_keypoint_to_img(f"{TRAIN_OUTPUT_DIR}/01384.json")
# hand_keypoint_to_img(f"00335.json")

# currently always overwrites old files; make it not overwrite them
def convert_keypoints_dir_to_video(input_dir: str, output_dir: str, overwrite_file: bool = False) :
    if not os.path.exists(input_dir) :
        raise Exception("Input directory does not exist.")
    if not os.path.exists(output_dir) :
        os.makedirs(output_dir, exist_ok=False)

    fail_count = 0

    for file in sorted(os.scandir(input_dir), key=lambda e: e.name) :
        if file.is_file() and file.name.endswith(".json") :
            if not overwrite_file and os.path.exists(f"{output_dir}/{file.name.strip('.json')}.npy") :
                print(f"{file.name.strip('.json')}.npy already exists... skipped")
                continue
            if "ordered_labels" in file.name :
                print(f"{file.name} encountered... Skipping.")
                continue
            # print(file.name)
            try :
                video = hand_keypoint_to_img(f"{input_dir}/{file.name}", 300)
            except :
                fail_count += 1
                print(f"Saving {file.name} failed")
                continue
            npy_path = os.path.join(output_dir, f"{file.name.strip('.json')}.npy")
            np.save(npy_path, video)
            print(f"{file.name.strip('.json')}.npy saved to {output_dir}")
    print(f"{fail_count} number of files failed to save")
convert_keypoints_dir_to_video(TRAIN_OUTPUT_DIR, f"{DIR}/train_output_video", True)
