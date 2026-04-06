import numpy as np
import os

def normalize_sequence_length_original(input_dir: str, output_dir, overwrite=False):
    """Normalize all feature files to have the same number of frames.
       Pads or truncates all .npy feature arrays in input_dir so they all have
       the same number of frames (rows). Uses the max length found across videos.
       This doesn't force width and height to be the same.
       
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
            
def normalize_sequence_length(input_dir: str, output_dir: str, max_frame_amount: int = -1, overwrite=False):
    """Normalize all feature files to have the same number of frames.
       Pads or truncates all .npy feature arrays in input_dir so they all have
       the same number of frames (rows). Uses the max length found across videos.
       This forces width and height to always be the same.
       
       input_dir: directory with features generated from gen_videos_features()
       output_dir: directory where processed files are saved"""

    if max_frame_amount <= 0 :
        max_frame_amount = get_max_video_frame_amount(input_dir)
    # else : max_frame_amount is already set

    for file in os.scandir(input_dir) :
        if file.is_file() and file.name.endswith(".npy"):
            out_path = os.path.join(output_dir, file.name)
            if not overwrite and os.path.exists(out_path):
                continue
            features = np.load(file.path)

            if features.size == 0 or file.name == "ordered_labels.npy":
                print(f"[WARNING] Skipping {file.name}: empty or invalid feature array (shape={features.shape}),(size={features.size})")
                continue

            # width and height should ALWAYS be the same
            n_frames, width, height, channels = features.shape
            if n_frames < max_frame_amount :
                pad_len = max_frame_amount - n_frames
                single_pad_frame = np.zeros((width,height,1), dtype=np.uint16)
                pad_frames = []
                for i in range(pad_len) :
                    pad_frames.append(single_pad_frame)
                pad_frames = np.array(pad_frames)
                # print(pad_frames.shape)
                # return
                padded = np.vstack([
                    features, pad_frames
                ])
            elif n_frames > max_frame_amount:
                # safety guard, should never enter this branch if data cleaning was done correctly
                raise ValueError(
                    f"[NormalizationError] Video '{file.name}' has {n_frames} frames, "
                    f"which exceeds the expected maximum of {max_frame_amount}. "
                    "This indicates that the dataset contains inconsistent feature lengths. "
                    "Recheck your cleaning or max_frame_amount computation step."
                )
            else:
                padded = features
            np.save(out_path, padded)
            print(f"Saved padded features: {out_path}")
            
def get_max_video_frame_amount(input_dir: str):
    files_checked_counter = 0
    max_length = 0
    for file in os.scandir(input_dir) :
        if file.name == "ordered_labels.npy":
            continue
        if file.is_file() and file.name.endswith(".npy"):
            arr = np.load(file.path)
            n_frames = arr.shape[0] # number of rows/frames
            max_length = max(max_length, n_frames)
            files_checked_counter += 1
            if files_checked_counter % 100 == 0 :
                print(f"Maximum Number of frames after {files_checked_counter} files checked: {max_length}")
    print(f"[normalize_sequence_length] Max frame length found: {max_length}")
    return max_length