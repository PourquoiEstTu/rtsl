import os
import time
# import numpy as np
from numpy import array as np_array
from numpy import load as np_load
from numpy import ndarray as np_ndarray
# from cuml.multiclass import MulticlassClassifier
from cuml.svm import SVC
from cuml.preprocessing import LabelEncoder
from cuml import metrics

DIR = "/u50/quyumr/archive"
JSON_PATH = f"{DIR}/WLASL_v0.3.json"
VIDEO_DIR = f"{DIR}/videos/"  # folder with your video files
TRAIN_OUTPUT_DIR_CLEANED = f"{DIR}/train_output_cleaned"
TEST_OUTPUT_DIR_CLEANED = f"{DIR}/test_output_cleaned"
VALIDATION_OUTPUT_DIR_CLEANED = f"{DIR}/validation_output_cleaned"
VALIDATION_OUTPUT_DIR_NORMALIZED = f"{DIR}/validation_output_normalized"
TRAIN_OUTPUT_DIR_NORMALIZED = f"{DIR}/train_output_normalized"
TEST_OUTPUT_DIR_NORMALIZED = f"{DIR}/test_output_normalized"

# class svm :
#     def __init__(self, classifier: str) :
#         if classifier == "SVC" :
#             self.clf = svm.SVC(decision_function_shape='ovo')
#         elif classifier == "NuSVC" :
#             self.clf = svm.NuSVC(decision_function_shape='ovo')
#         else :
#             raise Exception("Please give 'SVC' or 'NuSVC' as input to class")
# 
#     def train(self, X: np.ndarray, y:np.ndarray) :
#         self.clf.fit(X, y)
# 
#     def predict(self, x: np.ndarray) :
#         self.clf.predict(x)

# def fit_SVC(X: np.ndarray, y:np.ndarray) :
#     clf.fit

# tmp until this pr gets merged
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

    # note that we return a python list of np.ndarrays
    # it isn't an ndarray itself because the flattened features are different lengths
    return output

def main() -> int :
    start = time.time()
    # change this to training, test, validation depending on what you're doing
    X_train = np_array( flatten_directory_in_place(f"{TRAIN_OUTPUT_DIR_NORMALIZED}") )
    y_train = np_load(f"{TRAIN_OUTPUT_DIR_NORMALIZED}/ordered_labels_normalized.npy")
    X_val = np_array( flatten_directory_in_place(f"{VALIDATION_OUTPUT_DIR_NORMALIZED}") )
    y_val = np_load(f"{VALIDATION_OUTPUT_DIR_NORMALIZED}/ordered_labels_normalized.npy")

    le_y_train = LabelEncoder()
    le_y_val = LabelEncoder()
    y_train_numeric = le_y_train.fit_transform(y_train)
    y_val_numeric = le_y_val.fit_transform(y_val)
    end = time.time()
    print(f"Load and encode time: {end-start}")

    start = time.time()
    clf = SVC(C=1, kernel='rbf', gamma='auto', tol=1e-3, 
                decision_function_shape='ovr')
    clf.fit(X_train, y_train_numeric)
    # print(metrics.accuracy_score( y_val, clf.predict(X_val) ))
    end = time.time()
    print(f"Train and predict time elapsed: {end-start}")
    return 0
main()
