import os
import numpy as np
# import sklearn.svm as svm
# from cuml.multiclass import MulticlassClassifier
from cuml.svm import SVC
from cuml.preprocessing import LabelEncoder

DIR = "/u50/quyumr/rtsl-features"
JSON_PATH = f"{DIR}/WLASL_v0.3.json"
VIDEO_DIR = f"{DIR}/videos/"  # folder with your video files
TRAIN_OUTPUT_DIR_CLEANED = f"{DIR}/train_output_cleaned"
TEST_OUTPUT_DIR_CLEANED = f"{DIR}/test_output_cleaned"
VALIDATION_OUTPUT_DIR_CLEANED = f"{DIR}/validation_output_cleaned"
VALIDATION_OUTPUT_DIR_NORMALIZED = f"{DIR}/validation_output_normalized"

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
def flatten_directory(input_dir: str) -> list[np.ndarray] :
    """ Converts a directory of feature .npy files (2D arrays representing frame x features) into a 2D array representing (word x feature).
        Input: path to a directory of .npy files
        Output: 2D array of features"""

    output = []

    for file in os.scandir(input_dir):
        if file.is_file(): # sanity check
            features = np.load(f"{input_dir}/{file.name}")

        if (features.ndim != 2): # sanity check
            continue

        output.append(np.ndarray.flatten(features))

    # note that we return a python list of np.ndarrays
    # it isn't an ndarray itself because the flattened features are different lengths
    return output

def main() -> int :
    # change this to training, test, validation depending on what you're doing
    data_type = "validation"
    X = np.array( flatten_directory(f"{DIR}/{data_type}_output_normalized") )
    y = np.load(f"{DIR}/{data_type}_output_cleaned/ordered_labels.npy")
    le_y = LabelEncoder()
    y_numeric = le_y.fit_transform(y)
    print(X.shape)
    print(y_numeric.shape)
    return
    clf = SVC(C=1, kernel='rbf', gamma='auto', tol=1e-3, 
                decision_function_shape='ovr')
    clf.fit(X,y_numeric)
    return 0
main()
