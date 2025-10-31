import numpy as np
import sklearn.svm as svm

DIR = "/windows/Users/thats/Documents/archive"
JSON_PATH = f"{DIR}/WLASL_v0.3.json"
VIDEO_DIR = f"{DIR}/videos/"  # folder with your video files
TRAIN_OUTPUT_DIR_CLEANED = f"{DIR}/train_output_cleaned"
TEST_OUTPUT_DIR_CLEANED = f"{DIR}/test_output_cleaned"
VALIDATION_OUTPUT_DIR_CLEANED = f"{DIR}/validation_output_cleaned"

class svm :
    def __init__(self, classifier: str) :
        if classifier == "SVC" :
            self.clf = svm.SVC(decision_function_shape='ovo')
        elif classifier == "NuSVC" :
            self.clf = svm.NuSVC(decision_function_shape='ovo')
        else :
            raise Exception("Please give 'SVC' or 'NuSVC' as input to class")

    def train(self, X: np.ndarray, y:np.ndarray) :
        self.clf.fit(X, y)

    def predict(self, x: np.ndarray) :
        self.clf.predict(x)

# def fit_SVC(X: np.ndarray, y:np.ndarray) :
#     clf.fit

def main() :
    # svc = svm("SVC")
    pass
