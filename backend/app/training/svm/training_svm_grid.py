import os
import time
# import numpy as np
from numpy import array as np_array
from numpy import load as np_load
from numpy import ndarray as np_ndarray
import joblib

# from scikit.multiclass import MulticlassClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

DIR = ""
TRAIN_OUTPUT_FLATTENED = f"{DIR}/flattened_train_output_cleaned.npy"
TRAIN_OUTPUT_LABELS = f"{DIR}/train_output_labels.npy"
VALIDATION_OUTPUT_FLATTENED = f"{DIR}/flattened_validation_output_cleaned.npy"
VALIDATION_OUTPUT_LABELS = f"{DIR}/validation_output_labels.npy"

SAVED_MODELS_DIR = ""

start = time.time()

train_features = np_load(TRAIN_OUTPUT_FLATTENED)
train_labels = np_load(TRAIN_OUTPUT_LABELS)

validation_features = np_load(VALIDATION_OUTPUT_FLATTENED)
validation_labels = np_load(VALIDATION_OUTPUT_LABELS)

# cycles through the following parameters to find best combination of parameters
param_grid = [
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.1, 1, 10, 100, 1000]
    },
    {
        'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1, 10, 100, 1000],
        'svc__gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1]
    }
]

grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC()),
    param_grid,
    cv=3,
    verbose=3,
    n_jobs=-1
)

# model = make_pipeline(StandardScaler(), SVC(C=100, gamma=0.001, kernel='linear'))

grid.fit(train_features, train_labels)

# model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
# model.fit(X_train, y_train)

y_pred = grid.predict(validation_features)

print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)

best_model = grid.best_estimator_

joblib.dump(best_model, f"{SAVED_MODELS_DIR}/best_model.joblib")

acc = accuracy_score(validation_labels, y_pred)
print("Validation Accuracy:", acc)

y_pred2 = grid.predict(train_features)
acc = accuracy_score(train_labels, y_pred2)
print("Testing Accuracy:", acc)

end = time.time()
print(f"Train and predict time elapsed: {end-start}")
