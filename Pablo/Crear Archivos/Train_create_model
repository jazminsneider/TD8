import pandas as pd
import numpy as np
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from joblib import dump
from pathlib import Path
import os


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}

FILL_NA_WITH = -15
N_ESTIMATORS = 300
MAX_DEPTH = 10
MAX_FEATURES = 0.5
balance_method = "oversample"
n_jobs = 4

data_folder = Path("/home/gallin/Documentos/pablo/output/X_y/overlap")
output_folder = Path("/home/gallin/Documentos/pablo/output/X_val_probas2/overlap")
output_folder.mkdir(parents=True, exist_ok=True)
X = pd.read_csv(os.path.join(data_folder, "X.csv"), index_col=0)
X_columns = X.columns
indices = X.index
X = X.fillna(FILL_NA_WITH).values.astype(np.float32)
y_true = pd.read_csv(os.path.join(data_folder, "y.csv"), index_col=0)
sessions = pd.read_csv(os.path.join(data_folder, "sessions.csv"), index_col=0).values.squeeze()
    # tasks = pd.read_csv(os.path.join(data_folder, "tasks.csv"), index_col=0).values.squeeze()

np.random.seed(1234)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y_true.values.squeeze())
idx_train = indices

if any([c.endswith("_A") for c in X_columns]):
        # Mirror channels
        X_train_mirror = X.copy()
        for col_index, col in enumerate(X_columns):
            if col.endswith("_A"):
                other_col = col.replace("_A", "_B")
                other_col_index = list(X_columns).index(other_col)
                X_train_mirror[:, col_index] = X[:, other_col_index].copy()
                X_train_mirror[:, other_col_index] = X[:, col_index].copy()
        X_train = np.concatenate([X, X_train_mirror])
        y_train = np.concatenate([y, y])
        idx_train = np.concatenate([idx_train, idx_train])
    
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, max_features=MAX_FEATURES, n_jobs=n_jobs)

if balance_method == "oversample":
        resampling_indices, _ = RandomOverSampler().fit_resample(X=np.arange(len(y_train)).reshape(-1, 1), y=y_train)
        X_train, y_train = X_train[resampling_indices.squeeze()], y_train[resampling_indices.squeeze()]
        idx_train = idx_train[resampling_indices.squeeze()]

if balance_method == "class_weights":
        weights = get_class_weights(y_train)
        print("weights", weights)
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, class_weight=weights, max_features=MAX_FEATURES, n_jobs=n_jobs)

clf.fit(X_train, y_train)
dump(clf, os.path.join(output_folder, "rf_model.joblib"))

