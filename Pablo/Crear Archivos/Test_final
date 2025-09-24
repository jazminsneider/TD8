import pandas as pd
import helper
import joblib

import os.path
import numpy as np
from pathlib import Path

FILL_NA_WITH = -15
force=True
data_folder = "/home/gallin/Documentos/pablo/output/X_y/overlap_held"
model_fname = "/home/gallin/Documentos/pablo/output/X_val_probas2/overlap/rf_model.joblib"
output_folder = Path("/home/gallin/Documentos/pablo/MODELINMODELANFINAL/overlap_COHETE")
output_folder.mkdir(parents=True, exist_ok=True)
output_fname = os.path.join(output_folder, "predicted_probas.csv")
if not force and helper.exists(output_fname):
    helper.warning("{} already exist, skipping (use --force to overwrite)".format(output_fname))

output_folder = os.path.abspath(output_folder)
helper.mkdir_p(output_folder)

X = pd.read_csv(os.path.join(data_folder, "X.csv"), index_col=0)
X_values = X.fillna(FILL_NA_WITH).values.astype(np.float32)

model = joblib.load(model_fname)
probas = model.predict_proba(X_values)

y_true = pd.read_csv(os.path.join(data_folder, "y.csv"), index_col=0)

y_probas = pd.DataFrame(probas, index=X.index, columns=np.unique(y_true))

y_probas["real"] = y_true
y_probas.to_csv(output_fname)

