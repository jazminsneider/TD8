from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from helper import *
import glob
import helper



#función auxiliar que estandariza las pistas de audio
def standardize(df, drop_above_percentile, mask):
    filtered = df.loc[mask, :]
    top_thresh_percentiles = np.percentile(filtered, drop_above_percentile, axis=0)
    mean = filtered[filtered < top_thresh_percentiles].mean()
    stdev = filtered[filtered < top_thresh_percentiles].std()
    res = (df - mean) / stdev
    res.columns = [x + "_standardized" for x in df.columns]
    return res

#---------------------------------------------------------------------------------------

tracks_list="/home/gallin/Documentos/pablo/lists/tracks.lst"
output_folder="/home/gallin/Documentos/pablo/output/z_scored_tracks/"
output_list="/home/gallin/Documentos/pablo/lists/z_scored_tracks.lst"
force=False

helper.mkdir_p(os.path.dirname(output_list))
# Lista donde guardamos los archivos procesados
token_features_list = []

# Recorremos todos los CSV
for idx, track_fname in tqdm(helper.read_list(tracks_list)):
        tracks_output_filename = os.path.basename(track_fname).replace(".csv", "_standardized.csv")
        tracks_output_filename = os.path.join(output_folder, tracks_output_filename)

        if not force and helper.exists(tracks_output_filename):
            helper.warning("{} already exist, skipping (use --force to overwrite)".format(tracks_output_filename))
        else:
            tracks = pd.read_csv(track_fname, index_col="time")

            standardized_tracks = standardize(tracks.drop(["vad", "task", "intensity"], axis=1), drop_above_percentile=95, mask=~tracks.pitch.isna())
            standardized_tracks["intensity_standardized"] = standardize(tracks.loc[:, ["intensity"]], drop_above_percentile=95, mask=tracks.vad == 1)

            standardized_tracks["vad"] = tracks["vad"]
            standardized_tracks["task"] = tracks["task"]
            standardized_tracks.to_csv(tracks_output_filename)

        token_features_list.append((idx, tracks_output_filename))

helper.save_list(token_features_list, output_list)
