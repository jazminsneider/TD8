from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from helper import *


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

tracks_list='lists/tracks.lst'
token_features_list = []
for idx, track_fname in tqdm(read_list(tracks_list)):
    tracks_output_filename = track_fname.replace('.csv', '_standardized.csv')

    tracks = pd.read_csv(track_fname, index_col="time")

    standardized_tracks = standardize(tracks.drop(["vad", "task", "intensity"], axis=1), drop_above_percentile=95, mask=~tracks.pitch.isna())
    standardized_tracks["intensity_standardized"] = standardize(tracks.loc[:, ["intensity"]], drop_above_percentile=95, mask=tracks.vad == 1)

    standardized_tracks["vad"] = tracks["vad"]
    standardized_tracks["task"] = tracks["task"]
    standardized_tracks.to_csv(tracks_output_filename)

    token_features_list.append((idx, tracks_output_filename))

    save_list(token_features_list, 'lists/z_scored_tracks.lst')


def standardize(df, drop_above_percentile, mask):
    filtered = df.loc[mask, :]
    top_thresh_percentiles = np.percentile(filtered, drop_above_percentile, axis=0)

    mean = filtered[filtered < top_thresh_percentiles].mean()
    stdev = filtered[filtered < top_thresh_percentiles].std()
    res = (df - mean) / stdev
    res.columns = [x + "_standardized" for x in df.columns]
    return res


