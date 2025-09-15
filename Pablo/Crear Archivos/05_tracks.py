#!/usr/bin/env python
# coding: utf-8

import os.path
import tempfile



import numpy as np
import pandas as pd


from tqdm import tqdm
import glob
import re
import os 

def compute_tracks(filename):
    
    df=pd.read_csv(filename,sep=";")
    series = {}
    
    times=df["frameTime"]

    pitch = df["F0final_sma"]
    
    intensity = df["pcm_intensity_sma"]
    
    # loudness = ml.parsing.arff.get_column(data, "pcm_loudness_sma")
    jitter = df["jitterLocal"]
    
    shimmer = df["shimmerLocal"]
    
    logHNR = df["logHNR"]

    # intensity[~(intensity > 0)] = np.nan
    jitter[~(pitch > 0)] = np.nan
    shimmer[~(pitch > 0)] = np.nan
    logHNR[~(pitch > 0)] = np.nan
    pitch[~(pitch > 0)] = np.nan

    series["pitch"] = pitch
    series["intensity"] = intensity
    # series["loudness"] = loudness
    series["jitter"] = jitter
    series["shimmer"] = shimmer
    series["logHNR"] = logHNR

    return times, series


script_dir = os.path.dirname(os.path.abspath(__file__))
tasks_table = os.path.join(script_dir, 'tasks_uba.csv')
ipus_table = os.path.join(script_dir, 'ipus_uba.csv')
tasks = pd.read_csv(tasks_table)
token_features_list = []
ipus = pd.read_csv(ipus_table)


folder = os.path.join(script_dir, 'features_csvs', 'output')
output_folder = os.path.join(script_dir, 'features_csvs', 'features_arreglados')


for track_fname in tqdm(sorted(glob.glob(os.path.join(folder, "*.csv")))):
    base = os.path.basename(track_fname)
    nombre = re.match(r"s(\d+)\..*\.([AB])\.csv", base)
    session = int(nombre.group(1))
    channel = nombre.group(2)
    tracks_output_filename = "{}/{}{}_wavs.csv".format(output_folder, session,channel)

    
    group_for_mask = ipus[(ipus["session_number"] == session) & (ipus["channel"] == channel)]

    
    times, features = compute_tracks(track_fname)
    tracks_i = pd.DataFrame(features)
    tracks_i["time"] = times
    tracks_i = tracks_i[~tracks_i.time.duplicated(keep="last")]
    tracks_i.set_index("time", inplace=True)

    
    vad_mask = np.zeros(tracks_i.index.shape)
    for _, ipu in group_for_mask.iterrows():
        vad_mask[(tracks_i.index >= ipu.ipu_start_time) & (tracks_i.index <= ipu.ipu_end_time)] = 1

    tracks_i[vad_mask == 0] = np.nan
    tracks_i["vad"] = vad_mask
    tracks_i["time"] = tracks_i.index

    
    tracks_i["task"] = None
    for _, task in tasks[tasks.session == session].iterrows():
        tracks_i.loc[(tracks_i.index >= task.t0) & (tracks_i.index <= task.tf), "task"] = task.task_number

    
   
    tracks_i = tracks_i[~tracks_i.task.isna()]
    tracks_i.to_csv(tracks_output_filename, index=False)

    

