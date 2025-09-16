#!/usr/bin/env python
# coding: utf-8

import os.path
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import helper
from tqdm import tqdm
import glob
import re
import os 
import ml.utils
import os
import subprocess
from pathlib import Path
import shutil
import pandas as pd
import ml.parsing.arff

import ml.opensmile


#cambiar para adaptar:
from pathlib import Path
import tempfile

opensmile = Path("/home/gallin/opensmile/build/progsrc/smilextract/SMILExtract")
config_f = Path("/home/gallin/Documentos/pablo/female.ini")
config_m = Path("/home/gallin/Documentos/pablo/male.ini")

wavs_dir = Path("/home/gallin/games-corpus/.uba-games/b1-dialogue-wavs")
output_folder = Path("/home/gallin/Documentos/pablo/tracks")
output_folder.mkdir(parents=True, exist_ok=True)

ipus_table = Path("/home/gallin/Documentos/pablo/realrealipus.csv")
tasks_table = Path("/home/gallin/Documentos/pablo/tasks_uba.csv")

output_list = Path("/home/gallin/Documentos/pablo/lists/tracks.lst")
output_list.parent.mkdir(parents=True, exist_ok=True)

temp_folder = Path(tempfile.mkdtemp())


#-------------------------------------------------------------

tasks=pd.read_csv(tasks_table)
token_features_list=[]
ipus=pd.read_csv(ipus_table)

#-------------------------------------------------------------

def compute_tracks(filename, gender, temp_folder):
    if gender == "m":
        config = config_m
    else:
        config = config_f
    print(filename)
    filename = filename.replace(".phrases", ".wav")
    print(filename)
    filename= filename.replace("-phrases", "-wavs")
    print(filename)
    filename=os.path.join("/home/gallin",filename)
    print(filename)
    stem = Path(filename).stem  # nombre sin extensión

    cmd = [opensmile, "-C", config, "-I", filename]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR en {stem}: {e}")
        return None  # o lo que necesites

    src = Path("output/out.csv")
    print(src)
    if src.exists():
        output_dir = Path(temp_folder)
        output_dir.mkdir(parents=True, exist_ok=True)  # asegurarse de que existe
        dst = output_dir / f"{stem}.csv"
        shutil.move(src, dst)

        # Leer el CSV movido
        data = pd.read_csv(dst,sep=";")
        print(data.head())
        print(data.columns)

        series={}
        times=data["frameTime"]

        pitch = data["F0final_sma"]
        intensity = data["pcm_intensity_sma"]
        jitter = data["jitterLocal"]
        shimmer = data["shimmerLocal"]
        logHNR = data["logHNR"]

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

#-------------------------------------------------------------


for session_channel, group in ipus.groupby("session_channel"):
    session,channel=session_channel.split("_")
    session=int(session)

    tracks_output_filename="{}/{}.csv".format(output_folder,session_channel)

    tracks=pd.DataFrame()
    for wav in sorted(set(group.wav)):
        times,features=compute_tracks(wav,group.iloc[0].speaker_gender,temp_folder)
        tracks_i=pd.DataFrame(features) 
        tracks_i["time"]=times
        tracks_i=tracks_i[~tracks_i.time.duplicated(keep="last")]
        tracks_i.set_index("time",inplace=True)

        vad_mask=np.zeros(tracks_i.index.shape)
        

        group_for_mask = group

        for idx, ipu in group_for_mask.iterrows():
            vad_mask[(tracks_i.index >= ipu.ipu_start_time) & (tracks_i.index <= ipu.ipu_end_time)] = 1
        tracks_i[vad_mask == 0] = np.nan
        tracks_i["vad"] = vad_mask
        tracks_i["time"] = tracks_i.index

        
    
        tracks_i["task"] = None

        for _, task in tasks[tasks.session == session].iterrows():
            tracks_i.loc[(tracks_i.index >= task.t0) & (tracks_i.index <= task.tf), "task"] = task.task_number
        tracks = pd.concat([tracks, tracks_i], ignore_index=True)


    tracks = tracks[~tracks.task.isna()]
    tracks.to_csv(tracks_output_filename, index=False)

    token_features_list.append((session_channel, tracks_output_filename))

helper.save_list(token_features_list, output_list)


#----------------------------------------------------------

