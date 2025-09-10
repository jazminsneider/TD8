from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from helper import *
import glob


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
# carpeta a los csv's de audio

# Carpeta con los CSV originales
input_folder = "csvs/features"
# Carpeta de salida para los CSV estandarizados
output_folder = "standardized"
os.makedirs(output_folder, exist_ok=True)

# Lista donde guardamos los archivos procesados
token_features_list = []

# Recorremos todos los CSV
for track_fname in tqdm(sorted(glob.glob(os.path.join(input_folder, "*.csv")))):
    # Armamos el nombre de salida en la carpeta 'standardized'
    base_name = os.path.basename(track_fname).replace(".csv", "_standardized.csv")
    tracks_output_filename = os.path.join(output_folder, base_name)

    # Leemos el CSV
    tracks = pd.read_csv(track_fname, index_col="time")

    # Mostramos columnas para debug
    # print(tracks.columns)

    # DataFrame sin columnas de control
    data_to_standardize = tracks.drop(["vad", "task", "intensity"], axis=1, errors="ignore")

    # Mascara: solo filas válidas según pitch
    mask_pitch = "pitch" in tracks.columns and (~tracks.pitch.isna())

    if mask_pitch.any() if isinstance(mask_pitch, pd.Series) else mask_pitch:
        standardized_tracks = standardize(
            data_to_standardize,
            drop_above_percentile=95,
            mask=mask_pitch
        )
    else:
        # DataFrame vacío si no hay pitch válido
        standardized_tracks = pd.DataFrame(columns=data_to_standardize.columns)

    # Estandarización de intensidad si existe
    if "intensity" in tracks.columns and "vad" in tracks.columns and (tracks.vad == 1).any():
        standardized_tracks["intensity_standardized"] = standardize(
            tracks.loc[:, ["intensity"]],
            drop_above_percentile=95,
            mask=tracks.vad == 1
        )
    else:
        standardized_tracks["intensity_standardized"] = np.nan

    # Añadimos columnas de control
    if "vad" in tracks.columns:
        standardized_tracks["vad"] = tracks["vad"]
    if "task" in tracks.columns:
        standardized_tracks["task"] = tracks["task"]

    # Guardamos el CSV estandarizado
    standardized_tracks.to_csv(tracks_output_filename)

    # Guardamos el nombre en la lista
    token_features_list.append(tracks_output_filename)

# Guardamos la lista final
os.makedirs("lists", exist_ok=True)  # aseguramos que la carpeta exista
save_list(token_features_list, 'lists/z_scored_tracks.lst')
