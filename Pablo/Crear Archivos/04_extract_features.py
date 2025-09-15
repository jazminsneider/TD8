import os
import subprocess
from pathlib import Path
import shutil
import pandas as pd

opensmile = "/home/gallin/opensmile/build/progsrc/smilextract/SMILExtract" # Path al ejecutable de OpenSMILE
config_f = "/home/gallin/Documentos/pablo/female.ini" # Path al archivo de configuración femenino
config_m = "/home/gallin/Documentos/pablo/male.ini" # Path al archivo de configuración masculino
wavs_dir = Path("/home/gallin/games-corpus/.uba-games/b1-dialogue-wavs") # Path a la carpeta con los archivos WAV
output_dir = Path("/home/gallin/Documentos/pablo/output") # Path a la carpeta donde se guardarán los CSV generados

output_dir.mkdir(parents=True, exist_ok=True) 
df = pd.read_csv("ipus_ubaa.csv")
df["filename"] = df["wav"].str.split("/").str[-1]
for wav in wavs_dir.glob("*.wav"):
    stem = wav.stem
    audio = df[df["filename"] == f"{stem}.phrases"]
    print(f"Procesando {stem}...")
    gender = audio["speaker_gender"].iloc[0]
    if(gender == "m"): # masculino
        config = config_m 
    else:
        config = config_f # femenino
    cmd = [opensmile, "-C", config, "-I", str(wav)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR en {stem}: {e}")
        continue
    src = Path("output/out.csv")
    if src.exists():
        dst = output_dir / f"{stem}.csv"
        shutil.move(src, dst)
    else:
        print(f"No se generó CSV para {stem}")
