import glob
import pandas as pd
import os
corpus_folder = "games-corpus/.uba-games/b1-dialogue-turns/"
turns_files_regex = "*.turns"

#primero hay que definir los diccionarios que están en experiments.ini
#fijarse que onda con X3
OVERLAPPED_TRANSITIONS=["O","I","BI","BC_0","X2_0",]

turns_files = sorted(list(glob.glob(corpus_folder + turns_files_regex)))
assert len(turns_files) != 0

#este usa los archivos csv creados arribs de ipus, tasks y usa los .turns
ipus_table=pd.read_csv('csvs/ipus_uba.csv')
#el archivo tasks leido con pd se llama tasks
tasks=pd.read_csv('csvs/tasks_uba.csv')

#funcion auxiliar read_turns porque no encontré la libreria ml.utils
def read_turns(turns_file):
    with open(turns_file, encoding='utf-8') as f:
        lines = [line.strip().split() for line in f if line.strip()]
    return [(float(l[0]), float(l[1]), l[2]) for l in lines if len(l) == 3]


info = []
for turns_file in turns_files:
        turns = read_turns(turns_file)
        filename = os.path.basename(turns_file)
        parts = filename.split(".")
        session = parts[0][1:]  # s01 → 01
        channel = parts[3]
        object_id = parts[2]
        tasks_session = tasks[tasks.session == int(session)]

        for (t0, tf, tt_label) in turns:
            task = tasks_session[(tasks_session.t0 <= t0) & (tasks_session.tf >= t0)]
            if len(task) == 0:
                continue
            assert len(task) == 1
            task_id = task.iloc[0].task_number
            info.append(dict(
                tt_label=tt_label,
                session_number=session,
                speaker2=channel,
                task=task_id,
                ipu2_start_time=round(t0, 4),
                ipu2_end_time=round(tf, 4),
                overlapped_transition=tt_label in OVERLAPPED_TRANSITIONS
            ))

df = pd.DataFrame(info)
df.to_csv('csvs/tt-table.csv', index=None)
print(f"File saved at {"tt-table.csv"}")
print(df.tt_label.value_counts())
df.head()

