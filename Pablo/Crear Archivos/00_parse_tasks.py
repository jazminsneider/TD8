import glob
import pandas as pd
import os

# Ruta a la carpeta uba-tasks
uba_folder = 'games-corpus/.uba-games/b1-dialogue-tasks'

# Busca todos los archivos .tasks
tasks_files = sorted(glob.glob(os.path.join(uba_folder, 's*.tasks')))

def read_list(fname):
    with open(fname, encoding='utf-8') as f:
        lines = [line.strip().split() for line in f if line.strip()]
    return lines

res = []
for tasks_file in tasks_files:
    # Extraé el número de sesión del nombre del archivo (por ejemplo, s01)
    session = os.path.basename(tasks_file).split('.')[0][1:3]
    tasks = read_list(tasks_file)
    tasks_info = [line[0:3] for line in tasks if len(line) >= 3 and "Images" in line[2]]
    i = 1
    for task_t0, task_tf, task_data in tasks_info:
        if "Images" in task_data and "Describer" in task_data:
            if task_data.endswith(";"):
                task_data = task_data[:-1]
            task_dict = dict([y.split(":") for y in task_data.split(";")])
            task_dict["t0"] = float(task_t0)
            task_dict["tf"] = float(task_tf)
            task_dict["session"] = session
            task_dict["task_number"] = i
            res.append(task_dict)
            i += 1

df = pd.DataFrame(res)
df.to_csv('csvs/tasks_uba.csv', index=False)
#df.head()
