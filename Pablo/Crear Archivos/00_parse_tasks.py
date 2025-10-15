import glob
import pandas as pd
import os
import helper  

# Ruta a la carpeta uba-tasks
uba_folder = '/Users/jazsneider/Downloads/Debug/games-corpus/.uba-games/b1-dialogue-tasks'
output_fname = "csvs/tasks_uba.csv"

# Sesiones a procesar
sessions = {"01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"}

normalize_task_time = False
res = []

for session in sorted(sessions):
    # Filtra solo los archivos de la sesión actual
    tasks_files = sorted(glob.glob(os.path.join(uba_folder, f's{session}.objects.1.tasks')))
    print(tasks_files)
    
    i = 1  # Contador de tareas dentro de la sesión
    for tasks_file in tasks_files:
        tasks = helper.read_list(tasks_file)
        tasks_info = [line[0:3] for line in tasks if len(line) >= 3 and "Images" in line[2]]
        
        for task_t0, task_tf, task_data in tasks_info:
            if "Images" in task_data and "Describer" in task_data:
                # Quitar ; final si existe
                if task_data.endswith(";"):
                    task_data = task_data[:-1]

                # Convertir a diccionario
                task_dict = dict(y.split(":") for y in task_data.split(";"))

                # Agregar tiempos normalizados o no
                task_dict["t0"] = float(task_t0) if not normalize_task_time else 0
                task_dict["tf"] = float(task_tf) if not normalize_task_time else float(task_tf) - float(task_t0)

                # Agregar info de sesión y número de tarea
                task_dict["session"] = session
                task_dict["task_number"] = i

                res.append(task_dict)
                i += 1
            else:
                helper.info(f"Ignoring task line: {task_data}")

# Crear DataFrame y guardar CSV
df = pd.DataFrame(res)
os.makedirs(os.path.dirname(output_fname), exist_ok=True)  # Crear carpeta si no existe
df.to_csv(output_fname, index=False)

print(f"Se procesaron {len(res)} tareas. CSV guardado en {output_fname}.")
