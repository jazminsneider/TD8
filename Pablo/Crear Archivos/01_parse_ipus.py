import pandas as pd
import numpy as np
import glob
import os

phrases_folder = 'games-corpus/.uba-games/b1-dialogue-phrases'  # Carpeta con los .phrases
phonetic_dict_path = 'games-corpus/.uba-games/phonetic-dictionary-games.txt'
tasks_csv = 'csvs/tasks_uba.csv'  # El CSV generado antes
subjects_info_csv = 'csvs/subjects_info_uba.csv'  # Info de sujetos


#Leer el dict
def read_list(fname):
    with open(fname, encoding='utf-8') as f:
        lines = [line.strip().split() for line in f if line.strip()]
    return lines

phonemes_dictionary = read_list(phonetic_dict_path)
words_phones_count = {}
for row in phonemes_dictionary:
    if len(row) >= 2:
        word = row[1]
        try:
            cuenta = len(row[2:])
        except ValueError:
            continue
        words_phones_count[word] = cuenta
#leemos los archivos auxiliares
tasks = pd.read_csv(tasks_csv)
speakers_info = pd.read_csv(subjects_info_csv,sep=";", index_col="sessionID")



#procesas phrases
def read_wavesurfer(fname):
    with open(fname, encoding='utf-8') as f:
        lines = [line.strip().split(maxsplit=2) for line in f if line.strip()]
    # Cada línea: [start_time, end_time, phrase]
    return [(float(l[0]), float(l[1]), l[2]) for l in lines if len(l) == 3]

silence_tag = "#"
ipus = []
phrases_files = sorted(glob.glob(os.path.join(phrases_folder, '*.phrases')))

for filename in phrases_files:
    file_id = os.path.basename(filename).split(".")[0]  # s01.objects.1
    file_id = ".".join(os.path.basename(filename).split(".")[:-1])
    assert("A" in file_id or "B" in file_id)
    channel = "A" if "A" in file_id else "B"
    session = file_id[1:3]
    session_tasks = tasks[tasks.session == int(session)]
    speaker = speakers_info.loc[int(session), f"idSpeaker{channel}"]
    speaker_gender = speakers_info.loc[int(session), f"genderSpeaker{channel}"]
    phrases = [x for x in read_wavesurfer(filename) if x[2] != silence_tag]
    for (ipu_t0, ipu_tf, phrase) in phrases:
        dur = ipu_tf - ipu_t0
        phones_in_phrase = 0
        words_in_phrase = 0
        words = phrase.split(" ")
        out_of_dict_words = [w for w in words if w not in words_phones_count]
        if dur == 0:
            helper.warning("Ignoring empty IPU")
            continua
        if len(out_of_dict_words) > 0:
            helper.info("missing words in dict {}".format(out_of_dict_words))
            phones_in_phrase = np.nan
        else:
            for w in words:
                phones_in_phrase += words_phones_count[w]
                words_in_phrase += 1
        words_by_sec = round(words_in_phrase / dur, 4)
        phones_by_sec = round(phones_in_phrase / dur, 4)
        # Buscá la tarea correspondiente
        task_id = int(file_id.split(".")[2])
        task_found = session_tasks.loc[(session_tasks.task_number == task_id)]
        assert len(task_found) == 1
        task = task_found.iloc[0]  
        row = dict(
            token_id=f"{file_id}.task.{task.task_number}.t0.{ipu_t0:.4f}.tf.{ipu_tf:.4f}",
            ipu_start_time=round(ipu_t0, 4),
            ipu_end_time=round(ipu_tf, 4),
            task_start_time=task.t0,
            task_end_time=task.tf,
            task=task.task_number,
            duration=round(dur, 4),
            session_channel=f"{session}_{channel}",
            session_channel_task=f"{session}_{channel}_{task.task_number}",
            session_task=f"{session}_{task.task_number}",
            channel=channel,
            task_describer=task.Describer,
            task_target=task.Target,
            task_score=task.Score,
            corpus="uba",
            speaker=speaker,
            speaker_gender=speaker_gender,
            session_number=session,
            words_count=words_in_phrase,
            phones_count=phones_in_phrase,
            words_by_second=words_by_sec,
            phones_by_second=phones_by_sec, 
            wav=filename
        )
        ipus.append(row)

#guardamos en un nuevo .csv
df = pd.DataFrame(ipus)
df.to_csv('csvs/ipus_uba.csv', index=False)
