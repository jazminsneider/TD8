import glob
import pandas as pd
import os
import helper
import ml.utils


output_fname="csvs/tt-table.csv"
task_table="csvs/tasks_uba.csv"
ipus_table="csvs/ipus_uba.csv"
overlapped_transition={"O","I","BI","BC_O","X2_O"}

tasks = pd.read_csv(task_table)

# inputs:
# folder in which the ".turns" file are located
corpus_folder = "games-corpus/.uba-games/b1-dialogue-turns/"
turns_files_regex = "*.turns"
turns_files = sorted(list(glob.glob("{}{}".format(corpus_folder, turns_files_regex))))
assert len(turns_files) != 0

info = []
ipus_table = pd.read_csv(ipus_table)

for turns_file in turns_files:
    turns = ml.utils.read_turns(turns_file)

    session, _, object_id, channel, _ = turns_file.split("/")[-1].split(".")
    print(turns_file.split("/")[-1].split("."))
    session = session[1:]
    print(session)
    tasks_session = tasks[tasks.session == int(session)]
    for (t0, tf, tt_label) in turns:
        print(t0, tf, tt_label)
        task = tasks_session[(tasks_session.t0 <= t0) & (tasks_session.tf >= t0)]
        print(task.t0)
        print(len(task))
        if len(task) == 0:
            helper.warning("ignoring out-of-task transition for s{} at {} ({})".format(session, t0, tt_label))
            continue

        assert len(task) == 1
        task_id = task.iloc[0].task_number
        info.append(dict(tt_label=tt_label,
                            session_number=session,
                            speaker2=channel,
                            task=task_id,
                            ipu2_start_time=round(t0, 4),
                            ipu2_end_time=round(tf, 4),
                            overlapped_transition=tt_label in overlapped_transition
                            ))
df = pd.DataFrame(info)
to_other = {"A", "N", "L", "L-SIM"}
df["tt_label"] = df["tt_label"].apply(lambda x: "other" if x in to_other else x)

df.to_csv(output_fname, index=None)
helper.info("File saved at {}".format(output_fname))
helper.info(df.tt_label.value_counts())
