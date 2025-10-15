import pandas as pd
from tqdm import tqdm

ipus = pd.read_csv('csvs/ipus_uba.csv', index_col="token_id")
tt = pd.read_csv('csvs/tt-table.csv')

tt = tt[tt["tt_label"] != "H"]

for name, group in tqdm(ipus.groupby(["session_number", "task"])):
        res = []
        for channel in ["A", "B"]:
            in_task_ipus_interlocutor = group[group.channel != channel].sort_values(by="ipu_start_time")
            in_task_ipus_locutor = group[group.channel == channel].sort_values(by="ipu_start_time")

            for i in range(0, len(in_task_ipus_locutor) - 1):
                ipu1 = in_task_ipus_locutor.iloc[i]
                ipu2 = in_task_ipus_locutor.iloc[i + 1]
                interruptions = in_task_ipus_interlocutor[~((in_task_ipus_interlocutor.ipu_end_time <= ipu1.ipu_end_time) | (in_task_ipus_interlocutor.ipu_start_time >= ipu2.ipu_start_time))]
                if len(interruptions) == 0:
                    res.append(dict(tt_label="H", session_number=ipu2.session_number, speaker2=ipu2.channel, task=ipu2.task, ipu2_start_time=ipu2.ipu_start_time, ipu2_end_time=ipu2.ipu_end_time, overlapped_transition=False))
            tt = pd.concat([tt, pd.DataFrame(res)], ignore_index=True)
tt.to_csv("csvs/holds-tt-table.csv", index=False)
print(tt.tt_label.value_counts())
