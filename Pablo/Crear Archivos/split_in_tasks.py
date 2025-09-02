def debug(*msg):
    logger = logging.getLogger(__name__)
    logger.debug(" ".join([str(x) for x in msg]))

def read_list(filename, verbose=True):
    if not os.path.exists(filename):
        raise Exception("missing list: {}".format(filename))

    file = open(filename, "r")
    if verbose:
        debug("reading {}".format(filename))

    res = []
    lines = file.readlines()
    for line in lines:
        if line.strip() != "":
            chunks = line.split()
            if len(chunks) > 1:
                res.append(chunks)
            else:
                res.append(chunks[0])
    return list(res)


def save_list(lines, fname, verbose=True, separator="\t", append=False):
    def _multiple_line(line):
        return type(line) is list or type(line) is tuple or type(line) is np.ndarray
    if append:
        mode = "a"
    else:
        mode = "w"
    with open(fname, mode) as fn:
        for line in lines:
            if _multiple_line(line):
                fn.write(separator.join([str(l) for l in line]) + "\n")
            else:
                fn.write(line + "\n")
    if verbose:
        debug("saving {}".format(fname))



tasks = pd.read_csv('csvs/tasks_uba.csv')
tokens_list = []
output_folder = "tasks_features"
for sess_channel, track_fname in tqdm(read_list("CAMBIAR POR DIR DEL ARCHIVO")):
        sess, channel = sess_channel.split("_")
        tasks_for_session = tasks[tasks.session == int(sess)]
        if channel == "A":
            features_A = pd.read_csv(track_fname, index_col="time")
            features_B = pd.read_csv(track_fname.replace("_A", "_B"), index_col="time")

            for _, task_info in tasks_for_session.iterrows():
                task_id = task_info.task_number

                sess_task = "{}_{}".format(sess, task_id)
                output_fname = "{}/{}_{}.csv".format(output_folder, sess, task_id)
                features_task_A = features_A.loc[features_A.task == task_id, :].drop("task", axis=1)
                features_task_B = features_B.loc[features_B.task == task_id, :].drop("task", axis=1)
                joined = features_task_A.join(features_task_B, lsuffix="_A", rsuffix="_B")
                joined.to_csv(output_fname)
                tokens_list.append((sess_task, output_fname))
                
            save_list(tokens_list, "tasks_features.lst")
