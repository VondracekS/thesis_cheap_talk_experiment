
def load_data(group):
    for res_file in os.listdir("../data/data_processed/"):
        if group in res_file:
            data_processed = pd.read_csv(f"../data/data_processed/{res_file}").drop("Unnamed: 0", axis=1)
            data_processed['cost'] = data_processed.groupby('id').cumcount()*20+10

    return data_processed