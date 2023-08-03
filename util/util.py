import pandas as pd
import os


class DataWranglingUtils:
    """
    Class containing util functions to be used to wrangle the data from the experiment
    """

    @classmethod
    def load_data(cls, group: str) -> dict:
        """
        Loads the experiment data stored in .csv format
        :param group: determines whether data from control or treatment should be loaded
        :return: dictionary having the filename as key
        """
        assert group in ['control', 'treatment']
        data_processed = {}
        for res_file in os.listdir("../data/data_processed/"):
            if group in res_file:
                data_processed[str(res_file).split(".")[0]] = pd.read_csv(f"../data/data_processed/{res_file}").drop(
                    "Unnamed: 0", axis=1)

        print(f"Loaded data: {data_processed.keys()}")
        return data_processed

    @classmethod
    def data_into_df(cls, df_dict: dict) -> pd.DataFrame:
        """
        Concatenates the data frames from the dictionary
        """
        df_out = pd.concat(df_dict.values(), axis=1)
        df_out = df_out.loc[:, ~df_out.columns.duplicated()]
        return df_out

    @classmethod
    def get_player_type(cls, row_value: list) -> str:
        """
        Based on the shape and content of a list, determines the player type
        """
        if len(row_value) > 1:
            return 'irrational'
        if len(row_value) == 0:
            return 'static'
        if row_value[0][1] == 1:
            return 'aggresive'
        if row_value[0][1] == 2:
            return 'peaceful'

    @classmethod
    def get_cutoff(cls, df: pd.DataFrame, action_col: str, cost_col: str = 'cost', id_col: str = 'id'):
        """
        To each unique id, assign all costs at which his strategy has changed
        """
        cutoff = {k: [] for k in df[id_col].unique()}
        for i in range(len(df[cost_col]) - 1):
            if df[action_col][i + 1] != df[action_col][i] and df[id_col][i + 1] == df[id_col][i]:
                cutoff[df[id_col][i]].append((df[cost_col][i], df[action_col][i]))

        df_cutoffs = pd.DataFrame({'id': cutoff.keys(),
                                   f'{action_col}_cutoff': [v for v in cutoff.values()]})

        df_cutoffs[f'{action_col}_player_type'] = df_cutoffs[f'{action_col}_cutoff'].apply(cls.get_player_type)

        return df_cutoffs

    @classmethod
    def get_all_cutoffs(cls, df: pd.DataFrame, action_cols: list[str]) -> pd.DataFrame:
        df_out = cls.get_cutoff(df, action_cols[0])
        for col in action_cols[1:]:
            df_out = df_out.merge(cls.get_cutoff(df, col))
        return df_out

    @classmethod
    def get_first_cutoffs(cls, df: pd.DataFrame, cutoff_cols: list[str]) -> pd.DataFrame:
        df_out = df.copy()
        for cutoff_col in cutoff_cols:
            df_out[f'{cutoff_col}_first'] = df_out[cutoff_col].apply(lambda x: x[0][0] if len(x) != 0 else None)
        return df_out#

    @classmethod
    def get_combination_count(cls, df: pd.DataFrame, action_cols: list[str]):
        return df.groupby(action_cols).size().reset_index().rename(columns={0: 'combination_count'})
