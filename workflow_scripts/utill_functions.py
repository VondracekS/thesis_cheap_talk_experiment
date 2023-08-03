import re
import pandas as pd
from abc import abstractmethod


class ExperimentDataExtractor:

    def __init__(self, data_dict):

        self.data_dict = data_dict

    def create_gameplay(data: dict) -> dict:
        """
        Create a new dictionary containing only the information desribing how the particular player had played
        """

        data_gameplay = {}
        for cat, df in data.items():
            data_gameplay[cat] = df.filter(
                regex=".*participant\.(prediction.*|p_choices.*)").transpose().dropna().transpose()
            data_gameplay[cat].columns = [col.replace("participant.", "") for col in data_gameplay[cat].columns]

        return data_gameplay

    @abstractmethod
    def extract_actions_from_string(self, col):
        """
        All actions are stored as strings. This function
        is meant to convert these strings to list

        :param col: affected column
        :return: series of transformed data
        """
        col_out = col.copy()

        i = 0
        for val in col_out.values:
            col_out.values[i] = re.findall(r"\d+", val)
            i += 1
        return col_out

    @abstractmethod
    def create_gameplay(data: dict) -> dict:
        data_gameplay = {}
        for cat, df in data.items():
            data_gameplay[cat] = df.filter(
                regex=".*participant\.(prediction.*|p_choices.*)").transpose().dropna().transpose()
            data_gameplay[cat].columns = [col.replace("participant.", "") for col in data_gameplay[cat].columns]
        return data_gameplay

    @abstractmethod
    def extract_actions_from_string(col):
        """
        All actions are stored as strings. This function
        is meant to convert these strings to list

        :param col: affected column
        :return: series of transformed data
        """
        col_out = col.copy()

        i = 0
        for val in col_out.values:
            col_out.values[i] = re.findall(r"\d+", val)
            i += 1
        return col_out

    @abstractmethod
    def get_gameplay(df, col, id_var):
        """
        Purpose:
        Explode the data frame on selected column, i.e. convert the data frame to a long format instead
        """
        df_out = df.explode(col)[col].reset_index().rename(columns={"index": id_var})

        return df_out


    def set_costs(df, cost_list=[val for val in range(10, 200, 20)]):
        """
        Purpose:
        Add column denoting the costs for particular player
        """
        df_out = df.copy()
        n_iter = int(df_out.shape[0] / len(cost_list))

        df_out['cost'] = n_iter * cost_list

        return df_out


    def split_df_by_index(df, id='id'):
        """
        By unique index values, unnest the df into a dictionary of dfs
        """
        dfs_dict = {}
        df[id] = df[id].astype(int)
        for idx in df[id].unique():
            dfs_dict[idx] = df.loc[df[id] == idx].reset_index(drop=True)

        return dfs_dict
