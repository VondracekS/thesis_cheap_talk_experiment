import os
import pandas as pd
import numpy as np
import tqdm
import random
from random import choices


def calculate_payoffs(cost_player: int, action_player: str, action_opponent: str) -> int:
    """
    The function calculates payoffs from our game. It is applicable for both control and treatment group
    :param cost_player: determines the cost of the player, must be in the range(10, 190)
    :param action_player:
    :param action_opponent:
    :return:
    """
    # assert cost_player in range(10, 200, 10) and isinstance(cost_player, int), print(f'Error: Incorrect cost_player value: {cost_player}')
    assert action_player in {1, 2}, print(f'Error: Incorrect action_player value: {action_player}')
    assert action_opponent in {1, 2}, print(f'Error: Incorrect action_opponent value: {action_opponent}')

    if action_player == 1 and action_opponent == 1:
        payoff = 200 - cost_player
    if action_player == 1 and action_opponent == 2:
        payoff = 250 - cost_player
    if action_player == 2 and action_opponent == 2:
        payoff = 200
    if action_player == 2 and action_opponent == 1:
        payoff = 50

    return payoff


def gameplay(prob_1_b: int = 0.5, prob_2_b: int = 0.5) -> pd.DataFrame:
    """
    Given the opponent type (his probability to play action B), the function
    simulates one round of the game
    :param prob_1_b: Probability that the player 1 would play action B
    :param prob_2_b: Probability that the player 2 would play action B
    :return:
    """
    cost_pool = list(range(10, 200, 20))  # permissible pool of costs
    choice_pool = [1, 2]  # permissible pool of choices
    cost_1 = choices(cost_pool)[0]  # randomly selected cost for the Player A
    action_1 = choices(choice_pool, weights=[1 - prob_1_b, prob_1_b], k=1)[0]
    action_2 = choices(choice_pool, weights=[1 - prob_2_b, prob_2_b], k=1)[0]

    df_res = pd.DataFrame(
        {"cost": cost_1,
         "action_a": [action_1],
         "payoff": [calculate_payoffs(cost_1, action_1, action_2)]}
    )

    return df_res


def simulate_game(n_iter: int = 1000) -> pd.DataFrame:
    """
    Given the number of iterations per probability scale unit, the function runs a simulation of the above described game.
    In the end, we get a set of simulations, where each player played n_iter times with a probability taken from the range(0, 1, 0.1)
    The total number of iteration is following:
    length of range * length of range * n_iter. For case of n_iter=100, it is 9 * 9 * 100, i.e. 8100 iterations
    :param n_iter: number of iterations per probability unit
    :return:
    """
    gameplays = []
    for prob_1 in np.round(np.arange(0.1, 1, 0.2), 1):
        for prob_2 in np.round(np.arange(0.1, 1, 0.2), 1):
            df_sim = []
            for i in range(n_iter):
                df_sim.append(gameplay(prob_1, prob_2))
            df_sim = pd.concat(df_sim)
            df_sim['prob_1'] = prob_1
            df_sim['prob_2'] = prob_2
            gameplays.append(df_sim)
    gameplays = pd.concat(gameplays)
    return gameplays


def load_data(group):
    for res_file in os.listdir("../data/data_processed/"):
        if group in res_file:
            data_processed = pd.read_csv(f"../data/data_processed/{res_file}").drop("Unnamed: 0", axis=1)
            data_processed['cost'] = data_processed.groupby('id').cumcount() * 20 + 10

    return data_processed


def generate_simulated_pairs_result_control(experiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly samples from our experiment data and selects
    :param experiment_df: Experiment data
    :param id_col: Id column of the experiment_df
    :param action: Which action (i.e. prediciton about opponent's action, actual game choice
    :return: simulated
    """
    pairs_df = pd.DataFrame()
    ids = list(experiment_df['id'].unique())
    for p in range(len(ids) // 2):
        df_tmp = pd.DataFrame([{
            'id_1': ids.pop(random.randrange(len(ids))),
            'id_2': ids.pop(random.randrange(len(ids))),
            'cost_1': random.choices(list(range(10, 200, 20)))[0],
            'cost_2': ranodm.choices(list(range(10, 200, 20)))[0]
        }])

        df_tmp[f'action_1'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_1'].squeeze()) &
                                                (experiment_df['cost'] == df_tmp['cost_1'].squeeze())][
            'p_choices'].squeeze()
        df_tmp[f'action_2'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_2'].squeeze()) &
                                                (experiment_df['cost'] == df_tmp['cost_2'].squeeze())][
            'p_choices'].squeeze()

        df_tmp['prediction_1'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_1'].squeeze()) &
                                                   (experiment_df['cost'] == df_tmp['cost_2'].squeeze())][
            'prediction'].squeeze()
        df_tmp['prediction_2'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_2'].squeeze()) &
                                                   (experiment_df['cost'] == df_tmp['cost_1'].squeeze())][
            'prediction'].squeeze()

        df_tmp['payoff_1'] = df_tmp.apply(lambda x: calculate_payoffs(x.cost_1, x['action_1'], x['action_2']), axis=1)
        df_tmp['payoff_2'] = df_tmp.apply(lambda x: calculate_payoffs(x.cost_2, x['action_2'], x['action_1']), axis=1)

        pairs_df = pd.concat([pairs_df, df_tmp])

    return pairs_df.reset_index(drop=True)


def generate_gameplay_control(n_iter: int, df_experiment: pd.DataFrame) -> pd.DataFrame:
    """
    Iteratively creates n_iter iterations of the game with repetion
    :param n_iter: number of iterations
    :param df_experiment: original experiment results
    :param id_col: id column of the experiment df
    :param action: action to be simulated
    :return: concatenated data frames with gameplay results
    """

    gameplay = []
    for i in tqdm.tqdm(range(n_iter)):
        game_round = generate_simulated_pairs_result_control(df_experiment)
        game_round.index = [f"r_{i}_{ix}" for ix in game_round.index]
        gameplay.append(game_round)
    return pd.concat(gameplay)


def generate_simulated_pairs_result_treatment(experiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Randomly samples from our experiment data and selects
    :param experiment_df: Experiment data
    :param id_col: Id column of the experiment_df
    :return: simulated
    """
    pairs_df = pd.DataFrame()
    ids = list(experiment_df['id'].unique())
    action_col = {1: 'p_choices_ca', 2: 'p_choices_cm'}
    pred_col = {1: 'prediction_ca', 2: 'prediction_cm'}

    for p in range(len(ids) // 2):
        df_tmp = pd.DataFrame([{
            'id_1': ids.pop(random.randrange(len(ids))),
            'id_2': ids.pop(random.randrange(len(ids))),
            'cost_1': choices(list(range(10, 200, 20)))[0],
            'cost_2': choices(list(range(10, 200, 20)))[0]
        }])

        df_tmp['message_1'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_1'].squeeze()) &
                                                (experiment_df['cost'] == df_tmp['cost_1'].squeeze())][
            'message'].squeeze()
        df_tmp['message_2'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_2'].squeeze()) &
                                                (experiment_df['cost'] == df_tmp['cost_2'].squeeze())][
            'message'].squeeze()

        df_tmp['action_1'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_1'].squeeze()) &
                                               (experiment_df['cost'] == df_tmp['cost_1'].squeeze())][
            action_col[df_tmp['message_2'].squeeze()]].squeeze()
        df_tmp['action_2'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_1'].squeeze()) &
                                               (experiment_df['cost'] == df_tmp['cost_2'].squeeze())][
            action_col[df_tmp['message_1'].squeeze()]].squeeze()

        df_tmp['prediction_1'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_1'].squeeze()) &
                                                   (experiment_df['cost'] == df_tmp['cost_2'].squeeze())][
            pred_col[df_tmp['message_1'].squeeze()]].squeeze()
        df_tmp['prediction_2'] = experiment_df.loc[(experiment_df['id'] == df_tmp['id_2'].squeeze()) &
                                                   (experiment_df['cost'] == df_tmp['cost_1'].squeeze())][
            pred_col[df_tmp['message_2'].squeeze()]].squeeze()

        df_tmp['payoff_1'] = df_tmp.apply(lambda x: calculate_payoffs(x.cost_1, x.action_1, x.action_2), axis=1)
        df_tmp['payoff_2'] = df_tmp.apply(lambda x: calculate_payoffs(x.cost_2, x.action_2, x.action_1), axis=1)

        df_tmp['prediction_1_correct'] = df_tmp['prediction_1'] == df_tmp['action_2']
        df_tmp['prediction_2_correct'] = df_tmp['prediction_2'] == df_tmp['action_1']

        pairs_df = pd.concat([pairs_df, df_tmp])

    return pairs_df.reset_index(drop=True)


def generate_gameplay_treatment(n_iter: int, df_experiment: pd.DataFrame) -> pd.DataFrame:
    """
    Iteratively creates n_iter iterations of the game with repetion
    :param n_iter: number of iterations
    :param df_experiment: original experiment results
    :param id_col: id column of the experiment df
    :param message: action to be simulated
    :return: concatenated data frames with gameplay results
    """
    gameplay = []

    for i in tqdm.tqdm(range(n_iter)):
        game_round = generate_simulated_pairs_result_treatment(df_experiment)
        game_round.index = [f"r_{i}_{ix}" for ix in game_round.index]
        gameplay.append(game_round)

    return pd.concat(gameplay)


def is_pareto_effective(df_simulation):
    pareto_res = []
    for ix, row in tqdm.tqdm(df_simulation.iterrows(), total=len(df_simulation)):
        action_1_counterfactual = 3 - row['action_1']
        action_2_counterfactual = 3 - row['action_2']
        payoff_1_counterfactual = calculate_payoffs(row['cost_1'], action_1_counterfactual, row['action_2'])
        payoff_2_counterfactual = calculate_payoffs(row['cost_2'], action_2_counterfactual, row['action_1'])
        pareto_res.append(int(row['payoff_1'] > payoff_1_counterfactual and row['payoff_2'] > payoff_2_counterfactual))
    return pareto_res


def is_action_col(col):
    return set(col.unique()) == {2, 1}


def reformat_action_values(df):
    df = df.apply(lambda x: x - 1 if is_action_col(x) else x)
    return df


def gameplay_simulation_to_long(gameplay_df):
    gameplay_df['opp_1_cost'] = gameplay_df['cost_2']
    gameplay_df['opp_2_cost'] = gameplay_df['cost_1']
    df_1 = gameplay_df.loc[:, [col for col in gameplay_df if '1' in col or col == 'pareto_effective']]
    df_2 = gameplay_df.loc[:, [col for col in gameplay_df if '2' in col or col == 'pareto_effective']]

    df_1.columns = [col.split('_')[0] for col in df_1.columns]
    df_2.columns = [col.split('_')[0] for col in df_2.columns]

    df_complete = (pd.concat([df_1, df_2])
                   .reset_index(drop=True)
                   .rename(columns={'action': 'p_choices'}))

    return df_complete


def calculate_final_payoff(row):
    if random.choice([0, 1]) == 0:
        return row['payoff']
    else:
        return 50 if row['correct'] == 0 else 200


def get_occurrence_matrix(df, id_1: str = 'id_1', id_2: str = 'id_2'):
    unique_values_id_1 = df[id_1].unique()
    unique_values_id_2 = df[id_2].unique()

    # Create an empty co-occurrence matrix
    co_occurrence_matrix = pd.DataFrame(0, index=unique_values_id_1, columns=unique_values_id_2)

    # Loop through the rows and update the matrix
    for index, row in df.iterrows():
        id_1_value = row[id_1]
        id_2_value = row[id_2]
        co_occurrence_matrix.at[id_1_value, id_2_value] += 1
    df_out = pd.DataFrame(co_occurrence_matrix).sort_index()

    return df_out[sorted(df_out.columns)]
