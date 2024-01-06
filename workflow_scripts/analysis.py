# %%
import os
import pandas as pd

# %%

def load_data(group):
    for res_file in os.listdir("../data/data_processed/"):
        if group in res_file:
            data_processed = pd.read_csv(f"../data/data_processed/{res_file}").drop("Unnamed: 0", axis=1)
            data_processed['cost'] = data_processed.groupby('id').cumcount()*20+10

    return data_processed

data_control_long = load_data('control_full_long')
data_treatment_long = load_data('treatment_full_long')

def get_counts(df, var):
    df_counts = (df[['cost']+ var]
                 .groupby('cost')
                 .value_counts()
                 .reset_index()
                 .rename(columns={0: 'count'}))
    return df_counts

def plot_value_counts(df, var, **kwargs):
    df_counts = get_counts(df, var)
    df_counts[var] = df_counts[var].apply(lambda x: x.astype('str'))
    df_counts = (df_counts
                 .melt(id_vars=['cost', 'count'])
                 .groupby(['cost', 'variable', 'value'])
                 .sum()
                 .reset_index())
    fig = px.bar(df_counts, x='cost', y='count', color='value', facet_row='variable', color_discrete_map={'2': '#0000dc', '1': '#BEBEBE'}, **kwargs)
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("variable=", "")))
    fig.add_vline(x=50, line_color='red', line_dash="dot", annotation_text="Altruistic Bound")
    fig.add_vline(x=150, line_color='red', line_dash="dot", annotation_text="Irrational Bound")
    fig.update_layout(height=200*len(df_counts['variable'].unique()), width=1000, xaxis={"tickmode": 'linear', 'tick0': 10, 'dtick': 20},
                      yaxis={"tickmode": 'linear', 'tick0': 0, 'dtick': 5})
    fig.update_yaxes(range=[0, 25])

    return fig

plot_treatment = plot_value_counts(data_treatment_long.loc[data_treatment_long['id'].isin(compliant_id_treatment.values)],
                                   ['p_choices_ca', 'p_choices_cm', 'prediction_ca', 'prediction_cm'], template='plotly_white')
plot_control = plot_value_counts(data_control_long.loc[data_treatment_long['id'].isin(compliant_id_control.values)],
                                 ['p_choices', 'prediction'], template='plotly_white')
#%%
plot_treatment.write_image("../plots/treatment_count_clean.png")
plot_control.write_image("../plots/control_count_clean.png")