import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

import os
alt.renderers.enable('notebook')
alt.renderers.enable('mimetype')


def plot_distributions(data, bins = 40, dist_cols=None, class_label=None):
        
    if data is None:
        raise ValueError("Required arg 'data' cannot be empty")
    
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Please enter data of type pd.DataFrame")

    if not isinstance(bins, int):
        raise TypeError("bins should be of type int")

    if dist_cols is not None:
        if not isinstance(dist_cols, list):
            raise TypeError("The entered dist_cols should be of 'list' type")
        else:
            numeric_features = dist_cols
    else:
        numeric_features = list(data.select_dtypes('number').columns)

    if class_label is None:

        chart_numeric = alt.Chart(data).mark_bar().encode(
        alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=bins)),
                y=alt.Y('count()')
                ).properties(
                    width=300,
                    height=200
                ).repeat(
                    numeric_features, columns = 4
                ) 
        
    else:
        if not type(class_label) == str:
            raise TypeError("`class_label` should be of string type")
        else:

            chart_numeric = alt.Chart(data).mark_bar(opacity=0.6).encode(
            alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=bins)),
                    y=alt.Y('count()', stack=False),
                    color=class_label+':N'
                    ).properties(
                        width=300,
                        height=200
                    ).repeat(
                        numeric_features, columns = 4
                    ) 
    return chart_numeric


def plot_corr(data, corr='spearman'):
   
    if data is None :
        raise ValueError("Required arg 'data' cannot be empty")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Please enter data of type pd.DataFrame")
    if corr == '':
        raise ValueError("'corr' cannot be empty")
    if corr not in ['spearman', 'pearson', 'kendall']:
        raise ValueError("corr should be one of these: 'spearman', 'pearson', 'kendall'")

    corr_df= data.corr(corr).stack().reset_index(name='corr')
    corr_plot = alt.Chart(corr_df, title='Correlation Plot among all features and target').mark_rect().encode(
        x=alt.X('level_0', title='All features'),
        y=alt.Y('level_1', title='All features'),
        tooltip='corr',
        color=alt.Color('corr', scale=alt.Scale(domain=(-1, 1), scheme='purpleorange')))
    return corr_plot
  

def plot_splom(data, pair_cols=None):
  
    if data is None :
        raise ValueError("Required arg 'data' cannot be empty")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Please enter data of type pd.DataFrame")
    if pair_cols==[]:
        raise ValueError("pair_cols should not be empty list")
    if pair_cols is None:
        pair_cols = list(data.select_dtypes('number').columns)
    else:
        if not isinstance(pair_cols, list):
            raise TypeError("The entered pair_cols should be of 'list' type")
    splom_chart = alt.Chart(data).mark_point(opacity=0.3, size=10).encode(
    alt.X(alt.repeat('row'), type='quantitative'),
    alt.Y(alt.repeat('column'), type='quantitative')
    ).properties(
        width=200,
        height=200
    ).repeat(
        column=pair_cols,
        row=pair_cols
    )
    return splom_chart
    