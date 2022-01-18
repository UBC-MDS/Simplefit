import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

def cleaner(data):
     """Load data and Clean data(remove Nan rows, strip extra white spaces from column names, and data, convert all column names to lower case, etc)
        Return clean data, train_df.info() and train_df.describe()
        Parameters
        ----------
        data : pandas.DataFrame
            The dataset provided by the user in .csv format 
  
        Returns
        -------
            (DataFrame):  A cleaned and simplified DataFrame of the relevant columns for summary and visualization,
            (DataFrame):  for train_df.describe()
            (DataFrame):  for train_df.info()
        Examples
        --------
        >>> cleaner(example_data)
    """


def get_eda(data, dist_cols=None, pair_cols=None, corr_method="pearson", text_col=None, class_label=None):
    """This function creates common exploratory analysis visualizations on numeric and categorical columns in the dataset which are provided to it.
       The following visualizations would be generated:
        1. Histograms for all numeric columns or for columns specified in optional paramter 'dist_cols'
        2. Scatterplot Matrix (SPLOM) for all numeric columns or for columns specified in optional paramter 'pair_cols'
        3. Heatmap showing correlation (default='pearson', 'kendall' or 'spearman') between all numeric columns and categorical columns separately
        4. Count plots for categorical columns
        Parameters
        ----------
        data : pandas.DataFrame
            The dataframe for which exploratory analysis is to be done
        dist_cols : list, optional
            The subset of numeric columns for which the histogram plots have to be generated
        pairplot_cols : list, optional
            The subset of numeric columns for which pairplots have to be generated
        corr_method : str, optional
            Method correlation. Default value is 'pearson'. 
            Other Possible values:
                * pearson : standard correlation coefficient
                * kendall : Kendall tau correlation coefficient
                * spearman : Spearman rank correlation
        text_col : list, optional
            The column containing free form of text, example: "Hi, I didn't wsnt to go there"
        class_label : str, optional
            The name of the target column only in case of classification dataset. For regression dataset, it is not required
        Returns
        -------
        list
            A list of plot objects created by this function
        Examples
        -------
        >>> get_eda(df)
        >>> get_eda(df, pair_cols = ['danceability', 'loudness'], corr_method='kendall', class_label='target')
    """

def regressor(train_df, target_col, numeric_feats = None, categorical_feats=None, text_col=None, cv=5):
    """This function preprocess the data, fit baseline model(dummyregresor) and ridge with default setups to provide data scientists 
        easy access to the common models results(scores). 

        Parameters
        ----------
        train_df : pandas.DataFrame
            The clean train data which includes target column.
        target_col : str
            The column of the train data that has the target values.
        numeric_feats = list, optional
            The numeric features that needs to be considered in the model. If the user do not define this argument, the function will assume all the columns except the identified ones in other arguments as numeric features.
        categorical_feats : list, optional
            The categorical columns for which needs onehotencoder preprocessing.  
        text_col : list, optional
            The column containing free form of text, example: "Hi, I wasn't to go there" for doing countvectorizer preprocessing .
        cv : int, optional
            The number of folds on the data for train and validation set.

        Returns
        -------
        Data frame
            A data frame that includes test scores and train scores for each model.
        Examples
        -------
        >>> regressor(train_df, target_col = 'popularity', categorical_features='None')
        >>> regressor(train_df, target_col = 'popularity', numeric_feats = ['danceability', 'loudness'], categorical_feats=['genre'], text_col='track_name', cv=10)
    """


    X_train = train_df.drop(columns=target_col, axis=1)
    y_train = train_df[target_col[0]]
    
    text_feat = text_col[0]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_feats),
        (OneHotEncoder(), categorical_feats),
        (CountVectorizer(stop_words="english"), text_feat)
    )

    dummy = DummyRegressor()
    ridge = make_pipeline(preprocessor, Ridge())
    ridge_cv = make_pipeline(preprocessor, RidgeCV())
    lr = make_pipeline(preprocessor, LinearRegression())

    results = pd.Series(dtype='float64') 

    models = {"DummyRegressor": dummy, "Ridge" : ridge, "RidgeCv" : ridge_cv, "linearRegression" : lr}

    for model in models :
        scores = cross_validate(models[model], X_train, y_train, cv = cv)
        mean_scores = pd.DataFrame(scores).mean()
        results = pd.concat([results, mean_scores], axis = 1)
    results = results.drop(columns = 0, axis=1)
    results.columns = models.keys
    
    return results


def classifier(train_df, target_col, numeric_feats = None, categorical_feats=None, text_col=None, cv=5):
    """This function preprocess the data, fit baseline model(dummyclassifier) and logistic regression with default setups to provide data scientists 
        easy access to the common models results(scores). 
        Parameters
        ----------
        train_df : pandas.DataFrame
            The clean train data which includes target column.
        target_col : str
            The column of the train data that has the target values.
        numeric_feats = list, optional
            The numeric features that needs to be considered in the model. If the user do not define this argument, the function will assume all the columns except the identified ones in other arguments as numeric features.
        categorical_feats : list, optional
            The categorical columns for which needs onehotencoder preprocessing.  
        text_col : list, optional
            The column containing free form of text, example: "Hi, I wasn't to go there" for doing countvectorizer preprocessing.
        cv : int, optional
            The number of folds on the data for train and validation set.
        Returns
        -------
        Data frame
            A data frame that includes test scores and train scores for each model.
        Examples
        -------
        >>> classifier(train_df, target_col = 'target', categorical_features='None')
        >>> classifier(train_df, target_col = 'target', numeric_feats = ['danceability', 'loudness'], categorical_feats=['genre'], text_col='track_name', cv=10)
    """




