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


def classifier(train_df, target_col, numeric_feats = None, categorical_feats=None, text_col=None, cv=5):
    """This function preprocess the data, fit baseline model(dummyclassifier) and ridge with default setups to provide data scientists 
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
        >>> classifier(train_df, target_col = 'popularity', categorical_features='None')
        >>> classifier(train_df, target_col = 'popularity', numeric_feats = ['danceability', 'loudness'], categorical_feats=['genre'], text_col='track_name', cv=10)
    """




