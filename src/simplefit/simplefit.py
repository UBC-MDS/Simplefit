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


def get_eda(data, save_plot_loc, bins=40, dist_cols=None, pair_cols=None, corr_method="spearman", text_col=None, class_label=None):
    """This function creates common exploratory analysis visualizations on numeric and categorical columns in the dataset which are provided to it and 
       saves them in the location provided by the user in function as input
       The following visualizations would be generated:
        1. Histograms for all numeric columns or for columns specified in optional paramter 'dist_cols'
        2. Scatterplot Matrix (SPLOM) for all columns or specified columns in optional paramter 'pair_cols'
        3. Heatmap showing correlation (default='spearman', 'kendall' or 'pearson') between all columns
        
        Parameters
        ----------
        data : pandas.DataFrame
            The dataframe for which exploratory analysis is to be done
        save_plot_loc : str
            The location where you want to save the plots
        dist_cols : list, optional
            The subset of numeric columns for which the histogram plots have to be generated
        pairplot_cols : list, optional
            The subset of numeric columns for which pairplots have to be generated
        corr_method : str, optional
            Method correlation. Default value is 'spearman'. 
            Other Possible values:
                * pearson : standard correlation coefficient
                * kendall : Kendall tau correlation coefficient
                * spearman : Spearman rank correlation
        text_col : list, optional
            The column containing free form of text, example: "Hi, I didn't want to go there"
        class_label : str, optional
            The name of the target column only in case of classification dataset. For regression dataset, it is not required
        Returns
        -------
        pd.DataFrame
            A dataframe containing the summary statistics of the columns
        Examples
        -------
        >>> get_eda(df, save_plot_loc='/Users/navyadahiya/desktop/eda/')
        >>> get_eda(df, save_plot_loc='/Users/navyadahiya/desktop/eda/', pair_cols = ['danceability', 'loudness'], corr_method='kendall', class_label='target')
    """
    try:
 
        if data is None :
            raise ValueError("Required arg 'data' cannot be empty")
        if save_plot_loc is None or save_plot_loc== '':
            raise ValueError("Please provide a valid path in your system to save the plots. Example: 'Users/navyadahiya/desktop/''")
            # sys.exit(1)

        #check validity of file path, write test case
        if save_plot_loc is not None:

            #check validity of location:
            if not save_plot_loc.endswith('/'):
                raise ValueError("Please make sure that your location  ends with '/'. example: 'Users/navyadahiya/desktop/'")
            pathExists = os.path.exists(save_plot_loc)
            if not pathExists:
                os.makedirs(save_plot_loc)

            for file in os.listdir(save_plot_loc):
                if file.endswith('.png'):
                    os.remove(save_plot_loc+"/"+file)
            
        # check if data is of dataframe type or not:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input df is not of type pd.DataFrame")

        numeric_features = list(data.select_dtypes('number').columns)

        if text_col is not None:

            categorical_features = set(data.columns)-set(numeric_features)-set(text_col)
        else:
            categorical_features = set(data.columns)-set(numeric_features)
        
        if dist_cols is not None:
            numeric_features = dist_cols

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

        chart_numeric.save(save_plot_loc+'/numeric_feature_eda.png', scale_factor=2.0)
    
        
        ## correlation plot
        corr_df= data.corr(corr_method).stack().reset_index(name='corr')
        corr_plot = alt.Chart(corr_df, title='Correlation Plot among all features and target').mark_rect().encode(
            x=alt.X('level_0', title='All features'),
            y=alt.Y('level_1', title='All features'),
            tooltip='corr',
            color=alt.Color('corr', scale=alt.Scale(domain=(-1, 1), scheme='purpleorange')))
        corr_plot.save(save_plot_loc+'/correlation_plot.png')
    
    
        if pair_cols is None:
            pair_cols = numeric_features
    
        #splom plot
        splom = alt.Chart(data).mark_point(opacity=0.3, size=10).encode(
        alt.X(alt.repeat('row'), type='quantitative'),
        alt.Y(alt.repeat('column'), type='quantitative')
        ).properties(
            width=200,
            height=200
        ).repeat(
            column=pair_cols,
            row=pair_cols
        )
        splom.save(save_plot_loc+'/splom_plot.png')
        
        return data.describe(include='all')
    except Exception as e:
        print(e)



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




