from simplefit.regressor import regressor

import pandas as pd
from sklearn.model_selection import (train_test_split,)
import pytest

def test_regressor():
    """Test regrssor function outputs with SpotifyFeatures.csv file."""

    # spotify_df = pd.read_csv("tests/data/SpotifyFeatures.csv")
    # train_df, test_df = train_test_split(spotify_df, test_size=0.97, random_state=123)
    # regressor_df = regressor(train_df, target_col="popularity",
    #                         numeric_feats=['acousticness', 'danceability', 'duration_ms'],
    #                         categorical_feats=['genre'])
    # actual_test_score = regressor_df.loc["test_score"].tolist()
    # expected_test_score = [-0.0015605122150153417, 0.7312349806310714, 0.7313441392629443, 0.731393854470928]
    # actual_train_score = regressor_df.loc["train_score"].tolist()
    # expected_train_score = [0.0, 0.7347815872733937, 0.7349114231911118, 0.734915756374242]
    # assert actual_test_score == expected_test_score, "regressor modeled incorrectly test scores are not equal to what they should be!"
    # assert actual_train_score == expected_train_score, "regressor modeled incorrectly train scores are not equal to what they should be!"


    spotify_df = pd.read_csv("tests/data/SpotifyFeatures.csv")
    train_df, test_df = train_test_split(spotify_df, test_size=0.97, random_state=123)
    regressor_df = regressor(train_df, target_col="popularity",
                            numeric_feats=['acousticness', 'danceability', 'duration_ms'],
                            categorical_feats=['genre'])
    actual_test_score = regressor_df.loc["test_score"].tolist()
    expected_test_score = [-0.0015605122150153417, 0.7312349350768665, 0.7313442775914065, 0.731393854472432]
    actual_train_score = regressor_df.loc["train_score"].tolist()
    expected_train_score = [0.0, 0.7347815894858526, 0.7349114221789114, 0.734915756374242]
    assert actual_test_score == expected_test_score, "regressor modeled incorrectly test scores are not equal to what they should be!"
    assert actual_train_score == expected_train_score, "regressor modeled incorrectly train scores are not equal to what they should be!"

def test_regressor_error() :
    """
    Test edges cases 
    4 tests in total.
    """
    spotify_df = pd.read_csv("tests/data/SpotifyFeatures.csv")

    with pytest.raises(TypeError) as e:
        regressor(1, target_col = "popularity", numeric_feats=['acousticness'])
    assert str(e.value) == "train_df must be a pandas dataframe. Please pass a pd.core.frame.DataFrame train_df."

    with pytest.raises(TypeError) as e:
        regressor(spotify_df, target_col = 1, numeric_feats=['acousticness'])
    assert str(e.value) == "target_col must be a str. Please pass target column in str object."

    with pytest.raises(TypeError) as e:
        regressor(spotify_df, target_col = "popularity", numeric_feats=1)
    assert str(e.value) == "numeric_feats must be a list. Please pass a list of numeric columns."

    with pytest.raises(TypeError) as e:
        regressor(spotify_df, target_col = "popularity", numeric_feats=['acousticness'],categorical_feats=1)
    assert str(e.value) == "categorical_feats must be a list. Please pass a list of categorical columns."
