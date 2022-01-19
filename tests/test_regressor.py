from simplefit import regressor

import pandas as pd

import pytest

def test_regressor():
    """Test regrssor function outputs with SpotifyFeatures.csv file."""

    spotify_df = pd.read_csv("tests/data/SpotifyFeatures.csv")
    regressor_df = regressor(spotify_df, target_col="popularity",
                            numeric_feats=['acousticness', 'danceability', 'duration_ms'],
                            categorical_feats=['genre'])
    actual_test_score = regressor_df.loc["test_score"].tolist()
    expected_test_score = [-0.0015649766813072396, 0.6205786616436328, 0.6206776046507274, 0.6206768629762658]
    actual_train_score = regressor_df.loc["train_score"].tolist()
    expected_train_score = [0.0, 0.6251973188529132, 0.6253225222554233, 0.6253244177815244]
    assert actual_test_score == expected_test_score, "regressor modeled incorrectly test scores are not equal to what they should be!"
    assert actual_train_score == expected_train_score, "regressor modeled incorrectly train scores are not equal to what they should be!"


def test_regressor_error() :
    """
    Test error cases and error messages thrown by get_tweets.
    4 tests in total.
    """
    spotify_df = pd.read_csv("tests/data/SpotifyFeatures.csv")

    with pytest.raises(TypeError) as e:
        regressor(1, target_col = "popularity", numeric_feats=['acousticness'])
    assert str(e.value) == "train_df must be a panda dataframe. Please pass a pd.core.frame.DataFrame train_df."

    with pytest.raises(TypeError) as e:
        regressor(spotify_df, target_col = 1, numeric_feats=['acousticness'])
    assert str(e.value) == "target_col must be a str. Please pass target column in str object."

    with pytest.raises(TypeError) as e:
        regressor(1, target_col = "popularity", numeric_feats=1)
    assert str(e.value) == "numerci_feats must be a list. Please pass a list of numeric columns."

    with pytest.raises(TypeError) as e:
        regressor(1, target_col = "popularity", numeric_feats=['acousticness'],categorical_feats=['genre'])
    assert str(e.value) == "categorical_feats must be a list. Please pass a list of categorical columns."
