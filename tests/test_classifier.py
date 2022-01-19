from simplefit import simplefit
import pandas as pd
import pytest
from pytest import raises

def test_classifier():
    """Testing classifier function"""

    income_df = pd.read_csv("tests/data/adult.csv")
    df = classifier(income_df, target_col="income",
                            numeric_feats=['age', 'fnlwgt', 'hours.per.week', 'education.num', 'capital.gain', 'capital.loss'],
                            categorical_feats=['occupation', 'workclass', 'relationship', 'marital.status', 'native.country', 'education', 'sex'])
    actual_test_score = df.loc["test_score"].tolist()
    expected_test_score = [0.7605190597073452, 0.8509154674190895]
    actual_train_score = df.loc["train_score"].tolist()
    expected_train_score = [0.7605190419904456, 0.8533945981345206]
    assert actual_test_score == expected_test_score, "Incorrect test scores"
    assert actual_train_score == expected_train_score, "Incorrect train score"

def test_classifier_target() :
    """Testing target column"""
    income_df = pd.read_csv("tests/data/adult.csv")

    with pytest.raises(TypeError) as e:
        classifier(income_df, target_col = 1, numeric_feats = ['age'], categorical_feats = [])
    assert str(e.value) == "target_col should be a str. Please pass target column in str object."
