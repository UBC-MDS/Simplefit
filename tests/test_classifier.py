from simplefit.simplefit import classifier
import pandas as pd
import pytest
from pytest import raises
from sklearn.model_selection import (train_test_split,)

def test_classifier():
    """Testing classifier function"""

    income_df = pd.read_csv("tests/data/adult.csv")
    train_df, test_df = train_test_split(income_df, test_size=0.97, random_state=123)
    df = classifier(train_df, target_col="income", numeric_feats=['age', 'fnlwgt', 'hours.per.week', 'education.num', 'capital.gain', 'capital.loss'], categorical_feats=['occupation'])
    actual_test_score = df.loc["test_score"].tolist()
    expected_test_score = [0.7633228676085818, 0.8125065410779696]
    actual_train_score = df.loc["train_score"].tolist()
    expected_train_score = [0.7633198726156474, 0.8299185790735086]
    assert actual_test_score == expected_test_score, "Incorrect test scores"
    assert actual_train_score == expected_train_score, "Incorrect train score"

def test_classifier_target() :
    """Testing target column"""
    income_df = pd.read_csv("tests/data/adult.csv")

    with pytest.raises(TypeError) as e:
        classifier(income_df, target_col = "income", numeric_feats = 1, categorical_feats = ['occupation'])
    assert str(e.value) == "Numeric Features should be passed as a list"
