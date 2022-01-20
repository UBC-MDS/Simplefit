from simplefit.cleaner import cleaner
import pandas as pd
import numpy as np
import pytest
from pytest import raises

def df(lower_case=True):
    dataf = pd.DataFrame({'num_legs': [2, 4, 8, 0],
                   'num_  wings ': ['bad', 'Good', None, 'bad'],
                   'Num specimen seen ': [10, 2, 1, 8]},
                  index=['falcon', 'dog', 'spider', 'fish'])
    
    return dataf

def test_cleaner():
    """
    Test function to check the clean data.
    """
    # Tests whether a not dataframe input raises TypeError
    with raises(TypeError):
        cleaner(np.array([1, 2, 3, 4, 5]))

    # Tests whether a not dataframe input raises TypeError
    with raises(TypeError):
         cleaner(df(),lower_case=1)

def test_correct_output():
    # Tests that cleaner returns a dataframe 
    output = cleaner(df())

    assert type(output) == pd.DataFrame, "Output is not a pandas DataFrame"


def test_correct_columns():
    #TEST 2: Test that the dataframe has the appropriate column names
    output= cleaner(df(),lower_case=True)
    expected_columns = ['num_legs', 'num_  wings', 'num specimen seen']
    assert list(output.columns) == list(expected_columns), "not equal"
