from simplefit.eda import plot_distributions, plot_corr, plot_splom
import pandas as pd
import pytest
import altair as alt



def df():
    df = pd.DataFrame({
        'A': [100, 142, 30, 40],
        'B': [94, 55, 100, 120],
        'C': [94, 68, 20, 17],
        'D': [3.86, 4.05, 5.6, 1.0],
        'E': [34, 92, 100, 200],
        'name': ['Navya', 'Zoe', 'Sanchit', 'Reza']
    })
    return df

def test_plot_distributions():
    """Test plot_distributions on a dataframe."""

    # Case 1: Test default settings and return
    chart = plot_distributions(df())
    assert isinstance(
        chart, alt.vegalite.v4.api.RepeatChart
    ), "Altair RepeatChart object should be returned."
    assert chart.spec._kwds['mark'] == 'bar', "Altair mark should be 'bar'"

    # Case 2: Test non-default settings and return
    chart = plot_distributions(df(), bins = 40, dist_cols=['A', 'B', 'C'])
    assert isinstance(
        chart, alt.vegalite.v4.api.RepeatChart
    ), "Altair RepeatChart object should be returned."
    assert chart.spec._kwds['mark'] == 'bar', "Altair mark should be 'bar'"

    # Case 3: Test erroneous inputs
    with pytest.raises(Exception):
        plot_distributions([1, 2, 3])

    with pytest.raises(TypeError) as e:
        plot_distributions(df(), dist_cols= "A")


    with pytest.raises(Exception):
        plot_distributions(df(), bins = "ten")

    with pytest.raises(Exception):
        plot_distributions(df(), class_label= [123])



def test_plot_corr():

    """Test plot_corr on a dataframe."""
    # Case 1: Test default settings and return
    chart = plot_corr(df())
    assert isinstance(
        chart, alt.vegalite.v4.api.Chart
    ), "Altair Chart object should be returned."
    assert chart.mark == 'rect', "Altair mark should be 'rect'"

    # Case 2: Test non-default settings and return
    chart = plot_corr(df(), corr='spearman')
    assert isinstance(
        chart, alt.vegalite.v4.api.Chart
    ), "Altair Chart object should be returned."
    assert chart.mark == 'rect', "Altair mark should be 'rect'"

    # Case 3: Test erroneous inputs
    with pytest.raises(Exception) as e:
        plot_corr([1, 2, 3])
    

    with pytest.raises(Exception):
        plot_corr(df(), corr= "A")

    with pytest.raises(Exception):
        plot_corr(df(), corr=[])

   





def test_plot_splom():
    """Test plot_splom on a dataframe."""
    # Case 1: Test default settings and return
    chart = plot_splom(df())
    assert isinstance(
        chart, alt.vegalite.v4.api.RepeatChart
    ), "Altair RepeatChart object should be returned."
    assert chart.spec._kwds['mark']['type']  == 'point', "Altair mark should be 'point'"

    # Case 2: Test non-default settings and return
    chart = plot_splom(df(), pair_cols=["A", "B"])
    assert isinstance(
        chart, alt.vegalite.v4.api.RepeatChart
    ), "Altair RepeatChart object should be returned."
    assert chart.spec._kwds['mark']['type'] == 'point', "Altair mark should be 'point'"

    # Case 3: Test erroneous inputs
    with pytest.raises(Exception):
        plot_splom()

    with pytest.raises(Exception):
        plot_splom([1, 2, 3])

    with pytest.raises(Exception):
        plot_splom(df(), pair_cols= "A")

    with pytest.raises(Exception):
        plot_splom(df(), pair_cols=[])