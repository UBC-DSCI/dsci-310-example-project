import pandas as pd
import seaborn as sns
import altair as alt
from src import analysis, plotting
from pytest import approx


def test_download():
    data = analysis.load_data()
    assert isinstance(data, pd.DataFrame)


def test_wrangle():
    data = analysis.load_data()
    data = analysis.wrangle_data(data)
    assert isinstance(data, pd.DataFrame)
    assert len(data.columns) == 11


def test_split_data():
    data = analysis.load_data()
    data = analysis.wrangle_data(data)
    train, test = analysis.split_data(data, test_fraction=0.2)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(test) / (len(train) + len(test)) == approx(0.2, abs=0.05)


def test_reg_plot():
    data = analysis.load_data()
    data = analysis.wrangle_data(data)
    train, _ = analysis.split_data(data, test_fraction=0.2)
    fig = plotting.reg_subplots(train)
    assert isinstance(fig, sns.axisgrid.FacetGrid)
    assert isinstance(fig.data, pd.DataFrame)


def test_optimization():
    data = analysis.load_data()
    data = analysis.wrangle_data(data)
    train, test = analysis.split_data(data, test_fraction=0.2)
    X_train, y_train, _, _ = analysis.df_to_xy(train, test, normalize=True)
    cv_results_df = analysis.k_optimization(X_train, y_train, k_range=range(1, 11))
    fig = plotting.cv_results(cv_results_df)
    assert isinstance(cv_results_df, pd.DataFrame)
    assert len(cv_results_df) == 10
    assert isinstance(fig, alt.vegalite.v4.api.LayerChart)
    assert isinstance(fig.data,  pd.DataFrame)
    assert len(fig.data.columns) == 4