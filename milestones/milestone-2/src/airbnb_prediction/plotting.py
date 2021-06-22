import seaborn as sns
import altair as alt


def reg_subplots(dataframe):
    """Plot regression plots of features vs target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Airbnb data to plot.

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        Regression subplots.
    """
    sns.set_theme(font_scale=1.1)
    fig = sns.FacetGrid(
        dataframe.melt(id_vars=["price"]),
        col="variable",
        sharex=False,
        sharey=False,
        col_wrap=4,
        height=3,
    )
    fig.map(
        sns.regplot,
        "value",
        "price",
        color=".3",
        scatter_kws={"alpha": 0.2},
        ci=None,
        line_kws={"color": "red"},
    )
    fig.set_titles("{col_name}")
    return fig


def cv_results(dataframe):
    """Plot line chart of cross-validation results.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame of cross-validation results to plot.

    Returns
    -------
    alt.Chart
        Line chart of cross-validation results.
    """
    scatter = (
        alt.Chart(dataframe)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("k"),
            y=alt.Y("Mean"),
        )
    )

    errorbars = (
        alt.Chart(dataframe)
        .mark_errorbar()
        .encode(x=alt.X("k"), y=alt.Y("Min", title="MAE"), y2="Max")
    )

    chart = (scatter + errorbars).configure_axis(labelFontSize=14, titleFontSize=14)
    return chart
