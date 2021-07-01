import seaborn as sns
import altair as alt


def reg_subplots(
    dataframe,
    save=False,
    save_location="results/regression_plots.png",
):
    """Plot regression plots of features vs target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Airbnb data to plot.
    save : bool, optional
        Whether to save figure to file, by default False
    save_location : str, optional
        The save location of file.

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
    if save:
        fig.savefig(save_location)
    return fig


def cv_results(dataframe, save=False, save_location="results/k_optimization_plot.png"):
    """Plot line chart of cross-validation results.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame of cross-validation results to plot.
    save : bool, optional
        Whether to save figure to file, by default False
    save_location : str, optional
        The save location of file.


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
    if save:
        chart.save(save_location)
    return chart
