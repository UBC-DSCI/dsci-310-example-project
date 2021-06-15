import seaborn as sns
import altair as alt


def reg_subplots(data):
    sns.set_theme(font_scale=1.1)
    fig = sns.FacetGrid(
        data.melt(id_vars=["price"]),
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


def cv_results(data):
    scatter = (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("k"),
            y=alt.Y("Mean"),
        )
    )

    errorbars = (
        alt.Chart(data)
        .mark_errorbar()
        .encode(x=alt.X("k"), y=alt.Y("Min", title="RMSE"), y2="Max")
    )

    chart = (scatter + errorbars).configure_axis(labelFontSize=14, titleFontSize=14)
    return chart
