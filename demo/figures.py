import altair as alt

def render_most_similar(data, title):

    bars = (
    alt.Chart(data, height=400, title=title)
       .mark_bar()
       .encode(
           alt.X(
               'Confidence', 
               title='',
               scale=alt.Scale(domain=(0, 1.0), clamp=True),
               axis=None
            ),
            alt.Y(
               'Intent', 
               title='',
               sort=alt.EncodingSortField(
                   field='Confidence',
                   order='descending'
               )
            ),
            color=alt.Color('Confidence', legend=None, scale=alt.Scale(scheme='blues')),
            tooltip=[
                alt.Tooltip(
                    field='Intent',
                    type='nominal'
                ),
                alt.Tooltip(
                    field='Confidence',
                    format='.3f',
                    type='quantitative'
                )
            ]
       )
    )
    text = alt.Chart(data).mark_text(
        align='left',
        baseline='middle',
        dx=5,
        font='Roboto',
        size=12,
        color='black'
    ).encode(
        x=alt.X(
            'Confidence',
            axis=None
        ),
        y=alt.Y(
            'Intent',
            sort=alt.EncodingSortField(
                field='Confidence',
                order='descending'
            )
        ),
        text=alt.Text("Confidence", format=".3f"),
    )
    chart = bars + text
    chart = (chart.configure_axisX(
           labelFontSize=12,
           labelFont='Roboto',
           grid=False,
           domain=False
       )
       .configure_axisY(
           labelFontSize=12,
           labelFont='Roboto',
           grid=False,
           domain=False
       )
       .configure_view(
            strokeOpacity=0
       )
       .configure_title(
           fontSize=16,
           font='Roboto',
           dy=-10
       )
    )

    return chart