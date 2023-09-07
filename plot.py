import plotly.graph_objects as go


map_50_95 = []
mean_inference_time = []

fig = go.Figure(data=go.Scatter(x=x, y=y))

fig.update_layout(
    font_family="Courier New",
    yaxis=dict(title='# of Occurances', titlefont=dict(size=20)),
    xaxis=dict(title='Product Classes', titlefont=dict(size=20))
    )