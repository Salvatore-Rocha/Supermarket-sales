import dash
import plotly.graph_objects as go
from dash import dcc, html, callback
from dash.dependencies import Output, Input
import plotly.express as px 
import dash_bootstrap_components as dbc
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import dash_player as dp
from dash_bootstrap_templates import load_figure_template
import plotly.io as pio

# Create a Plotly layout with the desired dbc template
load_figure_template(["flatly", "flatly_dark"])
layout = go.Layout(template= pio.templates["flatly"])

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

# File
market_sales = pd.read_csv(csv_file)

header = html.H4(
    "This will be the header to this app", 
    className="bg-primary text-white p-2 mb-2 text-center"
                )

tab1 = dbc.Tab([dcc.Graph(id="line-chart", figure=px.line(template="bootstrap"))], label="Line Chart")
tab2 = dbc.Tab([dcc.Graph(id="scatter-chart", figure=px.scatter(template="bootstrap"))], label="Scatter Chart")
tab3 = dbc.Tab([grid], label="Grid", className="p-4")
tabs = dbc.Card(dbc.Tabs([tab1, tab2, tab3]))


app =  dash.Dash(__name__, 
                 external_stylesheets= [dbc.themes.FLATLY, dbc.icons.FONT_AWESOME, dbc_css],)
server = app.server
app.layout = dbc.Container(
    header
)

if __name__=='__main__':
    app.run_server(debug=True, port=8050)