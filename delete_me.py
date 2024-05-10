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
market_sales = pd.read_csv("https://github.com/Salvatore-Rocha/Supermarket-sales/blob/002314ff6501373a489db96a35c9bd205fdbff8b/supermarket_sales.csv")

print(market_sales.head())
