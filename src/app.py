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
#market_sales = pd.read_csv("https://github.com/Salvatore-Rocha/Supermarket-sales/blob/002314ff6501373a489db96a35c9bd205fdbff8b/supermarket_sales.csv")

header = html.H1(
    "This will be the header to this app", 
    className="bg-primary text-white p-2 mb-2 text-center"
                )

def create_card_4modal(title,subtitle,P_text):
    card = dbc.Card(
        dbc.CardBody([
                html.H3([html.I(className="bi bi-bank me-2"), title]),
                html.H5("$8.3M"),
                html.H6(html.I(subtitle, className="bi bi-caret-up-fill text-success")),
                    ]),
        className="text-center",
        style={"width": "19rem"})   
    return card 

def create_tab(City):
    tab = dbc.Tab([ 
            html.H1(f"{City}"), 
            dbc.Row([
                dbc.Col([ 
                    dbc.Row([
                        create_card_4modal("Customers","Total customers of the period","C"),
                        create_card_4modal("Sales","Total sales of the period","C"),
                        create_card_4modal("Gross Income","Total gross income of the period","C")
                            ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.H5("Select a month to see the sales behaviour"),
                            dcc.RadioItems(['January', 'February','March'], 
                                        'February', 
                                        inline=True,
                                        className="text-success",
                                        inputStyle={"margin-left":"6px", "margin-right": "2px"})
                            ],width=6 ),
                        dbc.Col([
                            html.H5("Select what you wanna see"),
                            dcc.RadioItems(['Customers', 'Sales','Gross Income'], 
                                       'Sales', 
                                       inline=True,
                                       className="text-success",
                                       inputStyle={"margin-left":"6px", "margin-right": "2px"})
                            ],width=6 ),
                        dcc.Graph(id=f"barplot_for_{City}", figure= {} )
                            ])
                        ],width= 6),
                dbc.Col([
                        dcc.Graph(id=f"Pie_1_for_{City}", figure= {}),
                        dcc.Graph(id=f"Pie_2_for_{City}", figure= {}),
                        ],width=4),
                dbc.Col([
                        dcc.Graph(id=f"1Bar_plot_for{City}", figure= {}),
                        ],width=2)
                    ])       
                ], label=f"{City} City",)
    return tab

#The branches are located ing Yangon, Naypyitaw, Mandalay
tab1 = create_tab("Yangon")
tab2 = create_tab("Naypyitaw")
tab3 = create_tab("Mandalay")
tab4 = create_tab("All")
tabs = dbc.Card(dbc.Tabs([tab1, tab2, tab3, tab4], style={'font-style': 'italic'}))

app =  dash.Dash(__name__, 
                 external_stylesheets= [dbc.themes.FLATLY, dbc.icons.FONT_AWESOME, dbc_css],)
#server = app.server
app.layout = dbc.Container([
        header,
        dbc.Row([ #Sli with 4 plots
            tabs
                ]),
    
],fluid=True,
  className="dbc dbc-ag-grid")

if __name__=='__main__':
    app.run_server(debug=True, port=8050)