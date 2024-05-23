import dash
import plotly.graph_objects as go
from dash import dcc, html, callback
from dash.dependencies import Output, Input, State, MATCH, ALL
import plotly.express as px 
import dash_bootstrap_components as dbc
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
from dash_bootstrap_templates import load_figure_template
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
from dash.exceptions import PreventUpdate
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
import numpy as np

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

# CSV File in Github (added ?raw=true at the end of the URL or it will not parse it correctly) 
market_sales = pd.read_csv("https://github.com/Salvatore-Rocha/Supermarket-sales/blob/002314ff6501373a489db96a35c9bd205fdbff8b/supermarket_sales.csv?raw=true")

# Fixing date format
market_sales['Date'] = pd.to_datetime(market_sales['Date'])

variables = {
    'City': ['City_Mandalay', 'City_Naypyitaw', 'City_Yangon'], 
    'Customer type': ['Customer type_Member', 'Customer type_Normal'], 
    'Gender': ['Gender_Female', 'Gender_Male'], 
    'Product line': ['Product line_Electronic accessories',
                     'Product line_Fashion accessories', 'Product line_Food and beverages',
                     'Product line_Health and beauty', 'Product line_Home and lifestyle',
                     'Product line_Sports and travel'], 
    'Date': ['Date_Friday','Date_Monday','Date_Saturday','Date_Sunday','Date_Thursday', 'Date_Tuesday', 'Date_Wednesday'], 
    'Time': ['Time_Evening','Time_Morning','Time_Night'], 
    'Payment': ['Payment_Cash', 'Payment_Credit card', 'Payment_Ewallet']
}

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        options=[{'label': k, 'value': k} for k in variables.keys()],
        value=list(variables.keys())[:3], 
        id='option-example-dropdown',
        multi=True,
        clearable=False
    ),
    html.Div(children={}, id='dropdown-container'),
    html.Button('Show Encoded Vector', id='button', disabled=True),
    html.Div(id='output-vector')
])

@callback(
    Output('dropdown-container', 'children'),
    Input('option-example-dropdown', 'value')
)
def update_dropdowns(selected_keys):
    if not selected_keys:
        raise PreventUpdate

    dropdowns = []
    for key in selected_keys:
        dropdowns.append(html.Label(key))
        dropdowns.append(dcc.Dropdown(
            options=[{'label': i, 'value': i} for i in variables[key]],
            id={'type': 'dynamic-dropdown', 'index': key},
            multi=False
        ))

    return dropdowns

@callback(
    Output('button', 'disabled'),
    [Input('option-example-dropdown', 'value'),
     Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value')]
)
def update_button_state(selected_keys, selected_values):
    if not selected_keys or None in selected_values:
        return True
    return False

@callback(
    [Output('output-vector', 'children'),
     Output({'type': 'dynamic-dropdown', 'index': ALL}, 'value')],
    [Input('button', 'n_clicks')],
    [State('option-example-dropdown', 'value'),
     State({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
     State({'type': 'dynamic-dropdown', 'index': ALL}, 'id')]
)
def update_output(n_clicks, selected_keys, selected_values, selected_ids):
    if not n_clicks:
        raise PreventUpdate

    if None in selected_values:
        raise PreventUpdate

    selected_dict = {item['index']: values for item, values in zip(selected_ids, selected_values)}

    encoded_vector = []
    for key in variables.keys():  # iterate through all keys in the fixed order
        if key in selected_keys:
            for option in variables[key]:
                if selected_dict.get(key) and option in selected_dict[key]:
                    encoded_vector.append(1)
                else:
                    encoded_vector.append(0)

    encoded_vector = np.array(encoded_vector).reshape(1, -1)

    # Clear options of all dropdowns
    cleared_values = [None] * len(selected_values)

    return [f'One-hot encoded vector: {encoded_vector[0]}'], cleared_values

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
