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
import joblib
import base64
import io

# Create a Plotly layout with the desired dbc template
load_figure_template(["flatly", "flatly_dark"])
layout = go.Layout(template= pio.templates["flatly"])

dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

# CSV File in Githubt (added ?raw=true at the end of the URL or it will not parse it correctly) 
market_sales = pd.read_csv("https://github.com/Salvatore-Rocha/Supermarket-sales/blob/002314ff6501373a489db96a35c9bd205fdbff8b/supermarket_sales.csv?raw=true")

#Fixing date format
market_sales['Date'] = pd.to_datetime(market_sales['Date'])

header = html.H1(
    "Urban Sales Metrics of Myanmar HUB", 
    className="bg-primary text-white p-2 mb-2 text-center"
                )

sources = html.Div(
    [
        html.P("By Eduardo Salvador Rocha"),
        html.Label(
            [
                "Links: ",
                html.A(
                    "Eduardo's GitHub|  ",
                    href="https://github.com/Salvatore-Rocha/Supermarket-sales",
                    target="_blank",
                ),
                html.A(
                    "Code (.py file) |   ",
                    href="https://github.com/Salvatore-Rocha/Supermarket-sales/blob/0344a62c2e0c00b254c93690b5cc873c8cfb77a7/src/app.py",
                    target="_blank",
                
                ),
                html.A(
                    "Code (Jupyter Notebook (Google Colab) |   ",
                    href="https://colab.research.google.com/drive/1UX7Bah8Sn1WaajXQQzInp-4RprBnz2qD?usp=sharing",
                    target="_blank",
                
                ),
            ]
        ),
    ]
)

def create_card(title,City):
    card = dbc.Card(
        dbc.CardBody([
                html.H5([
                        html.I(className="bi bi-bank me-2"), 
                            title
                        ]),
                html.H6(
                        children= {}, 
                        id= f"val_card_{title[0:4]}_{City}"
                        ),
                html.P(
                        html.I(children = {},
                                id= f"subt_card_{title[0:4]}_{City}", 
                                className="bi bi-caret-up-fill text-success")
                        ),
                    ]),
        className="text-center",
        style={"width": "14rem"})   
    return card 

def create_tab(City):
    tab = dbc.Tab([ 
            html.H3(
                    children=f"{City}",
                    id=f"tab_{City}"
                    ), 
            dbc.Row([
                dbc.Col([ #Main Body
                    dbc.Row([ #Upper Cards
                            create_card("Sales",City),
                            create_card("Gross Income",City),
                            create_card("COGS",City)
                            ]),
                    html.Br(),
                    dbc.Row([ #Barplot-date + Radio Items
                            dbc.Col([
                                html.P("Choose a month to view a metric's performance"),
                                dcc.RadioItems(id=f"radio_month_{City}",
                                            options= ['January', 'February','March', "All"], 
                                            value='February', 
                                            inline=True,
                                            className="text-success",
                                            inputStyle={"margin-left":"6px", "margin-right": "2px"},
                                            style={'fontSize': '12px'})
                                ],width=6 ),
                            dbc.Col([
                                html.P("Select a metric"),
                                dcc.RadioItems(id =f"radio_types_{City}",
                                        options= ['Customers', 
                                                    "Sales, Gross Income & COGS",
                                                    "Product Line"],
                                        value= 'Sales, Gross Income & COGS', 
                                        inline=False,
                                        className="text-success",
                                        inputStyle={"margin-left":"6px", "margin-right": "2px"},
                                        style={'fontSize': '12px'})
                                ],width=6 ),
                            dcc.Graph(id=f"barplot_for_{City}", 
                                    figure= {} )
                            ])
                        ],width= 6),
                dbc.Col([ #Product Line plt
                        html.Div(dcc.Graph(
                                        id=f"product_line_totals_{City}", 
                                        figure= {}
                                        ), 
                                 style={'marginLeft': '-50px','marginRight': '-80px','marginTop': '-35px'}
                                 ),
                        ],width=3),
                dbc.Col([ #Last column plts
                        html.Div(create_card("Customers",City), 
                                 style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
                                 ),
                        dcc.Graph(id=f"Custotype_bar_{City}",
                                  figure= {}
                                  ),
                        html.Div(
                            dcc.Graph(
                                id=f"Gender_bar_{City}", 
                                figure= {}
                                    ), 
                                style={'marginTop': '-100px'}
                                ),
                        html.Div(
                                dcc.Graph(id=f"Payment_pie_{City}", 
                                      figure= {}), 
                                style={'marginTop': '-100px'}
                                ),
                        ],width=3, className="my-0")
                    ])       
                ], label=f"{City}")
    return tab

def calc_cards_vals(month,city):
    #Sales {Total}, Gross Income {gross income}, COGS{cogs}, Customers
    if city == "All":
        _dff = market_sales
    else:
        _dff = market_sales[market_sales["City"] == city]
    
    if month == "All":
        dff = _dff
        month = "during Q1"
    else:
        dff = _dff[_dff["Date"].dt.month_name() == month]
        month = f"in {month}"
        
    Sales        = dff["Total"].sum().round()
    text_sales   = f"Total Sales {month}" 
    Cogs    = dff["cogs"].sum().round()
    text_cogs   = f"Total costs {month}"
    Gross_Income = dff["gross income"].sum().round()
    text_gross   = f"Total Sales {month}"
    Customers    = dff["Customer type"].count()
    text_custo   = f"Total Customers {month}"
    
    return "${:,.0f}".format(Sales), text_sales, \
           "${:,.0f}".format(Gross_Income), text_gross, \
           "${:,.0f}".format(Cogs), text_cogs, \
           Customers, text_custo

cmap_product_lines = { #If we dont use a color map, the plots will assing arbitrary colors to each cat every time they update
'Health and beauty': "#18BC9C" ,
'Electronic accessories': "#3498DB" ,
'Home and lifestyle': "#2ECC71",
'Sports and travel': "#5BC0DE", 
'Food and beverages': "#F39C12", 
'Fashion accessories': "#34495E"
     }
 
def complex_fig(dffs):    
    fig = go.Figure(
                    data=[
                        go.Bar(
                            name="Sales",
                            x=dffs["Date"],
                            y=dffs["Sales"],
                            offsetgroup=0,
                            hovertemplate='Date %{x}<br>Value: $%{y:.0f}'
                        ),
                        go.Bar(
                            name="COGS",
                            x=dffs["Date"],
                            y=dffs["COGS"],
                            offsetgroup=1,
                            hovertemplate='Date %{x}<br>Value: $%{y:.0f}'
                        ),
                        go.Bar(
                            name="Income",
                            x=dffs["Date"],
                            y=dffs["Gross_Income"],
                            offsetgroup=1,
                            base=dffs["COGS"], 
                            hovertemplate='Date %{x}'
                        )
                        ],
                        layout=go.Layout(
                            yaxis_title="Value"
                        )
                        )
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                              yaxis_title='',
                              yaxis_tickprefix='$',  
                              yaxis_tickformat=',.0f'  
                              )
    return fig 
 
def make_bar_plot_city(month,type,city):
    if city == "All":
        _dff = market_sales
    else:
        _dff = market_sales[market_sales["City"] == city]
    if month == "All":
        dff = _dff
    else:
        dff = _dff.loc[market_sales["Date"].dt.month_name() == month]
        
    if type == "Customers":
        dffs = dff.groupby('Date').agg(Customers=('Customer type', 'count')).reset_index()
        fig = px.bar(dffs,
             x="Date",
             y="Customers",
             barmode='group',
             color_discrete_sequence=["#34495E"],
             )
        
        return fig
         
    if type == "Sales, Gross Income & COGS":
        dffs = dff.groupby('Date').agg(Sales=('Total', 'sum'),
                                       Gross_Income = ('gross income', 'sum'),
                                       COGS = ('cogs', 'sum') ).round().reset_index()
        return complex_fig(dffs)
        
    if type == "Product Line":
        dffs = dff.groupby(['Date',"Product line"]).agg(Sales=('Total', 'sum')).reset_index()
        fig = px.bar(dffs,
                    x="Date",
                    y="Sales",
                    color= "Product line",
                    color_discrete_map=cmap_product_lines, 
                    barmode='stack',
                    template = "flatly"
                    )
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                              yaxis_tickprefix='$',  # Add dollar sign as prefix
                              yaxis_tickformat=',.0f'  # Format as floating point with 2 decimal places and comma separators
                              )
        return fig
 
def make_product_line_tree(month,city):
    if city == "All":
        _dff = market_sales
    else:
        _dff = market_sales[market_sales["City"] == city]
    
    if month == "All":
        dff = _dff
        month = "Q1"
    else:
        dff = _dff[_dff["Date"].dt.month_name() == month]
    
    dff.loc[:, "Total"] = dff["Total"].round()
    fig = px.treemap(dff, 
                 path=["Product line"], 
                 values='Total',
                 height=800,
                 color = "Product line",
                 color_discrete_map= cmap_product_lines
                 )
    fig.update_traces(textinfo='label+value',
                      texttemplate='%{label} <br>$%{value:,.0f}',
                      hovertemplate=' Line: %{label} <br> Total Sales: $%{value:,.0f}')
    fig.update_layout(title=dict(text=f"<i>Line Products Total Sales</i> - {month}",font=dict(size=15, color ="#34495E"),x=0.5, y=0.95),
                      )
    
    return fig

def make_count_type_bar(month,city,column_name):
    if city == "All":
        _dff = market_sales
    else:
        _dff = market_sales[market_sales["City"] == city]
    if month == "All":
        dff = _dff
        month = "Q1"
    else:
        dff = _dff[_dff["Date"].dt.month_name() == month]
        
    dffs = dff.groupby([column_name]).agg(Count=(column_name, 'count')).reset_index()
    dffs["Type"] = "Type"
    total_count = dffs['Count'].sum()
    dffs['Percentage'] = (dffs['Count'] / total_count) * 100
    dffs['Percentage'] = dffs['Percentage'].round(2).astype(str) + '%'
    dffs["Text"] = dffs['Count'].astype(str) + '<br>(' + dffs['Percentage'] + ')'
    fig = px.bar(dffs,
             y="Type",
             x="Count",
             color=column_name,
             orientation="h",
             text="Text",
             barmode='stack',
             )
    fig.update_layout(legend=dict(title_font_size=12,orientation='h', yanchor='bottom', xanchor='right',y=1.02, x=1),
                      height=250)
    fig.update_traces(insidetextanchor='middle',
                      hovertemplate='Count %{value:,.0f}')
    fig.update_xaxes(title='',showticklabels=False)
    fig.update_yaxes(title='',showticklabels=False)

    return fig

def make_circles_gender_payment_city(month,city):   
    if city == "All":
        _dff = market_sales
    else:
        _dff = market_sales[market_sales["City"] == city]
    
    if month == "All":
        dff = _dff
        month = "Q1"
    else:
        dff = _dff[_dff["Date"].dt.month_name() == month]
    
    dffs = dff.groupby(['Payment']).agg(Type=('Payment', 'count')).reset_index()
    fig = px.pie(dffs,
                values= "Type",
                names="Payment",
                hole=.3)
    fig.update_traces(textposition='inside',
                      textinfo='percent+value',
                      texttemplate= 'Count %{value:,.0f} <br> %{percent}',
                      hovertemplate=' Percent %{percent} <br> Count %{value:,.0f}')
    fig.update_layout(title=dict(text=f"Type of Payment - {month}",
                                 font=dict(size=13),
                                 x=0.5, y=0.9),
                      legend=dict(orientation='h', 
                                  yanchor='bottom', 
                                  xanchor='right',
                                  x=1, y=-0.1))
    return fig

def categorize_hour(hour):
    if 10 <= hour < 13:
        return 'Morning'
    elif 13 <= hour < 17:
        return 'Evening'
    else:
        return 'Night'

def data_encoding(_columns, target_value):
    #Columns: 'Day_of_week', 'Hour', 'Gender', 'Product line', 'City', 'Customer type', 'Payment'
    dff = market_sales.copy()

    if 'Date' in _columns:
        # Extract day of the week from 'Date' column
        dff['Date'] = dff['Date'].dt.day_name()
    
    if "Time" in _columns:
        # Convert 'Hour' column to datetime format and extract hour and apply categorization function 
        dff['Time'] = pd.to_datetime(dff['Time'], format='%H:%M').dt.hour
        dff['Time'] = dff['Time'].apply(categorize_hour)
    
    dff = pd.get_dummies(dff, columns= _columns)
    
    dff = dff.iloc[:,17-len(_columns):]
    dff[target_value] = market_sales[target_value]
    
    return dff

def get_text_color(color):
    # Calculate the luminance of the color
    r, g, b = tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    return 'white' if luminance < 0.5 else 'black'

def serialize_model(model):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return model_str

def deserialize_model(model_str):
    buffer = io.BytesIO(base64.b64decode(model_str.encode('utf-8')))
    model = joblib.load(buffer)
    return model

def make_ftimp_analysis(df,target):
    X = df.iloc[:,:-1]  # Features
    y = df[target] # Target variable

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X, y)

    # Get feature importances and plotting them
    feature_importances = rf_regressor.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    fig = go.Figure()
    #Inverted colors; clearer for lower, darker for higher
    sequential_colors = px.colors.sequential.Plasma[::-1]
    
    #textfont: Dark text for bright and white text for dark background colors
    fig.add_trace(go.Bar(
        y=feature_importance_df['Feature'],
        x=feature_importance_df['Importance'],
        orientation='h',
        hovertemplate='Feature %{y}<br> Value: %{x:.4f}<extra></extra>',
        text=feature_importance_df['Importance'].round(4),  
        textposition='inside',  
        insidetextanchor='end',
        marker=dict(color=feature_importance_df['Importance'], colorscale=sequential_colors), 
        textfont=dict(color=[get_text_color(color) for color in sequential_colors])
    ))

    fig.update_layout(
        title='Feature Importance Analysis',
        xaxis_title='Feature Importance',
        yaxis_title='',
        height=700,
        hoverlabel=dict(font=dict(color='white'))
    )

    return serialize_model(rf_regressor), fig

def make_train_curve(df,target):
    #Train and test are defined to have a ratio 80-20; RandomForestRegressor is the model used
    #Validation curve -> Test hyperparameters (n_estimaors) on score, Learning_curve -> Test num of samples on score
    X = df.iloc[:,:-1]  
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    model = RandomForestRegressor()

    # Learning curve data, neg_mean_squared_error is used for convention (maximizing)
    train_sizes, train_scores, test_scores = learning_curve(model, 
                                                            X_train, y_train, 
                                                            cv=3, train_sizes=np.linspace(0.1, 1.0, 10), 
                                                            scoring='neg_mean_squared_error')

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    print("Learning Done")
    # Plot the learning curve
    Fig1 = go.Figure()

    Fig1.add_trace(go.Scatter(
        x=train_sizes, y=train_scores_mean,
        mode='lines+markers',
        name='Training score',
        line=dict(color='#18BC9C'),
        hovertemplate='Training Error %{y:.2f}<br> Num.Samples: %{x:.0f}<extra></extra>'
    ))

    Fig1.add_trace(go.Scatter(
        x=train_sizes, y=test_scores_mean,
        mode='lines+markers',
        name='Cross-validation score',
        line=dict(color='#3498DB'),
        hovertemplate='CrossVal Error %{y:.2f}<br> Samples: %{x:.0f}<extra></extra>'
    ))

    # Fill between the upper and lower bounds of the scores
    Fig1.add_trace(go.Scatter(
        x=train_sizes, y=train_scores_mean + train_scores_std,
        mode='lines',
        line=dict(width=0),
        name="",
        hovertemplate='<extra></extra>',
        showlegend=False
    ))

    Fig1.add_trace(go.Scatter(
        x=train_sizes, y=train_scores_mean - train_scores_std,
        mode='lines',
        line=dict(width=0),
        name="",
        hovertemplate='<extra></extra>',
        fill='tonexty',
        showlegend=False,
        fillcolor='rgba(24, 188, 156, 0.2)'
    ))

    Fig1.add_trace(go.Scatter(
        x=train_sizes, y=test_scores_mean + test_scores_std,
        mode='lines',
        name="",
        hovertemplate='<extra></extra>',
        line=dict(width=0),
        showlegend=False
    ))

    Fig1.add_trace(go.Scatter(
        x=train_sizes, y=test_scores_mean - test_scores_std,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        name="",
        hovertemplate='<extra></extra>',
        showlegend=False,
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))

    Fig1.update_layout(
        title={'text': '<i>Learning Curve</i>', 'font': {'size': 13, "color":"#34495E"}},
        yaxis_title='Mean Squared Error',
        xaxis_title='Sample Size',
        xaxis_title_font=dict(size=10),
        legend=dict(orientation='h', yanchor='bottom', xanchor='right',y=1.02, x=1),
        showlegend=True,
        height = 350
                    )

    print("Exiting...")
    return Fig1

def make_validation_curve(df,target):
    #Train and test are defined to have a ratio 80-20; RandomForestRegressor is the model used
    #Validation curve -> Test hyperparameters (n_estimaors) on score, Learning_curve -> Test num of samples on score
    X = df.iloc[:,:-1]  
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    model = RandomForestRegressor()

    # Validation curve data
    param_range = np.arange(1, 201, 20)
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name="n_estimators", param_range=param_range,
        cv=5, scoring="neg_mean_squared_error")
    print("Validation done")
    # Calculate the mean and standard deviation for train and test scores
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot the validation curve
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=param_range, y=train_scores_mean,
        mode='lines+markers',
        name='Training score',
        line=dict(color='#669BBC'),
        hovertemplate='Training Error %{y:.2f}<br> Num.Estimators: %{x:.0f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=param_range, y=test_scores_mean,
        mode='lines+markers',
        name='Cross-validation score',
        line=dict(color='#F39C12'),
        hovertemplate='CrossVal Error %{y:.2f}<br> Samples: %{x:.0f}<extra></extra>'
    ))

    # Fill between the upper and lower bounds of the scores
    fig.add_trace(go.Scatter(
        x=param_range, y=train_scores_mean + train_scores_std,
        mode='lines',
        name="",
        hovertemplate='<extra></extra>',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=param_range, y=train_scores_mean - train_scores_std,
        mode='lines',
        line=dict(width=0),
        name="",
        hovertemplate='<extra></extra>',
        fill='tonexty',
        showlegend=False,
        fillcolor='rgba(102, 155, 188, 0.2)'
    ))

    fig.add_trace(go.Scatter(
        x=param_range, y=test_scores_mean + test_scores_std,
        mode='lines',
        name="",
        hovertemplate='<extra></extra>',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=param_range, y=test_scores_mean - test_scores_std,
        mode='lines',
        line=dict(width=0),
        name="",
        hovertemplate='<extra></extra>',
        fill='tonexty',
        showlegend=False,
        fillcolor='rgba(243, 156, 18, 0.2)'
    ))

    fig.update_layout(
        title={'text': '<i>Validation Curve</i>', 'font': {'size': 13, "color":"#34495E"}},
        xaxis_title='<i>Number of Estimators</i>',
        xaxis_title_font=dict(size=10),
        yaxis_title='Mean Squared Error',
        legend=dict(orientation='h', yanchor='bottom', xanchor='right',y=1.02, x=1),
        showlegend=True,
        height = 350
                    )
    
    print("Exiting...")
    return fig

variables = { #{Key:Values} to create dynamic dropdowns
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
     
#The branches are located in Yangon, Naypyitaw, Mandalay
tab2 = create_tab("Yangon")
tab3 = create_tab("Naypyitaw")
tab4 = create_tab("Mandalay")
tab5 = create_tab("All")
tab1 = dbc.Tab([ 
                html.Br(),
                dbc.Row([
                    dbc.Col([ # Main text & Prediction params
                        html.H3("On-the-Fly: Feature Impact and Model Analysis", 
                                className="text-success",
                                style={"font-weight": "bold"}),
                        html.P(["Welcome to this Supermarket Sales Dashboard! This project is built using the Python library of Dash \
                            and uses the dataset of the historical sales from a supermarket company. This dash app analyzes the records \
                            from three branches over three months and provides predictions and insights to help users understand \
                            store performance better. Please refer to the external links at the bottom for more details."],
                            style={"font-size": "12px", "text-align": "justify", "font-style": "italic"}),
                        html.Br(),    
                        html.P(["This particular tab utilizes a Random Forest Regressor technique to analize feature importance, visualize training and validation curves, \
                            and input predictor values for making predictions. The feature importance section ranks the variable's impact on the model's output \
                            (i.e. when making predictions); the curves provide insights into model performance during training and validation. \
                            The dropdown menus below adjust automatically based on variable selection.", 
                            dcc.Markdown("**Note that predictions will only be available once the training and validation plots are visible, and ALL the dropdowns bellow\
                                have a value selected**")
                            ],style={"font-size": "12px", "text-align": "justify", "font-style": "italic"}
                            ),
                        html.Div(children={}, id='dropdown-container'),
                        dbc.Button(children = {}, id="button", disabled=True, n_clicks=0),
                        dcc.Loading(id="loading-prediction",
                                            type="default",
                                            children= html.Div(id='output-vector'),
                                            ),
                        html.Hr(),
                        html.Div("Predictors:"),
                        dcc.Loading(id="loading-predictors",
                                            type="default",
                                            children= html.Div(id="output-variables"),
                                            ),
                        dcc.Store(id='store-trained-model'),
                            ]
                        ,width= 3),
                    dbc.Col([
                        dbc.Row(style={"height": "10px"}),
                        html.Hr(),
                        dbc.Row([ #Dpdn, Radio & button
                            dbc.Col([
                                html.P("Select the variables to study", 
                                        className="text-success",
                                        style={'font-style': 'italic'}),
                                html.P("⚠️ Every time a value in the dropdown list or radio items changes, the model needs to be updated before it can be used to make a prediction", 
                                         className="text-success",
                                         style={'text-align': 'left', 'font-size': '10px'}),
                                dcc.Dropdown(id= "ftimp_dpdn_vars", 
                                             options = ['Date', 'Time', 'Gender', 'Product line', \
                                                        'City', 'Customer type', 'Payment'],
                                             value= ['Date', 'Time', 'Gender', 'Product line'],
                                             multi = True,
                                             className="text-success"
                                             )
                                    ]),
                            dbc.Col([
                                html.P("Select the target to analyze", 
                                        className="text-success",
                                        style={'font-style': 'italic'}),
                                dcc.RadioItems(id= "ftimp_radio_target", 
                                             options = ['Total', 'Rating'],
                                             value= "Rating",
                                             className="text-success",
                                             style={'fontSize': '13px'}
                                             )
                                    ]),
                                ]),
                        dbc.Row([ # ML Plots
                            dbc.Col([ #Feature Importance
                                dcc.Loading(id="loading-ftipm",
                                            type="default",
                                            children= dcc.Graph(id = "feature_importance_plot",
                                                                figure = {}                            
                                                                ),
                                           )
                                    ], width= 7,className="my-0"),
                            dbc.Col([#Val & Training Curves
                                  html.Div([
                                        dbc.Button("Update Model ", id="update-button", color="primary")],
                                        className="d-grid gap-2",
                                          ),
                                  html.P("↓ Learning & Validation Curves ↓", 
                                         className="text-success",
                                         style={'text-align': 'center'}),
                                  dcc.Loading(id="loading-1",
                                            type="default",
                                            children= dcc.Graph(id = "learning_curve",
                                                          figure = {}                            
                                                         ),
                                            ),
                                    html.Div(
                                        dcc.Loading(id="loading-2",
                                            type="default",
                                            children= dcc.Graph(id = "validation_curve",
                                                          figure = {}                            
                                                         ),
                                                    ),
                                            style={'marginTop': '-20px'}
                                            ),            
                                    ], width= 5,className="my-0")
                                ]),

                            ])
                ])
                ], label="Rating and Sales Prediction",)
tabs = dbc.Card(dbc.Tabs([tab1, tab2, tab3, tab4, tab5], style={'font-style': 'italic'}))


app =  dash.Dash(__name__, 
                 external_stylesheets= [dbc.themes.FLATLY, dbc.icons.FONT_AWESOME, dbc_css],)
#server = app.server
app.layout = dbc.Container(style={'padding': '50px'},
    children=[
            header,
            dbc.Row([ #Carrousel with 5 windows
                    tabs
                    ]),
            dbc.Row([ #Links/ Sources
                    sources
                    ]),  
],fluid=True,
  className="dbc dbc-ag-grid")

#################################################################################################
#Barplot Yangon
_City = "Yangon"
@callback(
    Output(f"barplot_for_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"radio_types_{_City}","value"),
    Input(f"tab_{_City}","children")
)  
def make_plots(month,type,city):
    return make_bar_plot_city(month,type,city)

#Updating Cards 
#Sales {totaT}, Gross Income {gross income}, COGS{cogs}, Customers
@callback(
    Output(f"val_card_Sale_{_City}", "children"),
    Output(f"subt_card_Sale_{_City}", "children"),
    Output(f"val_card_Gros_{_City}", "children"),
    Output(f"subt_card_Gros_{_City}", "children"),
    Output(f"val_card_COGS_{_City}", "children"),
    Output(f"subt_card_COGS_{_City}", "children"),
    Output(f"val_card_Cust_{_City}", "children"),
    Output(f"subt_card_Cust_{_City}", "children"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
)  
  
def update_cards(month,city):    
    #Order : Sales, text_sales, Customers, text_custo, Gross_Income, text_gross
    return calc_cards_vals(month,city)

#Line product
@callback(
    Output(f"product_line_totals_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
    )

def make_treemap(month,city):
    return make_product_line_tree(month,city)

#Last Column Charts
@callback(
    Output(f"Custotype_bar_{_City}", "figure"),
    Output(f"Gender_bar_{_City}", "figure"),
    Output(f"Payment_pie_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
    )

def create_customers_plots(month, city):
    return make_count_type_bar(month,city,"Customer type"), make_count_type_bar(month,city,"Gender"), make_circles_gender_payment_city(month,city)


#################################################################################################
#Barplot Mandalay
_City = "Mandalay"
@callback(
    Output(f"barplot_for_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"radio_types_{_City}","value"),
    Input(f"tab_{_City}","children")
)
def make_plots(month,type,city):
    return make_bar_plot_city(month,type,city)

#Updating Cards 
#Sales {totaT}, Gross Income {gross income}, COGS{cogs}, Customers
@callback(
    Output(f"val_card_Sale_{_City}", "children"),
    Output(f"subt_card_Sale_{_City}", "children"),
    Output(f"val_card_Gros_{_City}", "children"),
    Output(f"subt_card_Gros_{_City}", "children"),
    Output(f"val_card_COGS_{_City}", "children"),
    Output(f"subt_card_COGS_{_City}", "children"),
    Output(f"val_card_Cust_{_City}", "children"),
    Output(f"subt_card_Cust_{_City}", "children"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
)  
  
def update_cards(month,city):    
    #Order : Sales, text_sales, Customers, text_custo, Gross_Income, text_gross
    return calc_cards_vals(month,city)

#Line product
@callback(
    Output(f"product_line_totals_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
    )

def make_treemap(month,city):
    return make_product_line_tree(month,city)

#Last Column Charts
@callback(
    Output(f"Custotype_bar_{_City}", "figure"),
    Output(f"Gender_bar_{_City}", "figure"),
    Output(f"Payment_pie_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
    )

def create_customers_plots(month, city):
    return make_count_type_bar(month,city,"Customer type"), make_count_type_bar(month,city,"Gender"), make_circles_gender_payment_city(month,city)

#################################################################################################
#Barplot Naypyitaw
_City = "Naypyitaw"
@callback(
    Output(f"barplot_for_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"radio_types_{_City}","value"),
    Input(f"tab_{_City}","children")
)  

def make_plots(month,type,city):
    return make_bar_plot_city(month,type,city)

#Updating Cards 
#Sales {totaT}, Gross Income {gross income}, COGS{cogs}, Customers
@callback(
    Output(f"val_card_Sale_{_City}", "children"),
    Output(f"subt_card_Sale_{_City}", "children"),
    Output(f"val_card_Gros_{_City}", "children"),
    Output(f"subt_card_Gros_{_City}", "children"),
    Output(f"val_card_COGS_{_City}", "children"),
    Output(f"subt_card_COGS_{_City}", "children"),
    Output(f"val_card_Cust_{_City}", "children"),
    Output(f"subt_card_Cust_{_City}", "children"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
)  
  
def update_cards(month,city):    
    #Order : Sales, text_sales, Customers, text_custo, Gross_Income, text_gross
    return calc_cards_vals(month,city)

#Line product
@callback(
    Output(f"product_line_totals_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
    )

def make_treemap(month,city):
    return make_product_line_tree(month,city)

#Last Column Charts
@callback(
    Output(f"Custotype_bar_{_City}", "figure"),
    Output(f"Gender_bar_{_City}", "figure"),
    Output(f"Payment_pie_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
    )

def create_customers_plots(month, city):
    return make_count_type_bar(month,city,"Customer type"), make_count_type_bar(month,city,"Gender"), make_circles_gender_payment_city(month,city)

#################################################################################################
#Barplot All
_City = "All"
@callback(
    Output(f"barplot_for_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"radio_types_{_City}","value"),
    Input(f"tab_{_City}","children")
)  

def make_plots(month,type,city):
    return make_bar_plot_city(month,type,city)

#Updating Cards 
#Sales {totaT}, Gross Income {gross income}, COGS{cogs}, Customers
@callback(
    Output(f"val_card_Sale_{_City}", "children"),
    Output(f"subt_card_Sale_{_City}", "children"),
    Output(f"val_card_Gros_{_City}", "children"),
    Output(f"subt_card_Gros_{_City}", "children"),
    Output(f"val_card_COGS_{_City}", "children"),
    Output(f"subt_card_COGS_{_City}", "children"),
    Output(f"val_card_Cust_{_City}", "children"),
    Output(f"subt_card_Cust_{_City}", "children"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
)  
  
def update_cards(month,city):    
    #Order : Sales, text_sales, Customers, text_custo, Gross_Income, text_gross
    return calc_cards_vals(month,city)

#Line product
@callback(
    Output(f"product_line_totals_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
    )

def make_treemap(month,city):
    return make_product_line_tree(month,city)

#Last Column Charts
@callback(
    Output(f"Custotype_bar_{_City}", "figure"),
    Output(f"Gender_bar_{_City}", "figure"),
    Output(f"Payment_pie_{_City}", "figure"),
    Input(f"radio_month_{_City}","value"),
    Input(f"tab_{_City}","children")
    )

def create_customers_plots(month, city):
    return make_count_type_bar(month,city,"Customer type"), make_count_type_bar(month,city,"Gender"), make_circles_gender_payment_city(month,city)

####################################################################################

@callback(
    Output("store-trained-model", "data"),
    Output("feature_importance_plot", "figure"),
    Input("update-button", "n_clicks"),
    State("ftimp_dpdn_vars", "value"),
    State("ftimp_radio_target", "value"),
)
def create_ftimp_plot(n_clicks, cat_columns, target):
    ctx = dash.callback_context
    # if not ctx.triggered:
    #     raise PreventUpdate

    # if n_clicks is None:
    #     raise PreventUpdate

    if cat_columns is []:
        raise PreventUpdate()

    dff = data_encoding(cat_columns, target)
    return make_ftimp_analysis(dff, target)

# Curves
# Curves
@callback(
    Output("learning_curve", "figure"),
    Input("update-button", "n_clicks"),
    State("ftimp_dpdn_vars", "value"),
    State("ftimp_radio_target", "value"),
    State("store-trained-model", "data"),
)
def create_predictors_plot(n_clicks, cat_columns, target, stored_data):
    # ctx = dash.callback_context
    # if not ctx.triggered:
    #     raise PreventUpdate

    # if n_clicks is None:
    #     raise PreventUpdate

    if cat_columns is []:
        raise PreventUpdate()


    dff = data_encoding(cat_columns, target)
    return make_train_curve(dff, target)


@callback(
    Output("validation_curve", "figure"),
    Input("ftimp_dpdn_vars", "value"),
    Input("ftimp_radio_target", "value"),
)
def create_predictors_plot(cat_columns, target):

    if cat_columns is []:
        raise PreventUpdate()


    dff = data_encoding(cat_columns, target)
    return make_validation_curve(dff, target)

######################
@callback(
    Output('dropdown-container', 'children'),
    Input('ftimp_dpdn_vars', 'value')
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
    Output('button', 'children'),
    [Input('ftimp_dpdn_vars', 'value'),
     Input("ftimp_radio_target","value"),
     Input({'type': 'dynamic-dropdown', 'index': ALL}, 'value')]
)
def update_button_state(selected_keys, target, selected_values):
    if not selected_keys or None in selected_values:
        return True, f"Predict a {target}"
    return False, f"Predict a {target}"

@callback(
    [Output('output-vector', 'children'),
     Output("output-variables","children"),
     Output({'type': 'dynamic-dropdown', 'index': ALL}, 'value')],
    [Input('button', 'n_clicks'),
     Input("store-trained-model","data"),
     Input("ftimp_radio_target","value")],
    [State('ftimp_dpdn_vars', 'value'),
     State({'type': 'dynamic-dropdown', 'index': ALL}, 'value'),
     State({'type': 'dynamic-dropdown', 'index': ALL}, 'id')]
)
def update_output(n_clicks, model, target, selected_keys, selected_values, selected_ids):
    if not n_clicks:
        raise PreventUpdate

    if None in selected_values:
        raise PreventUpdate
    
    selected_dict = {item['index']: values for item, values in zip(selected_ids, selected_values)}

    encoded_vector = []
    user_defined = {}
    for key in variables.keys():  # iterate through all keys in the fixed order
        if key in selected_keys:
            user_defined[key] = []
            for option in variables[key]:
                if selected_dict.get(key) and option in selected_dict[key]:
                    encoded_vector.append(1)
                    user_defined[key].append(option)
                else:
                    encoded_vector.append(0)

    encoded_vector = np.array(encoded_vector).reshape(1, -1)
    prediction = deserialize_model(model).predict(encoded_vector)
    
    # Clear options of all dropdowns
    cleared_values = [None] * len(selected_values)
    
    # Prepare the output message
    #user_defined_str = ', '.join([f"{key}: {' , '.join(values)}" for key, values in user_defined.items()])
    user_defined_str = ' | '.join([f"{key}: \"{', '.join(values)}\"" for key, values in user_defined.items()])
    
    return f"The {target} prediction is: {prediction.round(2)}", user_defined_str,cleared_values


if __name__=='__main__':
    app.run_server(debug=True, port=8050)