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
                    href="https://github.com/Salvatore-Rocha",
                    target="_blank",
                ),
                html.A(
                    "Code (.py file) |   ",
                    href="",
                    target="_blank",
                ),
            ]
        ),
    ]
)

def create_card(title,City):
    card = dbc.Card(
        dbc.CardBody([
                html.H3([
                        html.I(className="bi bi-bank me-2"), 
                            title
                        ]),
                html.H5(
                        children= {}, 
                        id= f"val_card_{title[0:4]}_{City}"
                        ),
                html.H6(
                        html.I(children = {},
                                id= f"subt_card_{title[0:4]}_{City}", 
                                className="bi bi-caret-up-fill text-success")
                        ),
                    ]),
        className="text-center",
        style={"width": "18rem"})   
    return card 

def create_tab(City):
    tab = dbc.Tab([ 
            html.H1(
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
                                html.H5("Select a month to see the sales behaviour"),
                                dcc.RadioItems(id=f"radio_month_{City}",
                                            options= ['January', 'February','March', "All"], 
                                            value='February', 
                                            inline=True,
                                            className="text-success",
                                            inputStyle={"margin-left":"6px", "margin-right": "2px"})
                                ],width=6 ),
                            dbc.Col([
                                html.H5("Select what you wanna see"),
                                dcc.RadioItems(id =f"radio_types_{City}",
                                        options= ['Customers', 
                                                    "Sales, Gross Income & COGS",
                                                    "Product Line"],
                                        value= 'Sales, Gross Income & COGS', 
                                        inline=False,
                                        className="text-success",
                                        inputStyle={"margin-left":"6px", "margin-right": "2px"})
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
                                 style={'marginLeft': '-50px','marginRight': '-80px'}
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
                ], label=f"{City}",)
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

cmap_product_lines = { #If we dont use a color map, the plots will assing arbitrary colors each time to each plot
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
                 height=850,
                 color = "Product line",
                 color_discrete_map= cmap_product_lines
                 )
    fig.update_traces(textinfo='label+value',
                      texttemplate='%{label} <br>$%{value:,.0f}',
                      hovertemplate=' Line: %{label} <br> Total Sales: $%{value:,.0f}')
    fig.update_layout(title=dict(text=f"Line Products Total Sales - {month}",font=dict(size=20),x=0.5, y=0.95),
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
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', xanchor='right',y=1.02, x=1),
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

def make_plots_last_column(month,city):
    return make_count_type_bar(month,city,"Customer type"), \
           make_count_type_bar(month,city,"Gender"),\
           make_circles_gender_payment_city(month,city)
   
#The branches are located in Yangon, Naypyitaw, Mandalay
tab1 = create_tab("Yangon")
tab2 = create_tab("Naypyitaw")
tab3 = create_tab("Mandalay")
tab4 = create_tab("All")
tab5 = dbc.Tab([ 
                dbc.Row([
                    dbc.Col([
                        html.H1("Title"),
                        html.P("Text")
                            ]
                        ,width= 3),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(id= "dpdn_left", 
                                             options = ['New York City', 'Montreal', 'Paris', 'London'],
                                             value= "London",
                                             )
                                    ]),
                            dbc.Col([
                                dcc.Dropdown(id= "dpdn_right", 
                                             options = ['New York City', 'Montreal', 'Paris', 'London'],
                                             value= "London",
                                             )
                                    ])
                                ]),
                        dcc.Graph(id = "sankey_plot",
                                  figure = {}                            
                            )
                            ])
                ])
                ], label="Dynamic Connections",)
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
    return make_plots_last_column(month,city)


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
    return make_plots_last_column(month,city)

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
    return make_plots_last_column(month,city)
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
    return make_plots_last_column(month,city)


if __name__=='__main__':
    app.run_server(debug=True, port=8050)