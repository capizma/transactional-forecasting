from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import run_forecast
import datetime
from datetime import date, timedelta, datetime
from dash.exceptions import PreventUpdate
import metrics
import dash_bootstrap_components as dbc

app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])

fname = 'Faster_Payment_Inbound_Retail.csv'

df = pd.read_csv('./data/'+fname)
df = df[['reporting_date','volume']]
df = df.sort_values(by='reporting_date')
df.columns = ['ds','y']
main_date = datetime.strptime(df['ds'].max(),"%Y-%m-%d")
until_date = datetime.strptime(df['ds'].max(),"%Y-%m-%d") + timedelta(days=365)

def produce_metrics(df):
    ls = []

    metrics_ls = metrics.generate_metrics(df)

    #ls.append(dbc.Col(html.Div(dbc.Card("Mean value - " + str(metrics_ls[1]), body=True))))
    #ls.append(dbc.Col(html.Div(dbc.Card("Max value - " + str(metrics_ls[2]) + " on "+str(metrics_ls[3]), body=True))))  
    #ls.append(dbc.Col(html.Div(dbc.Card("Min value - " + str(metrics_ls[4]) + " on "+str(metrics_ls[5]), body=True))))  
    #ls.append(dbc.Col(html.Div(dbc.Card("Standard Deviation - " + str(metrics_ls[6]), body=True))))  
    #ls.append(dbc.Col(html.Div(dbc.Card("90 Day Trend - " + str(metrics_ls[7]), body=True))))

    ls.append(
        dbc.Card(
            dbc.CardBody([
                html.H6("Mean volume", className="card-subtitle",style={'font-size':'14px'}),
                html.P(str(metrics_ls[1]),className="card-text",style={'font-size':'14px'})
            ]
        ), style = {'flex-grow':'1'},className='card_box')
    ),
    ls.append(
        dbc.Card(
            dbc.CardBody([
                html.H6("Max volume", className="card-subtitle",style={'font-size':'14px'}),
                html.P(str(metrics_ls[2]) + " on "+str(metrics_ls[3]),className="card-text",style={'font-size':'14px'})
            ]
        ), style = {'flex-grow':'1'},className='card_box')
    ),
    ls.append(
        dbc.Card(
            dbc.CardBody([
                html.H6("Min volume", className="card-subtitle",style={'font-size':'14px'}),
                html.P(str(metrics_ls[4]) + " on "+str(metrics_ls[5]),className="card-text",style={'font-size':'14px'})
            ]
        ), style = {'flex-grow':'1'},className='card_box')
    ),
    ls.append(
        dbc.Card(
            dbc.CardBody([
                html.H6("90 Day Trend", className="card-subtitle",style={'font-size':'14px'}),
                html.P(str(metrics_ls[7]),className="card-text",style={'font-size':'14px'})
            ]
        ), style = {'flex-grow':'1'},className='card_box')
    ),

    return html.Div(
    html.Div(ls,style={'display':'flex','flex-wrap':'wrap','gap':'5px','margin':'5px','margin-bottom':'0'}),id='actuals-metrics')

    return dbc.Row(ls)

def produce_initial_figure(test_show):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_show['ds'],
                    y=test_show['y'],
                    mode='lines',
                    name='y'
                    ))

    return fig

def produce_ci_figure(test_show):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_show['ds'],
                    y=test_show['y'],
                    mode='lines',
                    name='y'
                    ))
    fig.add_trace(go.Scatter(x=test_show['ds'],
                    y=test_show['yhat'],
                    mode='lines',
                    name='yhat'
                    ))
    
    fig.add_traces([go.Scatter(x = test_show['ds'], y = test_show['yhat_upper'],
                               mode = 'lines', line_color = 'rgba(0,0,0,0)',
                               showlegend = False),
                    go.Scatter(x = test_show['ds'], y = test_show['yhat_lower'],
                               mode = 'lines', line_color = 'rgba(0,0,0,0)',
                               name = '95% confidence interval',
                               fill='tonexty', fillcolor = 'rgba(255, 0, 0, 0.2)')])

    return fig

@callback(
    Output('graph-content', 'figure'),
    Output('cv-metrics','children'),
    Output('cv-period','children'),
    Input('run-forecast', 'n_clicks'),
    State('date-picker-past', 'start_date'),
    State('date-picker-past', 'end_date'),
    State('date-picker-future', 'date'),
    State('direction-picker', 'value'),
    prevent_initial_call=True
)
def update_graph(value,start_date,end_date,future_date,direction_value):
    if direction_value == 'Past':
        if start_date is not None and end_date is not None:
            if end_date > start_date:
                #start_date = date.fromisoformat(start_date)
                #end_date = date.fromisoformat(end_date)
                
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        
                results = run_forecast.run_past_forecast(fname,start_date,end_date)
                
                return produce_ci_figure(results[0]), [html.P("sMAPE - " + str(results[1]))], [html.P("Validation Period - " + str(results[2]))]
    if direction_value == 'Future':
        if future_date is not None:
            #future_date = date.fromisoformat(future_date)
            
            future_date = datetime.strptime(future_date, "%Y-%m-%d").date()
    
            results = run_forecast.run_forecast(fname,future_date)
            
            return produce_ci_figure(results[0]), [html.P("sMAPE - " + str(results[1]))], [html.P("Validation Period - " + str(results[2]))]
    raise PreventUpdate
    
@callback(
    Output('date-picker-future', 'disabled'),
    Output('date-picker-past', 'disabled'),
    Input('direction-picker', 'value'),
    prevent_initial_call=True
)
def update_direction(value):
    if value == 'Past':
        return True, False
    else:
        return False, True

app.layout = html.Div([
    html.H1(children='Header', style={'textAlign':'center'}),
    #html.Div([produce_metrics(df)],id='metrics-values'),
    
    html.Div([
        produce_metrics(df)
    ],id='actuals-metrics-parent'),

    html.Div([
        html.Div([
            dbc.Button('Run forecast!',color="danger",id='run-forecast',className='grow3'),
            html.Div([
                html.P("Window to forecast"),

                dcc.RadioItems(
                    options=['Past', 'Future'],
                    value='Future',
                    id='direction-picker'
                ),

                dcc.DatePickerSingle(
                    id='date-picker-future',
                    min_date_allowed=date(main_date.year, main_date.month, main_date.day) + timedelta(days=1),
                    max_date_allowed=date(until_date.year, until_date.month, until_date.day),
                    initial_visible_month=date(main_date.year, main_date.month, main_date.day)+ timedelta(days=1),

                    placeholder='Future date'
                ),
                
                dcc.DatePickerRange(
                    id='date-picker-past',
                    min_date_allowed=date(main_date.year, main_date.month, main_date.day) - timedelta(days=120),
                    max_date_allowed=date(main_date.year, main_date.month, main_date.day),
                    initial_visible_month=date(main_date.year, main_date.month, main_date.day) - timedelta(days=120),

                    start_date_placeholder_text='Past start',
                    end_date_placeholder_text='Past end',

                    disabled=True
                ),

            ],style={'border-bottom':'1px solid #bbb','padding':'10px'},className='grow5'),  
        ],className='iflex_container4'),
        
        html.Div([            
            dcc.Tabs([
                dcc.Tab(label='Graph',children=[
                    html.Div([
                        dcc.Graph(figure=produce_initial_figure(df),id='graph-content')     
                    ],id='graph-content-parent',style={'align-items':'center'})
                ])
            ],style={'font-size':'14px'})
        ],className='graph_controls'),

        html.Div([
            dbc.Button('Download results',color="danger",id='btn_csv',className='grow3'),
            dbc.Button('Download source',color="danger",id='btn_csv_source',className='grow3'),
            html.Div([
                html.P("Performance Metrics"),
                html.Div([
                    html.Div([html.P("Empty")],id='cv-metrics'),
                    html.Div([html.P("Validation Period")],id='cv-period')
                ])
            ],style={'border-bottom':'1px solid #bbb','padding':'10px'},className='grow4')
        ],className='iflex_container4')
    ],className = 'iflex_container3'),

])
    
if __name__ == '__main__':
  app.run_server(debug=False)