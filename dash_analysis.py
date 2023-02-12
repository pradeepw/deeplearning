import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc

from analysis_functions import *
from dash.dependencies import Input, Output

predictions_df = load_analysis_data('predictions.csv')

swin_predictions = predictions_df[predictions_df['model'] == 'swin']
vit_predictions = predictions_df[predictions_df['model'] == 'vit']
deit_predictions = predictions_df[predictions_df['model'] == 'deit']

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
app.layout = html.Div(
    children=[
        dbc.Row([
            dbc.Col([
                html.Label(
                    ['Select Image Labels'],
                    style={'font-weight': 'bold', "text-align": "right", "offset": 1}
                ),
                dcc.Dropdown(
                    value='ALL',
                    id='label-dropdown',
                    options=[
                                {
                                    'label': 'ALL',
                                    'value': 'ALL'
                                }
                            ] + [
                                {
                                    'label': i,
                                    'value': i
                                } for i in predictions_df.actual.unique()
                            ],
                    multi=True,
                    placeholder='Filter by image label...'
                )
            ]),
            dbc.Col([
                html.Label(
                    ['Select Image Imputation'],
                    style={'font-weight': 'bold', "text-align": "right", "offset": 1}
                ),
                dcc.Dropdown(
                    value='ALL',
                    id='imputation-dropdown',
                    options=[
                                {
                                    'label': 'ALL',
                                    'value': 'ALL'
                                }
                            ] + [
                                {
                                    'label': i,
                                    'value': i
                                } for i in predictions_df.imputation.unique()
                            ],
                    multi=True,
                    placeholder='Filter by image imputation...'
                )
            ])
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(id='swin')]),
            dbc.Col([dcc.Graph(id='vit')]),
            dbc.Col([dcc.Graph(id='deit')])
        ])
    ]
)


@app.callback(
    Output('swin', 'figure'),
    Input('label-dropdown', 'value'),
    Input('imputation-dropdown', 'value')
)
def get_swin_analysis(selected_labels, selected_imputations):
    filtered_df = predictions_df[predictions_df.model == 'swin']

    if 'ALL' not in selected_labels:
        filtered_df = filtered_df[filtered_df.actual.isin(selected_labels)]

    if 'ALL' not in selected_imputations:
        filtered_df = filtered_df[filtered_df.imputation.isin(selected_imputations)]

    performance_metrics = compute_performance_metric_model(filtered_df)
    fig = px.line(
        performance_metrics,
        x="metric",
        y="value",
        color='imputation',
        title='SWIN'
    )
    fig.update_layout(transition_duration=500)

    return fig

@app.callback(
    Output('vit', 'figure'),
    Input('label-dropdown', 'value'),
    Input('imputation-dropdown', 'value')
)
def get_vit_analysis(selected_labels, selected_imputations):
    filtered_df = predictions_df[predictions_df.model == 'vit']

    if 'ALL' not in selected_labels:
        filtered_df = filtered_df[filtered_df.actual.isin(selected_labels)]

    if 'ALL' not in selected_imputations:
        filtered_df = filtered_df[filtered_df.imputation.isin(selected_imputations)]

    performance_metrics = compute_performance_metric_model(filtered_df)
    fig = px.line(
        performance_metrics,
        x="metric",
        y="value",
        color='imputation',
        title='VIT'
    )
    fig.update_layout(transition_duration=500)

    return fig

@app.callback(
    Output('deit', 'figure'),
    Input('label-dropdown', 'value'),
    Input('imputation-dropdown', 'value')
)
def get_deit_analysis(selected_labels, selected_imputations):
    filtered_df = predictions_df[predictions_df.model == 'deit']

    if 'ALL' not in selected_labels:
        filtered_df = filtered_df[filtered_df.actual.isin(selected_labels)]

    if 'ALL' not in selected_imputations:
        filtered_df = filtered_df[filtered_df.imputation.isin(selected_imputations)]

    performance_metrics = compute_performance_metric_model(filtered_df)
    fig = px.line(
        performance_metrics,
        x="metric",
        y="value",
        color='imputation',
        title='DEIT'
    )
    fig.update_layout(transition_duration=500)

    return fig

app.title = 'Vision Transformers - Prediction Analysis'

if __name__ == '__main__':
    app.run_server(debug=True)
