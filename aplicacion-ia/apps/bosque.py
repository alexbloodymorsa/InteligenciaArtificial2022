from dash import Dash, html, dcc, dash_table           #para componentes core de dash y html           
from dash.dependencies import Input, Output, State    #
import pathlib                                  #para la manipulación de sistemas de archivos
import dash

import base64
import datetime
import io

from app import app
import plotly.express as px

import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import plotly.graph_objects as go

def generate_tree(estimadores, arbol_mostrar, feature_names_x, feature_names_y, df):
    for x in df.columns:
        if str(df[x].dtype) is 'object':
            df[x] = pd.factorize(df[x])[0]

    X = np.array(df[feature_names_x])
    Y = np.array(df[feature_names_y])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1234, shuffle = True)
    bosque = RandomForestRegressor(n_estimators = estimadores, random_state=0, max_depth=8, min_samples_leaf=2, min_samples_split=4)
    bosque.fit(X_train, Y_train)

    feature_names = feature_names_x

    score = bosque.score(X_validation, Y_validation)

    model = bosque.estimators_[arbol_mostrar]

    labels = [''] * model.tree_.node_count
    parents = [''] * model.tree_.node_count
    labels[0] = 'root'
    for i, (f, t, l, r) in enumerate(zip(
        model.tree_.feature,
        model.tree_.threshold,
        model.tree_.children_left,
        model.tree_.children_right,
    )):
        if l != r:
            labels[l] = f'{feature_names[f]} <= {t:g}'
            labels[r] = f'{feature_names[f]} > {t:g}'
            parents[l] = parents[r] = labels[i]

    fig = go.Figure(go.Treemap(
        branchvalues='total',
        labels=labels,
        parents=parents,
        values=model.tree_.n_node_samples,
        textinfo='label+value+percent root',
        marker=dict(colors=model.tree_.impurity),
        customdata=list(map(str, model.tree_.value)),
        hovertemplate='''
    <b>%{label}</b><br>
    impurity: %{color}<br>
    samples: %{value} (%{percentRoot:%.2f})<br>
    value: %{customdata}'''
    ))

    return fig, ('Score: ' + str(score))

def graph_correlation(dataframe):
    correlacion = pd.DataFrame(dataframe.corr())
    fig = px.imshow(correlacion[correlacion.columns])
    return fig
    

layout = html.Div(children =[
    html.H1(children='Algoritmo de Bosque Aleatorio', style={"textAlign": "center"}),
    html.P('''Un bosque aleatorio es la combinación de muchos árboles de decisión que solo operan 
        sobre secciones aleatorias de los datos. Cada uno de los árboles toma una decisión y se genera 
        un promedio o una votación de todos los resultados para obtener el resultado del bosque aleatorio.'''),

    html.H2('Subir los datos a utilizar:'),
    dcc.Upload(
        id='upload-data-bosque',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),

    html.H2('Tabla generada'),
    html.Div(id='output-data-bosque'),

    html.H2('Matriz de Correlación'),
    html.P('''La matriz de correlación indica qué tan relacionada está cada una de las variables.
        Se busca elegir un grupo de variables que no se parezcan mucho entre sí.'''),
    dcc.Graph(id='output-corr-bosque'),

    html.H2('Variables a utilizar'),
    dcc.Dropdown(id='memory-bosque-x', multi=True),

    html.H2('Variable a predecir'),
    dcc.Dropdown(id='memory-bosque-y', multi=False),

    html.H2('Generar Bosque'),
    html.P('El número de estimadores representa el número de árboles que se utiliza para generar el bosque.'),
    dcc.Input(id='input-bosque1', type='number', placeholder='número de estimadores',min=50, max=300,step=1),
    html.P('Selecciona el número de árbol para ver un pedazo del bosque.'),
    dcc.Input(id='input-bosque2', type='number', placeholder='árbol a mostrar',min=0, max=300,step=1),
    dcc.Graph(id = 'output-grafica-bosque'),
    html.P('El score indica del 0 al 1 qué tan bueno es un modelo.'),
    html.H2(id='output-score-bosque'),

    dcc.Store(id='df-bosque')

], style = {'width': '70%', 'margin-left':'auto', 'margin-right':'auto'})

@app.callback(
    Output('output-grafica-bosque', 'figure'),
    Output('output-score-bosque', 'children'),
    Input('input-bosque1', 'value'),
    Input('input-bosque2', 'value'),
    Input('memory-bosque-x', 'value'),
    Input('memory-bosque-y', 'value'),
    Input('df-bosque', 'data')
)
def update_output(numero_estimadores, arbol_mostrar, bosque_x, bosque_y, dicc):
    if arbol_mostrar is None:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(dicc)

    return generate_tree(numero_estimadores, arbol_mostrar, bosque_x, bosque_y, df)


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'id': c, 'name': c} for c in df.columns],
            virtualization=True,
            fixed_rows={'headers': True},
            style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 300},
            style_table={'height': 300}
        )
    ]), df

@app.callback(Output('output-data-bosque', 'children'),
              Output('output-corr-bosque', 'figure'),
              Output('df-bosque', 'data'),
              Output('memory-bosque-x', 'options'),
              Output('memory-bosque-y', 'options'),
              Input('upload-data-bosque', 'contents'),
              State('upload-data-bosque', 'filename'),
              State('upload-data-bosque', 'last_modified'))
def update_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    if list_of_contents is not None:
        children, data = parse_contents(list_of_contents, list_of_names, list_of_dates)
        grafica = graph_correlation(data)
        opciones = data.columns

        return children, grafica, data.to_dict(), opciones, opciones