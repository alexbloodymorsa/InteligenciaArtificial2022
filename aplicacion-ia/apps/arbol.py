from dash import Dash, html, dcc, dash_table             #para componentes core de dash y html           
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



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import plotly.graph_objects as go

def graph_correlation(dataframe):
    correlacion = pd.DataFrame(dataframe.corr())
    fig = px.imshow(correlacion[correlacion.columns])
    return fig
    

layout = html.Div(children =[
    html.H1(children='Algoritmo de Árbol de Decisión', style={"textAlign": "center"}),
    html.P('''Un árbol de decisión representa un algoritmo que ayuda a tomar decisiones. 
        A partir de un conjunto de decisiones se clasifica a un dato o se predice un valor. 
        Las decisiones agarran una variable del conjunto de datos y preguntan si se es mayor o 
        menor que un valor determinado. Dependiendo de las diferentes decisiones que se tomen 
        se llega a un resultado final.'''),

    html.H2('Subir los datos a utilizar:'),
    dcc.Upload(
        id='upload-data-arbol',
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
    html.Div(id = 'output-data-arbol'),

    html.H2('Matriz de Correlación'),
    html.P('''La matriz de correlación indica qué tan relacionada está cada una de las variables.
        Se busca elegir un grupo de variables que no se parezcan mucho entre sí.'''),
    dcc.Graph(id = 'output-corr-arbol'),

    html.H2('Variables a utilizar'),
    dcc.Dropdown(id='memory-arbol-x', multi=True),

    html.H2('Variable a predecir'),
    dcc.Dropdown(id='memory-arbol-y', multi=False),

    html.H2('Generar Árbol'),
    html.P('La profundidad máxima de un árbol de decisión representa el máximo de decisiones antes de que se genere un valor final.'),
    dcc.Input(id='input-arbol', type='number', placeholder='profunidad máxima',min=2, max=15,step=1),
    
    dcc.Graph(id = 'output-grafica-arbol'),
    html.P('El score indica del 0 al 1 qué tan bueno es un modelo.'),
    html.H2(id = 'output-score-arbol'),

    dcc.Store(id='df-arbol')

], style = {'width': '70%', 'margin-left':'auto', 'margin-right':'auto'})

@app.callback(
    Output('output-grafica-arbol', 'figure'),
    Output('output-score-arbol', 'children'),
    Input('input-arbol', 'value'),
    Input('memory-arbol-x', 'value'),
    Input('memory-arbol-y', 'value'),
    Input('df-arbol', 'data')
)
def update_output(profunidad_maxima, feature_names_x, feature_names_y, dicc):
    if profunidad_maxima is None:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(dicc)

    for x in df.columns:
        if str(df[x].dtype) is 'object':
            df[x] = pd.factorize(df[x])[0]

    print(df.columns)
    X = np.array(df[feature_names_x])
    Y = np.array(df[feature_names_y])
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1234, shuffle = True)
    model = DecisionTreeRegressor(max_depth=profunidad_maxima, min_samples_split=4, min_samples_leaf=2, random_state=0)
    model.fit(X_train, Y_train)
    feature_names = feature_names_x

    score = model.score(X_validation, Y_validation)

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

@app.callback(Output('output-data-arbol', 'children'),
              Output('output-corr-arbol', 'figure'),
              Output('df-arbol', 'data'),
              Output('memory-arbol-x', 'options'),
              Output('memory-arbol-y', 'options'),
              Input('upload-data-arbol', 'contents'),
              State('upload-data-arbol', 'filename'),
              State('upload-data-arbol', 'last_modified'))
def update_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    if list_of_contents is not None:
        children, data = parse_contents(list_of_contents, list_of_names, list_of_dates)
        grafica = graph_correlation(data)
        opciones = data.columns

        return children, grafica, data.to_dict(), opciones, opciones