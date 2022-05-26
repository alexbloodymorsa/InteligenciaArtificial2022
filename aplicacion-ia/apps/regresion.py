from dash import Dash, html, dcc, callback_context, ALL, dash_table   #para componentes core de dash y html           
from dash.dependencies import Input, Output, State     #
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

#Se importan las bibliotecas necesarias para generar el modelo de regresión logística
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def graph_correlation(dataframe):
    correlacion = pd.DataFrame(dataframe.corr())
    fig = px.imshow(correlacion[correlacion.columns])
    return fig

layout = html.Div(children =[
    html.H1(children='Algoritmo de Regresión Logística', style={"textAlign": "center"}),
    html.P('''La regresión logística permite hacer clasificaciones de datos. Se utiliza una 
        función matemática llamada sigmoide para generar una probabilidad de que el dato sea de una 
        clase o de otra. A partir de esa probabilidad se puede predecir a qué clase pertenece un dato.
        ¡Sube un archivo y genera una predicción a partir de un modelo de regresión logística!'''),

    html.H2('Subir los datos a utilizar:'),
    dcc.Upload(
        id='upload-data-regr',
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
    html.Div(id='output-data-regr'),

    html.H2('Matriz de Correlación'),
    html.P('''La matriz de correlación indica qué tan relacionada está cada una de las variables.
        Se busca elegir un grupo de variables que no se parezcan mucho entre sí.'''),
    dcc.Graph(id='output-corr-regr'),

    html.H2('Variables a utilizar'),
    dcc.Dropdown(id='memory-regr-x', multi=True),
    dcc.Store(id='memory-regr-out-x'),

    html.H2('Variable a predecir'),
    dcc.Dropdown(id='memory-regr-y', multi=False),
    dcc.Store(id='memory-regr-out-y'),

    html.H2('Generar Clasificación'),
    html.Button('Ingresar valores para predicción', id='btn-reg', n_clicks=0),
    html.Div(id='valores-regresion'),
    html.Button('Predecir', id='btn-pred', n_clicks=0),

    html.Div(id='prediccion-regresion'),
    html.P('El score indica del 0 al 1 qué tan bueno es un modelo.'),

    dcc.Store(id='df-regr'),

], style = {'width': '70%', 'margin-left':'auto', 'margin-right':'auto'})


@app.callback(
    Output('memory-regr-out-x', 'data'),
    Output('memory-regr-out-y', 'data'),
    Output('valores-regresion', 'children'),
    Input('btn-reg', 'n_clicks'),
    Input('memory-regr-x', 'value'),
    Input('memory-regr-y', 'value'),
    Input('df-regr', 'data')
)
def update_output(btn1, regr_x, regr_y, dict):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    predecir = []
    df = pd.DataFrame(dict)

    if 'btn-reg' in changed_id:
        for x in df.columns:
            if str(df[x].dtype) is 'object':
                df[x] = pd.factorize(df[x])[0]

        X = np.array(df[regr_x])
        Y = np.array(df[regr_y])
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1234, shuffle = True)
        Clasificacion = linear_model.LogisticRegression()
        Clasificacion.fit(X_train, Y_train)

        score = Clasificacion.score(X_validation, Y_validation)

        predecir.append(html.H2('Predicciones'))

        i = 0
        for columna in regr_x:
            predecir.append(html.H3(columna))
            i += 1
            predecir.append(dcc.Input(id={'type':'input-regr','index':i}, type='number', placeholder=columna))
        
        predecir.append(html.H2('Score: ' + str(score)))

        return regr_x, regr_y, predecir 
    else:
        return regr_x, regr_y, html.H2('Sin predicciones')

@app.callback(
    Output('prediccion-regresion', 'children'),
    Input('btn-pred', 'n_clicks'),
    Input('memory-regr-out-x','data'),
    Input('memory-regr-out-y','data'),
    Input({'type': 'input-regr', 'index': ALL}, 'value'),
    Input('df-regr', 'data')
)
def create_prediction(btn, regr_x, regr_y, valores, dict):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    prediccion = {}

    df = pd.DataFrame(dict)

    if 'btn-pred' in changed_id:
        for x in regr_x:
            prediccion[x] = []
        
        i = 0
        for valor in valores:
            prediccion.update({regr_x[i] : [valor]})
            i += 1

        X = np.array(df[regr_x])
        Y = np.array(df[regr_y])
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 1234, shuffle = True)
        Clasificacion = linear_model.LogisticRegression()
        Clasificacion.fit(X_train, Y_train)

        valor_prediccion = Clasificacion.predict(pd.DataFrame(prediccion))[0]

        return html.H2('Valor de predicción: ' + str(valor_prediccion))
    else:
        return ''

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

@app.callback(Output('output-data-regr', 'children'),
              Output('output-corr-regr', 'figure'),
              Output('df-regr', 'data'),
              Output('memory-regr-x', 'options'),
              Output('memory-regr-y', 'options'),
              Input('upload-data-regr', 'contents'),
              State('upload-data-regr', 'filename'),
              State('upload-data-regr', 'last_modified'))
def update_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    if list_of_contents is not None:
        children, data = parse_contents(list_of_contents, list_of_names, list_of_dates)
        grafica = graph_correlation(data)
        opciones = data.columns

        return children, grafica, data.to_dict(), opciones, opciones