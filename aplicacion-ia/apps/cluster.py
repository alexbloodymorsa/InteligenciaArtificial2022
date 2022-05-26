from dash import Dash, html, dcc, dash_table			#para componentes core de dash y html			
from dash.dependencies import Input, Output, State		#
import dash

import base64
import datetime
import io

from app import app
import plotly.express as px

#Bilbiotecas para clustering
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler 	# Estandarizar
from sklearn.cluster import KMeans 								# Kmeans
from kneed import KneeLocator 									# Método del codo

def graph_correlation(dataframe):
    correlacion = pd.DataFrame(dataframe.corr())
    fig = px.imshow(correlacion[correlacion.columns])
    return fig

layout = html.Div(children =[
    html.H1(children='Algoritmo de Clustering Particional', style={"textAlign": "center"}),
    html.P('''Algoritmo que genera distintas agrupaciones para clasificar un conjunto de datos. 
        Cada una de las agrupaciones representa un cluster. Estas agrupaciones se generan con ciertos 
        criterios de cercanía (distancias). El problema de este algoritmo es interpretar qué es cada 
        una de las agrupaciones.'''),

    html.H2('Subir los datos a utilizar:'),
    dcc.Upload(
        id='upload-data-cluster',
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
    html.Div(id='output-data-cluster'),

    html.H2('Matriz de Correlación'),
    html.P('''La matriz de correlación indica qué tan relacionada está cada una de las variables.
        Se busca elegir un grupo de variables que no se parezcan mucho entre sí.'''),
    dcc.Graph(id='output-corr-cluster'),

    html.H2('Variables a utilizar'),
    dcc.Dropdown(id='memory-cluster', multi = True),


    html.H2('Número de clústeres:'),
    html.P('Se puede elegir la cantidad de grupos en las que se dividen los datos a trabajar. '),
    dcc.Input(id='input-cluster1', type='number', placeholder='numero clústeres',min=2, max=12,step=1),

    html.H2('Clustering de la tabla'),
    dcc.Graph(id = 'output-grafica1'),

    dcc.Store(id='df-cluster'),


], style = {'width': '70%', 'margin-left':'auto', 'margin-right':'auto'})

@app.callback(
    Output('output-grafica1', 'figure'),
    Input('df-cluster', 'data'),
    Input('input-cluster1', 'value'),
    Input('memory-cluster','value')
)
def update_output(dict, num_clusters, feature_values):
    if num_clusters is not None:

        df = pd.DataFrame(dict)

        for x in df.columns:
            if str(df[x].dtype) is 'object':
                df[x] = pd.factorize(df[x])[0]

        matriz = np.array(df[feature_values])
        estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
        MEstandarizada = estandarizar.fit_transform(matriz)   # Se calculan la media y desviación y se escalan los datos
        SSE = []
        for i in range(2, 12):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(MEstandarizada)
            SSE.append(km.inertia_)
        kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
        MParticional = KMeans(n_clusters=num_clusters, random_state=0).fit(MEstandarizada)
        MParticional.predict(MEstandarizada)
        grafica = px.scatter(x=MEstandarizada[:,0], y=MEstandarizada[:,1], color=MParticional.labels_)

        return grafica
    else:
        return px.scatter()


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

@app.callback(Output('output-data-cluster', 'children'),
              Output('output-corr-cluster', 'figure'),
              Output('df-cluster', 'data'),
              Output('memory-cluster', 'options'),
              Input('upload-data-cluster', 'contents'),
              State('upload-data-cluster', 'filename'),
              State('upload-data-cluster', 'last_modified'))
def update_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    if list_of_contents is not None:
        children, data = parse_contents(list_of_contents, list_of_names, list_of_dates)
        grafica = graph_correlation(data)
        opciones = data.columns

        return children, grafica, data.to_dict(), opciones