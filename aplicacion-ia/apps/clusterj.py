from dash import Dash, html, dcc, dash_table				#para componentes core de dash y html			
from dash.dependencies import Input, Output, State		#
import pathlib									#para la manipulación de sistemas de archivos
import dash

import base64
import datetime
import io

from app import app
import plotly.figure_factory as ff

#Bilbiotecas para clustering
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler 	# Estandarizar
import scipy.cluster.hierarchy as shc 							# Clustering jerárquico
from sklearn.cluster import AgglomerativeClustering 			# Clustering jerárquico

def generar_clusterj(df):
    for x in df.columns:
        if str(df[x].dtype) is 'object':
            df[x] = pd.factorize(df[x])[0]
    matriz = np.array(df[df.columns])
    estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = estandarizar.fit_transform(matriz)   # Se calculan la media y desviación y se escalan los datos
    
    arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))

    return ff.create_dendrogram(MEstandarizada)


layout = html.Div(children =[
    html.H1(children='Algoritmo de Clustering Jerárquico', style={"textAlign": "center"}),
    html.P('Este es un algoritmo utilizado... Se necesita tenere apyori instalado para utilizarlo.'),

    html.H2('Subir los datos a utilizar:'),
    dcc.Upload(
        id='upload-data-clusterj',
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
    html.Div(id='output-data-clusterj'),

    html.H2('Clustering de la tabla'),
    dcc.Graph(id='graf-clusterj'),

], style = {'width': '70%', 'margin-left':'auto', 'margin-right':'auto'})


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

@app.callback(Output('output-data-clusterj', 'children'),
              Output('graf-clusterj', 'figure'),
              Input('upload-data-clusterj', 'contents'),
              State('upload-data-clusterj', 'filename'),
              State('upload-data-clusterj', 'last_modified'))
def update_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    if list_of_contents is not None:
        children, data = parse_contents(list_of_contents, list_of_names, list_of_dates)
        grafica = generar_clusterj(data)

        return children, grafica
