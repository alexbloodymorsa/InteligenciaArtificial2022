from dash import Dash, html, dcc, dash_table			#para componentes core de dash y html			
from dash.dependencies import Input, Output, State	#
import pathlib									#para la manipulación de sistemas de archivos
import dash

import base64
import datetime
import io

from app import app

#Biliotecas para Distancia
import pandas as pd                         # Para la manipulación y análisis de datos
import numpy as np                          # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance

def generate_table(dataframe, x=0):
    return html.Table([
    	html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[x][col]) for col in dataframe.columns
            ])
        ])
    ])


layout = html.Div(children =[
    html.H1(children='Métricas de Distancia', style={"textAlign": "center"}),

    html.P('''Las métricas de distancia son útiles para medir la distancia entre objetos con diferentes criterios.
    Si se desea viajar en una ciudad, no se puede solo ir del punto A al punto B en línea recta, 
    ¡existen edificios! Para ello se tienen diferentes métricas para calcular la distancia. También existen 
    maneras de calcular distancias entre datos, esto nos puede ayudar a saber qué tan parecidos son.'''),

    html.H2('Subir los datos a utilizar:'),
    dcc.Upload(
        id='upload-data-distancia',
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
    html.Div(id='output-data-distancia'),

    html.H2('Ingrese dos índices de filas.'),
    dcc.Input(id='input1', type='number', placeholder='', style={'marginRight':'10px'},
        	min=0, step=1),
    dcc.Input(id='input2', type='number', placeholder='', min=0, step=1),

    html.H2('Primer objeto'),
    html.Table(id='tabla_x'),

    html.H2('Segundo objeto'),
    html.Table(id='tabla_y'),

    html.H2('Distancia Euclidiana'),
    html.P('La distancia Euclidiana mide la distancia entre dos puntos utilizando una línea recta.'),
    html.P(id='euclidiana'),

    html.H2('Distancia Chebyshev'),
    html.P('La distancia de Chebyshev mide la mayor distancia entre las distintas coordenadas.'),
    html.P(id='chebyshev'),

    html.H2('Distancia Manhattan'),
    html.P('La distancia Manhattan mide la distancia como una cuadrícula (o una ciudad).'),
    html.P(id='manhattan'),

    html.H2('Distancia Minkowski'),
    html.P('La distancia de Minkowski genera una curva entre los dos puntos para medir la distancia.'),
    html.P(id='minkowski'),

    dcc.Store(id='df-distancia'),

], style = {'width': '70%', 'margin-left':'auto', 'margin-right':'auto'})

@app.callback(
    Output('euclidiana', 'children'),
    Output('chebyshev', 'children'),
    Output('manhattan', 'children'),
	Output('minkowski', 'children'),
	Output('tabla_x', 'children'),
	Output('tabla_y', 'children'),
	Input('input1', 'value'),
    Input('input2', 'value'),
    Input('df-distancia', 'data')
)
def update_output_div(input1, input2, dict):
    if input2 is None:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(dict)

    for x in df.columns:
        if str(df[x].dtype) is 'object':
            df[x] = pd.factorize(df[x])[0]

    Objeto1 = df.iloc[input1]
    Objeto2 = df.iloc[input2]

    dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
    dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
    dstManhattan = distance.cityblock(Objeto1,Objeto2)
    dstMinkowski = distance.minkowski(Objeto1,Objeto2, p=1.5)

    strEuclidiana = 'Resultado: {}'.format(dstEuclidiana)
    strChebyshev = 'Resultado: {}'.format(dstChebyshev)
    strManhattan = 'Resultado: {}'.format(dstManhattan)
    strMinkowski = 'Resultado: {}'.format(dstMinkowski)

    tabla1 = generate_table(df, input1)

    tabla2 = generate_table(df, input2)


    return strEuclidiana, strChebyshev, strManhattan, strMinkowski, tabla1, tabla2



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

@app.callback(Output('output-data-distancia', 'children'),
              Output('df-distancia', 'data'),
              Input('upload-data-distancia', 'contents'),
              State('upload-data-distancia', 'filename'),
              State('upload-data-distancia', 'last_modified'))
def update_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    if list_of_contents is not None:
        children, data = parse_contents(list_of_contents, list_of_names, list_of_dates)
        opciones = data.columns

        return children, data.to_dict()
