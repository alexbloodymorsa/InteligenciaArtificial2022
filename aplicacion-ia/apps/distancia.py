from dash import Dash, html, dcc				#para componentes core de dash y html			
from dash.dependencies import Input, Output		#
import pathlib									#para la manipulación de sistemas de archivos
import dash

from app import app

#Biliotecas para Distancia
import pandas as pd                         # Para la manipulación y análisis de datos
import numpy as np                          # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datos
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
df = pd.read_csv(DATA_PATH.joinpath("Hipoteca.csv"))

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

    html.P('Este es un algoritmo utilizado...'),

    html.H2('Subir los datos a utilizar:'),
    dcc.Upload(
        id='upload-data',
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

    html.H2('Ingrese dos índices de filas.'),
    dcc.Input(id='input1', type='number', placeholder='', style={'marginRight':'10px'},
        	min=0, max=df[df.columns[0]].count(),step=1),
    dcc.Input(id='input2', type='number', placeholder='', min=0, max=df[df.columns[0]].count(), step=1),

    html.H2('Primer objeto'),
    html.Table(id='tabla_x'),

    html.H2('Segundo objeto'),
    html.Table(id='tabla_y'),

    html.H2('Distancia Euclidiana'),
    html.P(id='euclidiana'),

    html.H2('Distancia Chebyshev'),
    html.P(id='chebyshev'),

    html.H2('Distancia Manhattan'),
    html.P(id='manhattan'),

    html.H2('Distancia Minkowski'),
    html.P(id='minkowski'),

])

@app.callback(
    Output('euclidiana', 'children'),
    Output('chebyshev', 'children'),
    Output('manhattan', 'children'),
	Output('minkowski', 'children'),
	Output('tabla_x', 'children'),
	Output('tabla_y', 'children'),
	Input('input1', 'value'),
    Input('input2', 'value'),
)
def update_output_div(input1, input2):
	if input2 is None:
		raise dash.exceptions.PreventUpdate

	Objeto1 = df.iloc[input1]
	Objeto2 = df.iloc[input2]
	dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
	dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
	dstManhattan = distance.cityblock(Objeto1,Objeto2)
	dstMinkowski = distance.minkowski(Objeto1,Objeto2, p=1.5)


	return 'Resultado: {}'.format(dstEuclidiana), 'Resultado: {}'.format(dstChebyshev), 'Resultado: {}'.format(dstManhattan), 'Resultado: {}'.format(dstMinkowski),generate_table(df, input1), generate_table(df, input2)

