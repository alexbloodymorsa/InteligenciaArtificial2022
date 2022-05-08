from dash import Dash, html, dcc				#para componentes core de dash y html			
from dash.dependencies import Input, Output		#
import pathlib									#para la manipulación de sistemas de archivos
import plotly.express as px                     #plotly
import dash

from app import app

#Bibliotecas para apriori
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
df = pd.read_csv(DATA_PATH.joinpath("movies.csv"), header=None)

def reglas_negocio(ResultadosC3):
    resultado = ""
    for item in ResultadosC3:
        #El primer índice de la lista
        Emparejar = item[0]
        items = [x for x in Emparejar]
        resultado += "\nRegla: " + str(item[0])

        #El segundo índice de la lista
        resultado += "\nSoporte: " + str(item[1])

        #El tercer índice de la lista
        resultado += "\nConfianza: " + str(item[2][0][2])
        resultado += "\nLift: " + str(item[2][0][3])

    return resultado

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(max_rows)
        ])
    ])

def generar_frecuencia(dataframe):
    Transacciones = df.values.reshape(-1).tolist()
    ListaM = pd.DataFrame(Transacciones)
    ListaM['Frecuencia'] = 0
    ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True)
    ListaM = ListaM.rename(columns={0 : 'Item'})
    return dcc.Graph(
            figure = px.bar(ListaM, x='Item', y='Frecuencia')
        )

layout = html.Div(children =[
    html.H1(children='Reglas de asociación (Algoritmo Apriori)', style={"textAlign": "center"}),

    html.P('Este es un algoritmo utilizado... Se necesita tenere apyori instalado para utilizarlo.'),

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

    html.H2('Datos a utilizar'),
    generate_table(df),

    html.H2('Frecuencia de los datos'),
    generar_frecuencia(df),

    html.H2('Parámetros de las reglas de asociación:'),
    dcc.Input(id='input-apr1', type='number', placeholder='soporte',min=0, max=1,step=.01),
    dcc.Input(id='input-apr2', type='number', placeholder='confianza', min=0, max=2, step=.1),
    dcc.Input(id='input-apr3', type='number', placeholder='lift',min=0, max=2 ,step=.1),

    html.H2('Reglas de asociación'),
    html.P(id='output'),

])

@app.callback(
    Output('output', 'children'),
    Input('input-apr1', 'value'),
    Input('input-apr2', 'value'),
    Input('input-apr3', 'value'),
)
def update_output(soporte, confianza, lift):
    if soporte is None:
        raise dash.exceptions.PreventUpdate

    MoviesLista = df.stack().groupby(level=0).apply(list).tolist()
    ReglasC3 = apriori(MoviesLista, min_support=soporte, min_confidence=confianza, min_lift=lift)


    return reglas_negocio(ReglasC3)