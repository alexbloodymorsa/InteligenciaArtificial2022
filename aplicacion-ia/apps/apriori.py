from dash import Dash, html, dcc, dash_table            #para componentes core de dash y html           
from dash.dependencies import Input, Output, State  #
import pathlib                                  #para la manipulación de sistemas de archivos
import plotly.express as px                     #plotly
import dash

import dash_bootstrap_components as dbc

import base64
import datetime
import io

from app import app

#Bibliotecas para apriori
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from apyori import apriori

#EXCEPCIONES

def reglas_negocio(Resultados):
    if Resultados == '':
        return html.P('No hay reglas de negocio')
    
    tarjetas = []

    i = 0
    for item in Resultados:
        resultado = []
        #El primer índice de la lista
        Emparejar = item[0]
        items = [x for x in Emparejar]
        peli = list(item[0])
        cadena = 'Regla ' + str(i + 1) + ': ' + str(peli[0]) + ' y ' + str(peli[1])
        resultado.append(html.H2(cadena))
        i += 1

        #El segundo índice de la lista
        cadena = 'Soporte:'  + str(item[1])
        resultado.append(html.P(cadena))

        #El tercer índice de la lista
        cadena = 'Confianza: ' + str(item[2][0][2])
        resultado.append(html.P(cadena))
        cadena = 'Lift: ' + str(item[2][0][3])
        resultado.append(html.P(cadena))

        tarjeta = dbc.Card(dbc.CardBody(resultado), style={"width": "30rem", 'float':'left', 'display':'inline'})

        tarjetas.append(tarjeta)

    return tarjetas

def generar_frecuencia(df):
    Transacciones = df.values.reshape(-1).tolist()
    ListaM = pd.DataFrame(Transacciones)
    ListaM['Frecuencia'] = 0
    ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True)
    ListaM = ListaM.rename(columns={0 : 'Item'})
    fig = px.bar(ListaM, x='Item', y='Frecuencia')
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    return fig

layout = html.Div(children =[
    html.H1(children='Reglas de asociación (Algoritmo Apriori)', style={"textAlign": "center"}),

    html.P('''¿Te has preguntado como Amazon, Netflix u otros servicios te recomiendan productos? 
        Pues la respuesta está en las reglas de recomendación. Las reglas de recomendación se generan a 
        partir de la información de lo que consume un usuario. El algoritmo apriori genera reglas de 
        recomendación utilizando las veces que aparecen los datos que se consumen (frecuencia), el soporte, 
        la confianza y la elevación. Se necesita tenere apyori instalado para utilizarlo.'''),

    html.H2('Subir los datos a utilizar:'),
    dcc.Upload(
        id='upload-data-apriori',
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
    html.Div(id='output-data-apriori'),

    html.H2('Frecuencia de los datos'),
    dcc.Graph(id='graf-frecuencia'),

    html.H2('Parámetros de las reglas de asociación:'),

    html.P('El soporte indica qué tan importante es una regla en un conjunto.'),
    dbc.Input(id='input-apr1', type='number', placeholder='soporte',min=0, max=1,step=.01),
    html.P('La confianza indica qué tan fiable es una regla. '),
    dbc.Input(id='input-apr2', type='number', placeholder='confianza', min=0, max=2, step=.1),
    html.P('La elevación indica el aumento de la probabilidad del primer producto al segundo.'),
    dbc.Input(id='input-apr3', type='number', placeholder='elevación',min=0, max=2 ,step=.1),

    html.H2('Reglas de asociación'),
    html.Div(id='output'),

    dcc.Store(id='df-apriori'),

], style = {'width': '70%', 'margin-left':'auto', 'margin-right':'auto'})

@app.callback(
    Output('output', 'children'),
    Input('input-apr1', 'value'),
    Input('input-apr2', 'value'),
    Input('input-apr3', 'value'),
    Input('df-apriori', 'data')
)
def update_output(soporte, confianza, lift, dict):
    if lift is None:
        raise dash.exceptions.PreventUpdate

    df = pd.DataFrame(dict)
    lista = df.stack().groupby(level=0).apply(list).tolist()
    reglas = apriori(lista, min_support=soporte, min_confidence=confianza, min_lift=lift)

    return reglas_negocio(reglas)

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
            virtualization=True,
            fixed_rows={'headers': True},
            style_cell={'minWidth': 95, 'width': 95, 'maxWidth': 300},
            style_table={'height': 300}
        )
    ]), df

@app.callback(Output('output-data-apriori', 'children'),
              Output('graf-frecuencia', 'figure'),
              Output('df-apriori', 'data'),
              Input('upload-data-apriori', 'contents'),
              State('upload-data-apriori', 'filename'),
              State('upload-data-apriori', 'last_modified'))
def update_file(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    if list_of_contents is not None:
        children, data = parse_contents(list_of_contents, list_of_names, list_of_dates)
        grafica = generar_frecuencia(data)

        return children, grafica, data.to_dict()