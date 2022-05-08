from dash import Dash, html, dcc				#para componentes core de dash y html			
from dash.dependencies import Input, Output		#
import pathlib									#para la manipulación de sistemas de archivos

from app import app
import plotly.express as px

#Bilbiotecas para clustering
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler 	# Estandarizar
from sklearn.cluster import KMeans 								# Kmeans
from sklearn.metrics import pairwise_distances_argmin_min 		# Métricas Kmeans
from kneed import KneeLocator 									# Método del codo
import scipy.cluster.hierarchy as shc 							# Clustering jerárquico
from sklearn.cluster import AgglomerativeClustering 			# Clustering jerárquico

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
df = pd.read_csv(DATA_PATH.joinpath("Hipoteca.csv"))

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

def particional():
    MatrizHipoteca = np.array(df[['ingresos', 'gastos_comunes', 'pago_coche', 'gastos_otros', 'ahorros', 'vivienda', 'estado_civil', 'hijos', 'trabajo']])
    estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = estandarizar.fit_transform(MatrizHipoteca)   # Se calculan la media y desviación y se escalan los datos
    SSE = []
    for i in range(2, 12):
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(MEstandarizada)
        SSE.append(km.inertia_)
    kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
    MParticional = KMeans(n_clusters=4, random_state=0).fit(MEstandarizada)
    MParticional.predict(MEstandarizada)
    return dcc.Graph(
        figure = px.scatter(x=MEstandarizada[:,0], y=MEstandarizada[:,1], color=MParticional.labels_)
        )


layout = html.Div(children =[
    html.H1(children='Algoritmos de Clustering', style={"textAlign": "center"}),
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

    html.H2('Tabla generada'),
    generate_table(df),

    html.H2('Número de clústeres:'),
    dcc.Input(id='input-cluster1', type='number', placeholder='numero clústeres',min=0, max=12,step=1),

    html.H2('Clustering de la tabla'),
    particional(),


])
