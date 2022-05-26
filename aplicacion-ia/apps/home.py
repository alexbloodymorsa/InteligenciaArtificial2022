from dash import Dash, html, dcc				#para componentes core de dash y html			
from dash.dependencies import Input, Output		#
import pathlib									#para la manipulación de sistemas de archivos

import dash_bootstrap_components as dbc

from app import app

layout = html.Div(children =[

	html.H1('Bienvenido a Jarreth', style={"textAlign": "center"}),

	dbc.Card([

    	dbc.CardBody([

    	html.H2('Jarreth'),

	    html.H3('''es un programa hecho con Dash, capaz de resolver problemas de inteligencia 
	    	artificial, específicamente de aprendizaje automático, utilizando datos que proporciona 
	    	el usuario. Jarreth es ideal como introducción a los algoritmos más utilizados en el aprendizaje 
	    	automático. Permite al usuario modificar parámetros de cada uno de los modelos y entender 
	    	cada uno de los algoritmos con la información que contiene.
	    	'''),

	    dcc.Markdown('''#### Componentes de Jarreth:'''),

	    dbc.NavItem(dbc.NavLink('Asociación', href='/apps/apriori')),
	    dbc.NavItem(dbc.NavLink('Mediciones Distancia', href='/apps/distancia')),
	    dbc.NavItem(dbc.NavLink('Clustering Particional', href='/apps/cluster')),
	    #dbc.NavItem(dbc.NavLink('Clustering Jerárquico', href='/apps/clusterj')),
	    dbc.NavItem(dbc.NavLink('Regresión Logística', href='/apps/regresion')),
	    dbc.NavItem(dbc.NavLink('Árbol de Decisión', href='/apps/arbol')),
	    dbc.NavItem(dbc.NavLink('Bosque Aleatorio', href='/apps/bosque'))]),
    ]),

	html.H4('Creado por: Alejandro Barreiro Valdez'),

    html.Div(id='home')

], style = {'width': '70%', 'margin-left':'auto', 'margin-right':'auto'})