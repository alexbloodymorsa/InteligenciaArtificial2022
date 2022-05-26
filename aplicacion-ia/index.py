from dash import Dash, html, dcc
from dash.dependencies import Input, Output 
import dash_bootstrap_components as dbc

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import apriori, cluster, distancia, home, clusterj, regresion, arbol, bosque


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink('Asociación', href='/apps/apriori')),
            dbc.NavItem(dbc.NavLink('Mediciones Distancia', href='/apps/distancia')),
            dbc.NavItem(dbc.NavLink('Clustering Particional', href='/apps/cluster')),
            #dbc.NavItem(dbc.NavLink('Clustering Jerárquico', href='/apps/clusterj')),
            dbc.NavItem(dbc.NavLink('Regresión Logística', href='/apps/regresion')),
            dbc.NavItem(dbc.NavLink('Árbol de Decisión', href='/apps/arbol')),
            dbc.NavItem(dbc.NavLink('Bosque Aleatorio', href='/apps/bosque')),
        ],
        brand="Jarreth",
        brand_href="/apps/home",
        color="primary",
        dark=True
    ),
    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/home':
        return home.layout
    if pathname == '/apps/apriori':
        return apriori.layout
    if pathname == '/apps/cluster':
        return cluster.layout
    if pathname == '/apps/distancia':
        return distancia.layout
    if pathname == '/apps/clusterj':
        return clusterj.layout
    if pathname == '/apps/regresion':
        return regresion.layout
    if pathname == '/apps/arbol':
        return arbol.layout
    if pathname == '/apps/bosque':
        return bosque.layout
    else:
        return home.layout


if __name__ == '__main__':
    app.run_server(debug=True)