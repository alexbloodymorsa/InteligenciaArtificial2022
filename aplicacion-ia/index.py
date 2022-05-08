from dash import Dash, html, dcc
from dash.dependencies import Input, Output 

# Connect to main app.py file
from app import app
from app import server

# Connect to your app pages
from apps import apriori, cluster, distancia, home


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
    	dcc.Link('Home |', href='/apps/home'),
        dcc.Link('Asociación |', href='/apps/apriori'),
        dcc.Link('Clustering |', href='/apps/cluster'),
        dcc.Link('Mediciones Distancia |', href='/apps/distancia'),
    ], className="row"),
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
    else:
        return "404 Page Error! Please choose a link"


if __name__ == '__main__':
    app.run_server(debug=True)