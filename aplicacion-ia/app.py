from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.UNITED], suppress_callback_exceptions=True)

server = app.server