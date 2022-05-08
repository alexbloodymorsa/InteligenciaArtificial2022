from dash import Dash, html, dcc				#para componentes core de dash y html			
from dash.dependencies import Input, Output		#
import pathlib									#para la manipulación de sistemas de archivos

from app import app

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
#df = pd.read_csv(DATA_PATH.joinpath("movies.csv"))

layout = html.Div(children =[
    html.H1(children='Bienvenido', style={"textAlign": "center"}),
    html.Div(id='home')

])