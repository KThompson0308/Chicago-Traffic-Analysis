import os

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc 
import dash_html_components as html

from plotly import graph_objs as go
from plotly.graph_objs import *
from dash.dependencies import Input, Output, State, Event


app = dash.Dash(__name__)
server = app.server
app.title = "Chicago Car Crashes"


mapbox_token = os.environ.get('MAPBOX_TOKEN')




if __name__ == '__main__':
    app.run_server(debug=True)


