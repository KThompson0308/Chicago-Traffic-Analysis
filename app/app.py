import os

import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, Event
from jupyter_plotly_dash import JupyterDash

app = JupyterDash('Chicago Car Crashes')
mapbox_token = os.environ.get('MAPBOXTOKEN')