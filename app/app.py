import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import jupyter_plotly_dash
import pandas as pd
import plotly.graph_objs as go


app = dash.Dash(__name__)
server = app.server
app.title = "Automobile Crashes in Chicago"


mapbox_token = os.environ.get('MAPBOXTOKEN')

dtypes = {'TRAFFIC_CONTROL_DEVICE': 'category', 'DEVICE_CONDITION': 'category',
         'WEATHER_CONDITION': 'category', 'LIGHTING_CONDITION': 'category', 'FIRST_CRASH_TYPE': 'category',
         'TRAFFICWAY_TYPE': 'category', 'ROADWAY_SURFACE_COND': 'category', 'ROAD_DEFECT': 'category',
         'REPORT_TYPE': 'category', 'CRASH_TYPE': 'category', 'INTERSECTION_RELATED_I': 'category',
         'NOT_RIGHT_OF_WAY_I': 'category', 'HIT_AND_RUN_I': 'category', 'DAMAGE': 'category', 
         'PRIM_CONTRIBUTORY_CAUSE': 'category', 'SEC_CONTRIBUTORY_CAUSE': 'category', 'STREET_DIRECTION': 'category',
         'STREET_NAME': 'category', 'PHOTOS_TAKEN_I': 'category', 'STATEMENTS_TAKEN_I': 'category',
         'DOORING_I': 'category', 'WORK_ZONE_I': 'category', 'WORK_ZONE_TYPE': 'category', 'WORKERS_PRESENT_I': 'category',
         'MOST_SEVERE_INJURY': 'category'}


crashes = pd.read_csv('../data/TrafficCrashesChicago.csv',
                      parse_dates=['CRASH_DATE_EST_I', 'CRASH_DATE',
                                   'DATE_POLICE_NOTIFIED'],
                      dtype=dtypes,
                      nrows = 500)



app.layout = html.Div(
    id="root",
    children=[
        html.Div(
            id="header",
        )
    ]
)



if __name__ == '__main__':
    app.run_server(debug=True)
