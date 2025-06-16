#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

weather = pd.read_csv("C:/Users/Admin/Downloads/bengaluru_weather_2015_2020.csv")
aggregated = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/mlmini/kaggle dataset/aggregated.csv")
routes = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/mlmini/kaggle dataset/routes.csv")
stops = pd.read_csv("C:/Users/Admin/OneDrive/Desktop/mlmini/kaggle dataset/stops.csv")

weather['Date'] = pd.to_datetime(weather['Date'])
weather['date_only'] = weather['Date'].dt.date
routes['date_only'] = pd.to_datetime('2015-01-01').date()
routes['date_only'] = pd.to_datetime(routes['date_only']).dt.date
routes['route_length'] = np.random.randint(5, 30, size=len(routes))

weather['hour'] = weather['Date'].dt.hour
weather['is_peak'] = weather['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
weather['day_of_week'] = weather['Date'].dt.dayofweek
weather['temp_avg'] = (weather['Temp Max'] + weather['Temp Min']) / 2
weather['is_rainy'] = weather['Rain'].apply(lambda x: 1 if x > 0 else 0)

trips = pd.DataFrame({
    'trip_id': range(1, 1001),
    'route_id': np.random.choice(routes['id'], 1000),
    'date_only': pd.date_range('2015-01-01', '2020-12-31').date.tolist()[:1000],
    'stop_sequence': np.random.randint(1, 20, 1000)
})

trips['date_only'] = pd.to_datetime(trips['date_only']).dt.date

merged_data = pd.merge(trips, routes[['id', 'route_length']], left_on='route_id', right_on='id', how='left')
merged_data = pd.merge(merged_data, weather, on='date_only', how='left')

np.random.seed(42)
delay_conditions = (merged_data['Rain'] > 5) | (merged_data['hour'].isin([8, 9, 17, 18]))
merged_data['delay'] = np.where(delay_conditions, 1, 0)

features = ['hour', 'is_peak', 'Rain', 'temp_avg', 'route_length', 'stop_sequence']
X = merged_data[features]
y = merged_data['delay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight={0:1, 1:2}, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred))

plt.figure(figsize=(10, 5))
plt.barh(features, model.feature_importances_)
plt.title("Feature Importance for Delay Prediction")
plt.show()

new_data = pd.DataFrame({
    'hour': [8, 14],
    'is_peak': [1, 0],
    'Rain': [12, 0],
    'temp_avg': [25, 30],
    'route_length': [20, 10],
    'stop_sequence': [5, 3]
})
print("\nPredicted Delays for New Data:", model.predict(new_data))


import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "BMTC Bus Delay Predictor"

routes = [
    {'label': '500D - Majestic to Electronic City', 'value': '500D'},
    {'label': '335E - KR Market to Whitefield', 'value': '335E'},
    {'label': '401G - Kempegowda Station to Banashankari', 'value': '401G'},
    {'label': '600A - Shivajinagar to Yelahanka', 'value': '600A'}
]

weather_conditions = {
    'sunny': {'temp': (25, 35), 'rain': 0, 'icon': '☀️'},
    'partly_cloudy': {'temp': (22, 30), 'rain': 0, 'icon': '⛅'},
    'rainy': {'temp': (20, 27), 'rain': (5, 15), 'icon': '🌧️'},
    'heavy_rain': {'temp': (18, 25), 'rain': (15, 30), 'icon': '⛈️'}
}

try:
    model = joblib.load('bmtc_delay_model.pkl')
except:
    np.random.seed(42)
    X = pd.DataFrame({
        'hour': np.random.randint(0, 24, 1000),
        'is_peak': np.random.choice([0, 1], 1000, p=[0.7, 0.3]),
        'rain': np.random.uniform(0, 30, 1000),
        'temp': np.random.uniform(15, 35, 1000)
    })
    y = ((X['hour'].isin([7,8,9,16,17,18])) & (X['rain'] > 5)).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'bmtc_delay_model.pkl')

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("BMTC Bus Delay Prediction", className="text-center my-4"))),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("Select Your Journey", className="mb-3"),
                html.Label("Select Route", className="form-label"),
                dcc.Dropdown(id='route-dropdown', options=routes, value='500D', className="mb-3"),
                html.Label("Select Time", className="form-label"),
                dcc.Dropdown(
                    id='hour-dropdown',
                    options=[{'label': f"{h%12 or 12}{' AM' if h < 12 else ' PM'}", 'value': h} for h in range(24)],
                    value=8,
                    className="mb-3"
                ),
                html.Label("Current Weather", className="form-label"),
                dcc.RadioItems(
                    id='weather-radio',
                    options=[{'label': f" {weather_conditions[w]['icon']} {w.replace('_', ' ').title()}", 'value': w} for w in weather_conditions],
                    value='sunny',
                    className="mb-4"
                ),
                dbc.Button("Check Delay Prediction", id='predict-button', color="primary", className="w-100 mb-4"),
                dcc.Store(id='intermediate-store')
            ], className="p-4 border rounded shadow-sm")
        ], md=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Results", className="fw-bold"),
                dbc.CardBody([
                    html.Div(id='prediction-output', className="mb-4"),
                    html.Hr(),
                    html.Div(id='route-info')
                ])
            ], className="shadow-sm mb-4"),
            dbc.Card([
                dbc.CardHeader("Route Map"),
                dbc.CardBody([dcc.Graph(id='route-map', config={'displayModeBar': False})])
            ], className="shadow-sm")
        ], md=8)
    ])
], fluid=True)

@app.callback(
    Output('intermediate-store', 'data'),
    [Input('predict-button', 'n_clicks')],
    [State('route-dropdown', 'value'),
     State('hour-dropdown', 'value'),
     State('weather-radio', 'value')]
)
def calculate_prediction(n_clicks, route, hour, weather):
    if n_clicks is None:
        return dash.no_update
    weather_data = weather_conditions[weather]
    temp = np.random.uniform(*weather_data['temp'])
    rain = weather_data['rain'] if isinstance(weather_data['rain'], int) else np.random.uniform(*weather_data['rain'])
    input_data = pd.DataFrame({
        'hour': [hour],
        'is_peak': [1 if hour in [7,8,9,16,17,18] else 0],
        'rain': [rain],
        'temp': [temp]
    })
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    time_str = f"{hour%12 or 12}{' AM' if hour < 12 else ' PM'}"
    data = {
        'route': route,
        'time_str': time_str,
        'hour': hour,
        'weather': weather,
        'temp': temp,
        'rain': rain,
        'prediction': int(prediction),
        'probability': float(proba[1] if prediction else proba[0]),
        'is_peak': int(input_data['is_peak'][0])
    }
    return json.dumps(data)

@app.callback(
    [Output('prediction-output', 'children'),
     Output('route-info', 'children')],
    [Input('intermediate-store', 'data')]
)
def update_output(json_data):
    if not json_data:
        return "", ""
    try:
        data = json.loads(json_data)
    except:
        return "", ""
    if data['prediction']:
        alert = dbc.Alert([
            html.Div([
                html.Span("🚨", style={'font-size': '2rem', 'margin-right': '10px'}),
                html.Div([
                    html.H4("Delay Expected!", className="alert-heading mb-2"),
                    html.P(f"Probability: {data['probability']*100:.1f}%", className="mb-1"),
                    html.P(f"Route {data['route']} at {data['time_str']}", className="mb-1")
                ])
            ], style={'display': 'flex', 'align-items': 'center'}),
            html.Hr(),
            html.H5("Primary Delay Factors:", className="mt-3"),
            html.Ul([
                html.Li(f"{weather_conditions[data['weather']]['icon']} Weather: {data['weather'].replace('_', ' ').title()}"),
                html.Li(f"🌡️ Temperature: {data['temp']:.1f}°C"),
                html.Li(f"💧 Rainfall: {data['rain']:.1f} mm"),
                html.Li(f"🕒 {'Peak Hour' if data['is_peak'] else 'Off-Peak Hour'}")
            ], className="mt-2")
        ], color="danger", className="p-3")
    else:
        alert = dbc.Alert([
            html.Div([
                html.Span("✅", style={'font-size': '2rem', 'margin-right': '10px'}),
                html.Div([
                    html.H4("No Delay Expected", className="alert-heading mb-2"),
                    html.P(f"Probability: {data['probability']*100:.1f}%", className="mb-1"),
                    html.P(f"Route {data['route']} at {data['time_str']}", className="mb-1")
                ])
            ], style={'display': 'flex', 'align-items': 'center'}),
            html.Hr(),
            html.H5("Current Conditions:", className="mt-3"),
            html.Ul([
                html.Li(f"{weather_conditions[data['weather']]['icon']} Weather: {data['weather'].replace('_', ' ').title()}"),
                html.Li(f"🌡️ Temperature: {data['temp']:.1f}°C"),
                html.Li(f"💧 Rainfall: {data['rain']:.1f} mm"),
                html.Li(f"🕒 {'Peak Hour' if data['is_peak'] else 'Off-Peak Hour'}")
            ], className="mt-2")
        ], color="success", className="p-3")

    route_info = [
        html.H4(f"Route {data['route']} Information", className="mb-3"),
        dbc.Row([
            dbc.Col(html.Div([html.P("⏱️ Departure Time", className="mb-1 text-muted"), html.H4(data['time_str'])], className="p-3 border rounded text-center"), md=4),
            dbc.Col(html.Div([html.P("🌡️ Temperature", className="mb-1 text-muted"), html.H4(f"{data['temp']:.1f}°C")], className="p-3 border rounded text-center"), md=4),
            dbc.Col(html.Div([html.P("💧 Rainfall", className="mb-1 text-muted"), html.H4(f"{data['rain']:.1f} mm")], className="p-3 border rounded text-center"), md=4)
        ])
    ]
    return alert, route_info

@app.callback(
    Output('route-map', 'figure'),
    [Input('route-dropdown', 'value')]
)
def update_map(route):
    routes_coords = {
        '500D': {'lat': [12.97, 12.93, 12.90], 'lon': [77.59, 77.62, 77.65]},
        '335E': {'lat': [12.96, 12.98, 13.00], 'lon': [77.57, 77.60, 77.63]},
        '401G': {'lat': [12.98, 12.96, 12.94], 'lon': [77.58, 77.56, 77.54]},
        '600A': {'lat': [12.97, 13.00, 13.03], 'lon': [77.60, 77.62, 77.65]}
    }
    fig = px.line_mapbox(lat=routes_coords[route]['lat'], lon=routes_coords[route]['lon'], zoom=11, height=400)
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0}, hovermode=False)
    fig.add_trace(go.Scattermapbox(lat=[routes_coords[route]['lat'][0]], lon=[routes_coords[route]['lon'][0]], mode='markers', marker=dict(size=14, color='green'), text=['Start'], hoverinfo='text'))
    fig.add_trace(go.Scattermapbox(lat=[routes_coords[route]['lat'][-1]], lon=[routes_coords[route]['lon'][-1]], mode='markers', marker=dict(size=14, color='red'), text=['End'], hoverinfo='text'))
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8058)


# In[ ]:




