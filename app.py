#!/usr/bin/python
# coding: utf-8

# Dash
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Librerías estándar 
import os
import pandas as pd
import numpy as np
import scipy.signal as sgnl
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from copy import deepcopy
from collections import defaultdict

# Clase CovidData
from covid import CovidData

# Fecha colombiana de reporte
current_date = pd.to_datetime((datetime.now(timezone.utc) - timedelta(hours=5)).date())
data_path = os.path.join('data', 'covid-' + str(datetime.date(current_date)) + '.csv')
necessary_columns = [
                        'id_de_caso', 'fecha_de_notificaci_n', 'ciudad_municipio_nom', 'departamento_nom', 
                        'recuperado', 'edad', 'sexo', 'fuente_tipo_contagio', 'estado', 'pais_viajo_1_nom', 
                        'fecha_inicio_sintomas','fecha_muerte', 'fecha_diagnostico', 'fecha_recuperado', 
                        'fecha_reporte_web' 
                    ]

# Obtiene información de covid Colombia
try:
    covid_data = pd.read_csv(data_path)
except FileNotFoundError:
    covid_data = pd.read_json('https://www.datos.gov.co/resource/gt2j-8ykr.json?$limit=10000000')
    covid_data = covid_data[necessary_columns]
    covid_data.to_csv(data_path)

# Instancia la información en la clase CovidData
cd = CovidData(covid_data)
cd.preprocessing_data()

covid_data = cd.covid_data
d_hat = cd.d_hat
w_hat = cd.w_hat

# Funciones auxiliares

def thousand_sep(n: int) -> str:
    return f'{n:,}'

mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
app = dash.Dash(__name__, external_scripts=[mathjax])
app.title = 'Rt Colombia'
server = app.server

graph_config = {
    'modeBarButtonsToRemove': [
        'autoScale2d', 'select2d', 'zoom2d',
        'pan2d', 'toggleSpikelines',
        'hoverCompareCartesian',
        'zoomOut2d', 'zoomIn2d',
        'hoverClosestCartesian',
        'resetScale2d', 'lasso2d',
    ]
}

layout_graph = {
    'legend': {
        'orientation': 'h',
        'x': 0.5,
        'xanchor': 'center'
    },
    'margin': {'t': 40, 'r': 40, 'l': 40, 'b': 60},
    'hovermode': 'closest',
    'plot_bgcolor': '#fff',
    'yaxis': {
        'showgrid': True,
        'gridcolor': 'whitesmoke'
    },
    'xaxis': {
        'showgrid': True,
        'gridcolor': 'whitesmoke' 
    },
}

app.layout = html.Div([
    html.H2(
        dcc.Markdown('COVID-19 Colombia: cálculo de $ R_{t} $ en tiempo real'),
        className='title',
    ),
    html.H6(
        dcc.Markdown(f'Información obtenida de la [base de datos](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data) recopilada por el Instituto Nacional de Salud'),
        style={'text-align': 'center'}
    ),
    html.Div([
        html.Div([
            html.P('Seleccione un rango de fechas', className='control_label'),
            dcc.DatePickerRange(
                id='fecha',
                min_date_allowed=covid_data['fecha_sintomas'].min(),
                max_date_allowed=current_date,
                initial_visible_month=current_date,
                end_date=current_date,
                start_date=datetime(2020, 4, 1),
                display_format='DD-MMM-YYYY',
                first_day_of_week=1,
            ),
            html.P('Filtro por departamentos', className='control_label'),
            dcc.Dropdown(
                id='departamento',
                options=[{'label': dpto, 'value': dpto} for dpto in np.sort(covid_data['departamento'].unique())],
                placeholder='Buscar departamento o D. E.',
                multi=True,
            ),
            html.P('Filtro por ciudades', className='control_label'),
            dcc.Dropdown(
                id='municipio',
                options=[{'label': city, 'value': city} for city in np.sort(covid_data['municipio'].unique())],
                placeholder='Buscar ciudad o municipio',
                multi=True,
            ),
            ],
            style={'width': '25%'},
            className='pretty_container',
        ),
        html.Div([
            html.Div(
                dcc.Graph(
                    id='rt_graph',
                    config=graph_config,
                    figure=go.Figure(
                        layout=layout_graph,
                    )
                ),
                className='pretty_container',
            ),
        ],
            style={'width': '75%'},
        ),
    ],
    className='row',
    style={'display': 'flex'},
    ),
    dcc.Markdown(f'**IMPORTANTE:** El reporte de infectados presenta un tiempo medio de retraso de {w_hat} \
        días, por lo que la interpretación de los valores de Rt para los últimos 15 días debe ser hecha con precaución.'),
    html.Div([    
        html.Div(
            dcc.Graph(
                id='daily_infectados',
                config=graph_config,
                figure=go.Figure(
                    layout=layout_graph,
                )
            ),
            style={'width': '50%'},
            className='pretty_container',
        ),
        html.Div(
            dcc.Graph(
                id='daily_deaths',
                config=graph_config,
                figure=go.Figure(
                    layout=layout_graph,
                )
            ),
            style={'width': '50%'},
            className='pretty_container',
        ),
    ],
        className='row',
        style={'display': 'flex'},
    ),
    html.Div([    
        html.Div(
            dcc.Graph(
                id='cum_infectados',
                config=graph_config,
                figure=go.Figure(
                    layout=layout_graph,
                )
            ),
            style={'width': '50%'},
            className='pretty_container',
        ),
        html.Div(
            dcc.Graph(
                id='cum_deaths',
                config=graph_config,
                figure=go.Figure(
                    layout=layout_graph,
                )
            ),
            style={'width': '50%'},
            className='pretty_container',
        ),
    ],
        className='row',
        style={'display': 'flex'},
    ),
    html.Div([
        html.Div(
            dcc.Graph(
                id='reported',
                config=graph_config,
                figure=go.Figure(
                    layout=layout_graph
                )
            ),
            style={'width': '50%'},
            className='pretty_container',
        ),
        html.Div(
            dcc.Graph(
                id='status_infectados',
                config=graph_config,
                figure=go.Figure(
                    layout=layout_graph
                )
            ),
            style={'width': '50%'},
            className='pretty_container',
        ),
    ],
        className='row',
        style={'display': 'flex'},
    ),
    html.Div(
        dash_table.DataTable(
            id='days_table',
            style_as_list_view=True,
            style_cell={'padding': '5px'},
            style_header={
                'backgroundColor': 'white',
                'fontWeight': 'bold'
            },
        ),
        className='pretty_container',
    ),
    dcc.Markdown('Elaborado por:'),
    dcc.Markdown('- Jairo Díaz, División de Ciencias Básicas, Universidad del Norte - Barranquilla'),
    dcc.Markdown('- Jairo Espinosa, Facultad de Minas, Universidad Nacional de Colombia - Medellín'),
    dcc.Markdown('- Héctor López - Barranquilla'),
    dcc.Markdown('- Bernardo Uribe, División de Ciencias Básicas, Universidad del Norte - Barranquilla'),
    dcc.Markdown('La información completa de este proyecto se puede consultar en :'),
    dcc.Markdown('http://www.rtcolombia.com'),
    dcc.Markdown('Sociedad Colombiana de Matemáticas'),
    ],
className='container',
style={'display': 'flex', 'flex-direction': 'column'},
)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
@app.callback(
    [
        Output('rt_graph', 'figure'),
        Output('reported', 'figure'),
        Output('daily_infectados', 'figure'),
        Output('cum_infectados', 'figure'),
        Output('days_table', 'columns'),
        Output('days_table', 'data'),
        Output('daily_deaths', 'figure'),
        Output('cum_deaths', 'figure'),
        Output('status_infectados', 'figure'),
    ],
    [
        Input('fecha', 'start_date'),
        Input('fecha', 'end_date'),
        Input('departamento', 'value'),
        Input('municipio', 'value'),
    ],
    [
        State('rt_graph', 'figure'),
        State('reported', 'figure'),
        State('daily_infectados', 'figure'),
        State('cum_infectados', 'figure'),
        State('daily_deaths', 'figure'),
        State('cum_deaths', 'figure'),
        State('status_infectados', 'figure'),
    ]
)
def update_figure(start_date: datetime, end_date: datetime, \
    dpto: list=None, municipio: list=None, rt_graph=None, reported=None, daily_infectados=None, \
        cum_infectados=None, daily_deaths=None, cum_deaths=None, status_infectados=None) -> list:

    if dpto is None:
        dpto = list()
    if municipio is None:
        municipio = list()

    trecuperacion = d_hat

    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    locations = [*dpto, *municipio]
    df, df_covid, df_covid_raw, covid_dict = calculate_variables(locations, start_date)

    # Crea vector de tiempo para graficar
    time_vector = list(df_covid[(start_date <= df_covid.index) & (df_covid.index <= end_date)].index)

    df_covid_filter = df_covid[(start_date <= df_covid.index) & (df_covid.index <= end_date)]
    df_covid_raw_filter = df_covid_raw[(start_date <= df_covid.index) & (df_covid.index <= end_date)]

    annotation_dict = {
        'yanchor': 'bottom',
        'xref': 'x',
        'xanchor': 'center',
        'yref': 'y',
        'ay': -40,
        'ax': 0,
        'showarrow': True,
        'arrowhead': 2,
    }
    cuarentenas = [
        datetime(2020, 3, 25),
        datetime(2020, 4, 11),
        datetime(2020, 4, 27),
        datetime(2020, 5, 11),
        datetime(2020, 6, 1),
    ]
    
    data_rt = [
        {
            'x': time_vector[1:],
            'y': np.zeros(len(time_vector)-1) + 1,
            'hoverinfo': 'none',
            'line': {
                'color': 'red',
                'width': 1,
                'dash': 'dash'
            },
            'showlegend': False,
        },
        {
            'x': time_vector[1:],
            'y': np.zeros(len(time_vector)-1) + trecuperacion*np.log(2)/14+1,
            'hoverinfo': 'none',
            'name': 'Duplicación de casos en 14 días',
            'line': {
                'color': 'orangered',
                'width': 1,
                'dash': 'solid'
            },
            'showlegend': True,
        },
        {
            'x': time_vector[1:],
            'y': np.zeros(len(time_vector)-1) + trecuperacion*np.log(2)/30+1,
            'hoverinfo': 'none',
            'name': 'Duplicación de casos en 30 días',
            'line': {
                'color': 'gold',
                'width': 1,
                'dash': 'solid'
            },
            'showlegend': True,
        }

    ]

    # Update Rt
    for i, (location, (df_location, df_covid_location)) in enumerate(covid_dict.items()):
        update_rt(df_location, df_covid_location, location, start_date, end_date, rt_graph, data_rt, annotation_dict, \
            cuarentenas, colors[i], trecuperacion, estimados=True)
    
    update_status(covid_dict, status_infectados)

    return (
        rt_graph,
        *update_infectados(df_covid_filter, df_covid_raw_filter, reported, daily_infectados, cum_infectados),
        *update_matrix(df_covid, df_covid_raw),
        *update_deaths(df_covid_filter, df_covid_raw_filter, daily_deaths, cum_deaths),
        status_infectados,
    )

def update_rt(df, df_covid, name, start_date, end_date, rt_graph, data_rt, annotation_dict, cuarentenas, color, trecuperacion, estimados=False):
    if estimados:
        filt = 'estimados'
        msg = 'ajustado (nowcast)'
        dash = 'dashdot'
    else:
        filt = 'infectados'
        msg = 'sin ajuste'
        dash = 'solid'
    
    time_vector = list(df_covid.index)
    cumulcases = df_covid[filt] - df_covid['recuperados']

    activos = df_covid['infectados_activos']
    infectados = np.array(df_covid['estimados'], dtype=int)
    diarios = np.concatenate(([infectados[0]], [infectados[i] - infectados[i-1] for i in range(1, len(infectados))]))
    diarios = sgnl.filtfilt([1/3, 1/3, 1/3], [1.0], diarios)
    d = trecuperacion

    # Estima rt tomando usando los días de contagio promedio
    rt_raw = d * np.diff(np.log(cumulcases.astype('float64'))) + 1
    rt_raw = d * diarios / activos
    if len(rt_raw) > 9:
        rt_filt = sgnl.filtfilt([1/3, 1/3, 1/3], [1.0], rt_raw)
    else:
        rt_filt = rt_raw
    
    meanlen = 7
    aa = np.zeros(meanlen)
    rt_raw0 = rt_raw.copy()
    for i in range(meanlen):
        i1 = -meanlen-(meanlen-i)
        i2 = -(meanlen-i)
        aa[i] = np.mean(np.diff(rt_raw0[i1:i2]))
        rt_raw0[i2] = rt_raw0[i2-1] - aa[i]
    
    if len(rt_raw) > 9:
        rt_filt0 = sgnl.filtfilt([1/3, 1/3, 1/3], [1.0], rt_raw0)
    else:
        rt_filt0 = rt_raw0


    start = time_vector.index(start_date)
    end = time_vector.index(end_date)
    time_vector = time_vector[start + 1: end + 2]
    rt_filt = rt_filt[start: end + 1]
    rt_filt0 = rt_filt0[start: end + 1]
    
    new_data = {
        'x': time_vector, 
        'y': rt_filt, 
        'mode': 'lines', 
        'name': f'Rt {name} ' + msg,
        'line': {'color': color, 'dash': dash},
    }
    data_rt.append(new_data)

    new_data0 = {
        'x': time_vector, 
        'y': rt_filt0, 
        'fill':'tonexty',
        'mode': 'lines', 
        'name': f'Rt0 {name} ' + 'tendencia 9 días',
        'line': {'color': color, 'dash': dash},
    }
    data_rt.append(new_data0)

    annotations = list()
    for i, fecha_cuarentena in enumerate(cuarentenas):
        new_dict = deepcopy(annotation_dict)
        if fecha_cuarentena in time_vector:
            new_dict['y'] = rt_filt[time_vector.index(fecha_cuarentena)]
        else:
            continue
        new_dict['x'] = fecha_cuarentena
        new_dict['text'] = f'{i + 1}ᵃ cuarentena'
        annotations.append(new_dict)

    rt_graph['data'] = data_rt
    rt_graph['layout']['yaxis']['title'] = 'Rt'
    rt_graph['layout']['annotations'] = annotations


def update_infectados(df_covid, df_covid_raw, reported, daily_infectados, cum_infectados):
    time_vector = list(df_covid.index)

    infectados = df_covid_raw['nuevos_infectados'] 
    estimados = df_covid_raw['nuevos_estimados']
    reportados = df_covid_raw['nuevos_reportados']

    infectados_cum = df_covid['infectados'] 
    estimados_cum = df_covid['estimados']
    
    data_rep = [
        {
            'x': time_vector,
            'y': reportados,
            'type': 'bar',
            'name': 'Casos reportados por día',
        },
    ]
    reported['data'] = data_rep
    reported['layout']['yaxis']['title'] = 'Número de casos reportados por día'
    
    data_daily = [
        {
            'x': time_vector,
            'y': infectados,
            'type': 'bar',
            'name': 'Infectados por fecha síntomas',
        },
        {
            'x': time_vector,
            'y': estimados - infectados,
            'type': 'bar',
            'name': 'Estimados por nowcast',
        }
    ]
    daily_infectados['data'] = data_daily
    daily_infectados['layout']['yaxis']['title'] = 'Infectados diarios por fecha síntomas'
    daily_infectados['layout']['barmode'] = 'stack'

    data_cum = [
        {
            'x': time_vector,
            'y': infectados_cum,
            'type': 'bar',
            'name': 'Infectados reportados por fecha síntomas',
        },
        {
            'x': time_vector,
            'y': estimados_cum - infectados_cum,
            'type': 'bar',
            'name': 'Estimados acumulados por nowcast',
        }
    ]
    cum_infectados['data'] = data_cum
    cum_infectados['layout']['yaxis']['title'] = 'Infectados acumulados por fecha de síntomas'
    cum_infectados['layout']['barmode'] = 'stack'
    return reported, daily_infectados, cum_infectados


def update_deaths(df_covid, df_covid_raw, daily_deaths, cum_deaths):
    time_vector = list(df_covid.index)
    data_cum = [
        {
            'x': time_vector,
            'y': df_covid['fallecidos'],
            'type': 'bar',
            'name': 'Fallecidos acumulados',
        }
    ]
    data_daily = [
        {
            'x': time_vector,
            'y': df_covid_raw['nuevos_fallecidos'],
            'type': 'bar',
            'name': 'Fallecidos acumulados',
        }
    ]
    cum_deaths['data'] = data_cum
    cum_deaths['layout']['yaxis']['title'] = 'Fallecidos acumulados'
    daily_deaths['data'] = data_daily
    daily_deaths['layout']['yaxis']['title'] = 'Fallecidos diarios'
    return daily_deaths, cum_deaths


def update_status(covid_dict, status_infectados):
    locations = list(covid_dict.keys())
    options = ['Hospital', 'Hospital Uci', 'Recuperado', 'Fallecido']
    y = defaultdict(list)
    for location in locations:
        df = covid_dict[location][0]
        df = df.groupby('atencion').count()[['id']]
        for option in options:
            try:
                y[option].append(df.loc[option, 'id'])
            except KeyError:
                y[option].append(0)

    data = [
        {
            'x': locations,
            'y': y[option],
            'type': 'bar',
            'name': option,
        } for option in options
    ]
    status_infectados['data'] = data



def update_matrix(df_covid, df_covid_raw):
    data_table = df_covid.merge(df_covid_raw, how='inner', left_index=True, right_index=True).reset_index().rename(columns={'index': 'fecha'}).tail(20).iloc[::-1]
    data_table['fecha'] = data_table['fecha'].dt.date
    data_table['infectados_activos'] = data_table['estimados'] - data_table['recuperados']
    data_table['recuperados'] = data_table['recuperados'] - data_table['fallecidos']
    data_table['nuevos_recuperados'] = data_table['nuevos_recuperados'] - data_table['nuevos_fallecidos']
    rename_dict = {
        'fecha': 'Fecha',
        'infectados': 'Infectados acumulados', 
        'fallecidos': 'Fallecidos',
        'estimados': 'Infectados-nowcast',
        'infectados_activos': 'Infectados activos',
        'nuevos_infectados': 'Infectados por FIS', 
        'nuevos_fallecidos': 'Fallecidos por día',
        'nuevos_estimados': 'Infectados-nowcast por día',
        'nuevos_reportados': 'Reportados por día',
        }
    data_table = data_table.rename(columns=rename_dict)
    data_table = data_table[['Fecha', 'Infectados acumulados', 'Fallecidos', 'Infectados-nowcast', \
        'Infectados por FIS', 'Fallecidos por día', 'Infectados-nowcast por día', 'Reportados por día']]
    columns = [{'name': col, 'id': col} for col in data_table.columns]
    data = data_table.to_dict('records')
    return columns, data


def update_table(df, info_table):
    # Actualiza tabla
    positivos = df.shape[0]
    importados = df[df['tipo_contagio'] == 'Importado'].shape[0]
    recuperados = df[df['atencion'] == 'Recuperado'].shape[0]
    fallecidos = df[df['atencion'] == 'Fallecido'].shape[0]
    casa = df[df['atencion'] == 'Casa'].shape[0]
    hosp = df[df['atencion'] == 'Hospital'].shape[0]
    uci = df[df['atencion'] == 'Hospital Uci'].shape[0]
    activos = casa + hosp + uci
    table_values = [
        ['Positivos', 'Importados', 'Recuperados','Fallecidos'], 
        list(map(thousand_sep, [positivos, importados, recuperados, fallecidos])),
        ['Activos', 'En casa', 'Hospitalizados', 'En UCI'],
        list(map(thousand_sep, [activos, casa, hosp, uci]))
    ]

    info_table['data'][0]['cells']['values'] = table_values
    return info_table


def calculate_days(time_vector, df): # deprecated
    d_vector = list()
    for day in time_vector:
        new_df = df[df['fecha_sintomas'] <= day]
        n = new_df['dias'].count()
        d_raw = new_df['dias'].median(skipna=True)
        if n >= 20:
            d = d_raw
        elif n == 0:
            d = d_hat
        else:
            d = d_raw * n / 20 + d_hat * (20-n)/20
        d_vector.append(d)
    return d_vector

def delay_probability(df):
    df = df[df['fecha_reporte'] > (pd.Timestamp(current_date-timedelta(days=14)))] #si quiere mirar solo los retrasos de reportes recientes
    df = df[df['dias_retraso'] >= 0 ] #solo los que tienen retraso positivo. Hay retrasos negativos.
    total = df.shape[0]
    df_filter = df.groupby('dias_retraso', sort=True).count()['id']
    probabilities = {ix: sum(df_filter.loc[df_filter.index <= ix])/total for ix in df_filter.index}
    return probabilities

def get_dfs(df, start_date):
    # Número de infectados por fecha
    df1 = df.groupby('fecha_sintomas').count()[['id']].rename(columns={'id': 'nuevos_infectados'})
    # Número de recuperados por fecha
    df2 = df.groupby('fecha_recuperacion').count()[['id']].rename(columns={'id': 'nuevos_recuperados'})
    # Número de fallecidos por fecha
    df3 = df.groupby('fecha_muerte').count()[['id']].rename(columns={'id': 'nuevos_fallecidos'})
    # Número de reportados por fecha
    df4 = df.groupby('fecha_reporte').count()[['id']].rename(columns={'id': 'nuevos_reportados'})
    # Mergea (y ordena) los DataFrames
    df_merged = df1.merge(df2, how='outer', left_index=True, right_index=True).merge(df3, how='outer', left_index=True, right_index=True).merge(df4, how='outer', left_index=True, right_index=True)
    # Corrige fecha de inicio en caso de ser necesario
    start_date = min(df_merged.index.min(), start_date)
    # Crea DataFrame de fechas continuas desde el principio de la epidemia
    df_dates = pd.DataFrame(index=pd.date_range(start=start_date, end=current_date))
    # Rellena el DataFrame para que en los días que no hubo casos reportados asignar el valor de 0
    df_covid_raw = df_dates.merge(df_merged, how='left', left_index=True, right_index=True, sort=True).fillna(0)
    # Agrega estimados
    p = delay_probability(df)
    probabilities = [1 / p[day] if day in p else 1 for day in (current_date - df_dates.index).days]
    df_covid_raw['nuevos_estimados'] = (df_covid_raw['nuevos_infectados'] * probabilities).apply(lambda x: round(x))
    # Crea DataFrame con los infectados acumulados hasta la fecha
    rename_dict = {
        'nuevos_infectados': 'infectados', 
        'nuevos_recuperados': 'recuperados', 
        'nuevos_fallecidos': 'fallecidos',
        'nuevos_estimados': 'estimados',
        'nuevos_reportados': 'reportados'
        }
    df_covid = df_covid_raw.cumsum().rename(columns=rename_dict)

    # Crea lista con los infectados activos
    infectados_activos = [sum(df_covid_raw.loc[start_date + timedelta(days=i-11):start_date + timedelta(days=i-3)]['nuevos_estimados']) for i in range(len(df_covid_raw))]
    df_covid['infectados_activos'] = infectados_activos

    return df_covid_raw, df_covid


def calculate_variables(locations, start_date):
    if not locations:
        df = covid_data
        df_covid_raw, df_covid = get_dfs(df, start_date)
        covid_dict = {'Colombia': [df, df_covid]}
    else:
        covid_dict = dict()
        dfs = list()
        raws = list()
        cleans = list()
        for location in locations:
            df = covid_data[(covid_data['departamento'] == location) | (covid_data['municipio'] == location)]
            df_covid_raw, df_covid = get_dfs(df, start_date)
            dfs.append(df)
            raws.append(df_covid_raw)
            cleans.append(df_covid)
            covid_dict[location] = (df, df_covid)
        
        df_covid_raw = pd.concat(raws).groupby(level=0, sort=True).sum()
        df_covid = pd.concat(cleans).groupby(level=0, sort=True).sum()
        df = pd.concat(dfs).reset_index(drop=True)

    return df, df_covid, df_covid_raw, covid_dict

if __name__ == '__main__':
    app.run_server(debug=False)
