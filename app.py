# Dash
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Librerías estándar 
import pandas as pd
import numpy as np
import scipy.signal as sgnl
import plotly.graph_objects as go
from datetime import datetime, timedelta
from copy import deepcopy

# Clase CovidData
from covid import CovidData

# Fecha reporte
current_date = datetime.now().date()

# Obtiene información de covid Colombia
covid_data = pd.read_json('https://www.datos.gov.co/resource/gt2j-8ykr.json?$limit=1000000')

# Instancia la información en la clase CovidData
cd = CovidData(covid_data)
cd.preprocessing_data()

covid_data = cd.covid_data
d_hat = cd.d_hat
w_hat = cd.w_hat

# Funciones auxiliares

def thousand_sep(n: int) -> str:
    return f'{n:,}'

app = dash.Dash(__name__)
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

app.layout = html.Div([
    html.H1(
        'COVID19 Colombia',
        className='title', #TODO
        style={'text-align': 'center'}
    ),
    html.H3(
        'Cálculo de Rt en tiempo real',
        className='subtitle', #TODO
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
                start_date=current_date - timedelta(days=30),
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
            html.Div([
                html.Div(
                    dcc.Markdown(f'Tiempo medio de infecciosidad: {d_hat}'),
                    style={'width': '25%'},
                    className='pretty_container',
                ),
                html.Div(
                    dcc.Markdown(f'Tiempo medio de retraso en el reporte: {w_hat}'),
                    style={'width': '25%'},
                    className='pretty_container',
                ),
                html.Div(
                    dcc.Markdown(f'Infectados activos: {100}'),
                    style={'width': '25%'},
                    className='pretty_container',
                ),
                html.Div(
                    dcc.Markdown(f'Fallecidos: {1}'),
                    style={'width': '25%'},
                    className='pretty_container',
                ),
        ],
        style={'display': 'flex', 'flex-direction': 'row'},
        ),
            html.Div(
                dcc.Graph(
                    id='rt_graph',
                    config=graph_config,
                    figure=go.Figure(
                        layout={
                            'legend': {
                                'orientation': 'h',
                                'x': 0.5,
                                'xanchor': 'center'
                            },
                            'title': {'text': ''},
                            'margin': {'t': 40, 'r': 40, 'l': 40, 'b': 60},
                            'hovermode': 'closest',
                            'plot_bgcolor': '#fff',
                            'yaxis': {
                                'title': 'Rt',
                                'showgrid': True,
                                'gridcolor': 'whitesmoke',
                                'rangemode': 'tozero',
                            },
                            'xaxis': {
                                'showgrid': True,
                                'gridcolor': 'whitesmoke',
                            },
                        }
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
    dcc.Markdown('**IMPORTANTE:** El [reporte de infectados y recuperados](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data) \
        presenta en promedio un retraso mayor a 7 días, por lo que la interpretación de los valores de Rt para la última semana debe ser hecha con precaución.'),
    html.Div([
        html.Div(
            dcc.Graph(
                id='log_infectados',
                config=graph_config,
                figure=go.Figure(
                    layout={
                        'legend': {
                            'orientation': 'h',
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        'margin': {'t': 40, 'r': 40, 'l': 40, 'b': 60},
                        'hovermode': 'closest',
                        'plot_bgcolor': '#fff',
                        'yaxis': {
                            'title': 'log(Infectados activos)',
                            'showgrid': True,
                            'gridcolor': 'whitesmoke'
                        },
                        'xaxis': {
                            'showgrid': True,
                            'gridcolor': 'whitesmoke' 
                        },
                    }
                )
            ),
            style={'width': '50%'},
            className='pretty_container',
        ),        
        html.Div(
            dcc.Graph(
                id='daily_infectados',
                config=graph_config,
                figure=go.Figure(
                    layout={
                        'legend': {
                            'orientation': 'h',
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        'margin': {'t': 40, 'r': 40, 'l': 40, 'b': 60},
                        'hovermode': 'closest',
                        'plot_bgcolor': '#fff',
                        'yaxis': {
                            'title': 'Infectados diarios',
                            'showgrid': True,
                            'gridcolor': 'whitesmoke'
                        },
                        'xaxis': {
                            'showgrid': True,
                            'gridcolor': 'whitesmoke'
                        },
                        'barmode': 'stack',
                    }
                )
            ),
            style={'width': '50%'},
            className='pretty_container',
        )
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
    html.Div(
        dcc.Graph(
            id='cum_deaths',
            config=graph_config,
            figure=go.Figure(
                layout={
                    'legend': {
                        'orientation': 'h',
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    'margin': {'t': 40, 'r': 40, 'l': 40, 'b': 60},
                    'hovermode': 'closest',
                    'plot_bgcolor': '#fff',
                    'yaxis': {
                        'title': 'Fallecidos acumulados',
                        'showgrid': True,
                        'gridcolor': 'whitesmoke'
                    },
                    'xaxis': {
                        'showgrid': True,
                        'gridcolor': 'whitesmoke'
                    },
                }
            )
        ),
        className='pretty_container',
    ),
    dcc.Graph(
        id='info_table',
        figure=go.Figure(
            go.Table(
                cells={
                    'line_color': 'darkslategray',
                    'fill_color': ['lightgray', 'white','lightgray', 'white'],
                    'font_size': 12,
                    'height': 30
                },
                header = {
                    'values': ['Casos', 'Número', 'Infectados', 'Número'],
                    'line_color': 'darkslategray',
                    'fill_color': 'gray',
                    'font': {'color':'white', 'size': 12},
                    'height': 30
                },
            )
        )
    ),
    dcc.Markdown('Elaborado por:'),
    dcc.Markdown('- Jairo Díaz, División de Ciencias Básicas, Universidad del Norte - Barranquilla'),
    dcc.Markdown('- Jairo Espinosa, Facultad de Minas, Universidad Nacional de Colombia - Medellín'),
    dcc.Markdown('- Héctor López'),
    dcc.Markdown('- Bernardo Uribe, División de Ciencias Básicas, Universidad del Norte - Barranquilla'),
    dcc.Markdown('La información completa de este proyecto se puede consultar en :'),
    dcc.Markdown('https://sites.google.com/site/bernardouribejongbloed/home/RtColombia'),
    dcc.Markdown('Sociedad Colombiana de Matemáticas'),
    ],
className='container',
style={'display': 'flex', 'flex-direction': 'column'},
)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
@app.callback(
    [
        Output('rt_graph', 'figure'),
        Output('log_infectados', 'figure'),
        Output('daily_infectados', 'figure'),
        Output('info_table', 'figure'),
        Output('days_table', 'columns'),
        Output('days_table', 'data'),
        Output('cum_deaths', 'figure'),
    ],
    [
        Input('fecha', 'start_date'),
        Input('fecha', 'end_date'),
        Input('departamento', 'value'),
        Input('municipio', 'value'),
    ],
    [
        State('rt_graph', 'figure'),
        State('log_infectados', 'figure'),
        State('daily_infectados', 'figure'),
        State('info_table', 'figure'),
        State('cum_deaths', 'figure'),
    ]
)
def update_figure(start_date: datetime, end_date: datetime, dpto: str=None, municipio: str=None, \
    rt_graph=None, log_infectados=None, daily_infectados=None, info_table=None, cum_deaths=None) -> list:
    if dpto is None:
        dpto = list()
    if municipio is None:
        municipio = list()

    start_date, end_date = pd.to_datetime(start_date), pd.to_datetime(end_date)
    locations = [*dpto, *municipio]
    df, df_covid, df_covid_raw, covid_dict = calculate_variables(locations, start_date)

    # Crea vector de tiempo para graficar
    time_vector = list(df_covid[(start_date <= df_covid.index) & (df_covid.index <= end_date)].index)

    df_covid_filter = df_covid[(start_date <= df_covid.index) & (df_covid.index <= end_date)]
    df_covid_raw_filter = df_covid_raw[(start_date <= df_covid.index) & (df_covid.index <= end_date)]

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
        }
    ]

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
    ]
    for i, (location, (df_location, df_covid_location)) in enumerate(covid_dict.items()):
        update_rt(df_location, df_covid_location, location, start_date, end_date, rt_graph, data_rt, annotation_dict, cuarentenas, colors[i])
        update_rt(df_location, df_covid_location, location, start_date, end_date, rt_graph, data_rt, annotation_dict, cuarentenas, colors[i], estimados=True)

    return (
        rt_graph,
        *update_infectados(df_covid_filter, df_covid_raw_filter, log_infectados, daily_infectados, start_date, end_date),
        update_table(df, info_table), 
        *update_matrix(df_covid, df_covid_raw),
        update_deaths(df_covid_filter, cum_deaths, start_date, end_date),
    )

def update_rt(df, df_covid, name, start_date, end_date, rt_graph, data_rt, annotation_dict, cuarentenas, color, estimados=False):
    if estimados:
        filt = 'estimados'
        msg = 'ajustado (nowcast)'
        dash = 'dashdot'
    else:
        filt = 'infectados'
        msg = 'sin ajuste'
        dash = 'solid'
    
    time_vector = list(df_covid.index)
    d_vector = calculate_days(time_vector[1:], df)
    
    cumulcases = df_covid[filt] - df_covid['recuperados']

    # Estima rt tomando usando los días de contagio promedio
    rt_raw = d_vector * np.diff(np.log(cumulcases.astype('float64'))) + 1
    if len(rt_raw) > 9:
        rt_filt = sgnl.filtfilt([1/3, 1/3, 1/3], [1.0], rt_raw)
    else:
        rt_filt = rt_raw

    start = time_vector.index(start_date)
    end = time_vector.index(end_date)
    time_vector = time_vector[start + 1: end + 2]
    rt_filt = rt_filt[start: end + 1]
    
    new_data = {
        'x': time_vector, 
        'y': rt_filt, 
        'mode': 'lines', 
        'name': f'Rt {name} ' + msg,
        'line': {'color': color, 'dash': dash},
    }
    data_rt.append(new_data)

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
    # rt_graph['layout']['title']['text'] = f'Tiempo medio de recuperación: {round(d_vector[-1], 2)} días'
    rt_graph['layout']['annotations'] = annotations


def update_infectados(df_covid, df_covid_raw, log_infectados, daily_infectados, start_date, end_date):
    time_vector = list(df_covid.index)
    cumulcases = df_covid['infectados'] - df_covid['recuperados']
    estimate_cumulcases = df_covid['estimados'] - df_covid['recuperados']
    log_infect = np.log(cumulcases.astype('float64'))
    log_estim = np.log(estimate_cumulcases.astype('float64'))
    data_log = [
        {
            'x': time_vector,
            'y': log_infect,
            'mode': 'lines',
            'name': 'Activos',
        },
        {
            'x': time_vector,
            'y': log_estim,
            'mode': 'lines',
            'name': 'Estimados',
        }
    ]
    data_cum = [
        {
            'x': time_vector,
            'y': df_covid_raw['nuevos_infectados'],
            'type': 'bar',
            'name': 'Infectados reportados',
        },
        {
            'x': time_vector,
            'y': df_covid_raw['nuevos_estimados'] - df_covid_raw['nuevos_infectados'],
            'type': 'bar',
            'name': 'Estimados',
        }
    ]
    log_infectados['data'] = data_log
    daily_infectados['data'] = data_cum
    return log_infectados, daily_infectados


def update_deaths(df_covid, cum_deaths, start_date, end_date):
    time_vector = list(df_covid.index)
    cumuldeaths = df_covid['fallecidos']
    data_deaths = [
        {
            'x': time_vector,
            'y': cumuldeaths,
            'type': 'bar',
            'name': 'Fallecidos acumulados',
        }
    ]
    cum_deaths['data'] = data_deaths
    return cum_deaths


def update_matrix(df_covid, df_covid_raw):
    data_table = df_covid.merge(df_covid_raw, how='inner', left_index=True, right_index=True).reset_index().rename(columns={'index': 'fecha'}).tail(10).iloc[::-1]
    data_table['fecha'] = data_table['fecha'].dt.date
    data_table['recuperados'] = data_table['recuperados'] - data_table['fallecidos']
    data_table['nuevos_recuperados'] = data_table['nuevos_recuperados'] - data_table['nuevos_fallecidos']
    parse_head = lambda x: ' '.join(map(lambda y: y.title(), x.split('_')))
    columns = [{'name': parse_head(i), 'id': i} for i in data_table.columns]
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


def calculate_days(time_vector, df):
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
    # Mergea (y ordena) los DataFrames
    df_merged = df1.merge(df2, how='outer', left_index=True, right_index=True).merge(df3, how='outer', left_index=True, right_index=True)
    # Crea DataFrame de fechas continuas desde el principio de la epidemia
    df_dates = pd.DataFrame(index=pd.date_range(start=min(df_merged.index.min(), start_date), end=current_date))
    # Rellena el DataFrame para que en los días que no hubo casos reportados asignar el valor de 0
    df_covid_raw = df_dates.merge(df_merged, how='left', left_index=True, right_index=True, sort=True).fillna(0)
    # Agrega estimados
    p = delay_probability(df)
    probabilities = [1 / p[day] if day in p else 1 for day in (datetime.now() - df_dates.index).days]
    df_covid_raw['nuevos_estimados'] = (df_covid_raw['nuevos_infectados'] * probabilities).apply(lambda x: round(x))
    # Crea DataFrame con los infectados acumulados hasta la fecha
    rename_dict = {
        'nuevos_infectados': 'infectados', 
        'nuevos_recuperados': 'recuperados', 
        'nuevos_fallecidos': 'fallecidos',
        'nuevos_estimados': 'estimados'
        }
    df_covid = df_covid_raw.cumsum().rename(columns=rename_dict)
    
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
    app.run_server(debug=True, host='0.0.0.0')