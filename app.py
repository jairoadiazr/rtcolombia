import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import scipy.signal as sgnl
import plotly.graph_objects as go
from datetime import datetime, timedelta
from copy import deepcopy

# Fecha reporte
current_date = datetime.now().date()

# Obtiene información de covid Colombia
covid_data = pd.read_json('https://www.datos.gov.co/resource/gt2j-8ykr.json?$limit=1000000')

# Crea diccionarios para renombrar columnas
rename_dict = {
    'id_de_caso': 'id',
    'fecha_de_notificaci_n': 'fecha_notificacion',
    'codigo_divipola': 'id_municipio',
    'ciudad_de_ubicaci_n': 'municipio',
    'departamento': 'departamento',
    'atenci_n': 'atencion',
    'edad': 'edad',
    'sexo': 'sexo',
    'tipo': 'tipo_contagio',
    'estado': 'estado_salud',
    'pa_s_de_procedencia': 'pais_procedencia',
    'fis': 'fecha_sintomas',
    'fecha_de_muerte': 'fecha_muerte',
    'fecha_diagnostico': 'fecha_diagnostico',
    'fecha_recuperado': 'fecha_recuperacion',
    'fecha_reporte_web': 'fecha_reporte'
}

# Renombra las columnas
covid_data = covid_data.rename(columns=rename_dict)

# Unifica valores de las columnas
columnas_corregir = ['municipio', 'departamento', 'atencion', 'sexo',
                     'tipo_contagio', 'estado_salud', 'pais_procedencia']
for col in columnas_corregir:
    covid_data[col] = covid_data[col].fillna('-')
    covid_data[col] = covid_data[col].apply(lambda x: x.title())

# ¿Qué hacer con los pacientes recuperados sin fecha de recuperación?
falta_fecha_recuperacion = covid_data[(covid_data['fecha_recuperacion'] == '-   -') &
                                      (covid_data['atencion'] == 'Recuperado')].shape[0]
if falta_fecha_recuperacion:
    print(f'Faltantes fecha recuperación: {falta_fecha_recuperacion}')

# Fechas
fechas = ['fecha_notificacion', 'fecha_diagnostico', 'fecha_sintomas', 
          'fecha_muerte', 'fecha_recuperacion', 'fecha_reporte']

# Reemplaza fechas con valores '-   -' o 'Asintomático' por np.datetime64('NaT')
for fecha in fechas:
    covid_data[fecha] = covid_data[fecha].replace(['-   -', 'Asintomático'], np.datetime64('NaT'))
    try:
        covid_data[fecha] = pd.to_datetime(covid_data[fecha])
    except Exception as e:
        print('Hay una fecha en formato incorrecto: ', e)
        covid_data[fecha] = pd.to_datetime(covid_data[fecha], errors='coerce')

# Para los fallecidos, se asigna su fecha de recuperación como su fecha de muerte
covid_data.loc[covid_data['estado_salud'] == 'Fallecido', 'fecha_recuperacion'] = covid_data[covid_data['estado_salud'] == 'Fallecido']['fecha_muerte']

# Calcula el número de días desde la fecha de inicio de síntomas hasta la fecha de recuperación
covid_data['dias'] = (covid_data['fecha_recuperacion'] - covid_data['fecha_sintomas']).apply(lambda x: x.days)

# Funciones auxiliares

def thousand_sep(n: int) -> str:
    return f'{n:,}'

# Colors from tab10 palette
colors = ['#d62728', '#ff7f0e', '#1f77b4'][::-1]

external_stylesheets = ['https://cdn.rawgit.com/gschivley/8040fc3c7e11d2a4e7f0589ffc829a02/raw/fe763af6be3fc79eca341b04cd641124de6f6f0d/dash.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Rt Colombia'
server = app.server


app.layout = html.Div(
    [
        html.H1(
            children='COVID19 Colombia',
            style={'text-align': 'center'}
        ),
        html.H3(
            children='Cálculo de Rt en tiempo real',
            style={'text-align': 'center'}
        ),
        html.H6(
            [
                html.Label('Departamento o distrito especial'),
                dcc.Dropdown(
                    id='departamento',
                    options=[{'label': dpto, 'value': dpto} for dpto in np.concatenate((np.array(['Todos']),np.sort(covid_data['departamento'].unique())))],
                    placeholder='Seleccione un departamento o distrito especial',
                ),
                html.Label('Municipio'),
                dcc.Dropdown(
                    id='municipio',
                    options=[{'label': city, 'value': city} for city in np.sort(covid_data['municipio'].unique())],
                    placeholder='Seleccione un municipio',
                ),
                html.Label('Filtro de fecha'),
                dcc.DatePickerRange(
                    id='fecha',
                    min_date_allowed=covid_data['fecha_sintomas'].min(),
                    max_date_allowed=current_date,
                    initial_visible_month=current_date,
                    end_date=current_date,
                    start_date=current_date - timedelta(days=30)
                )
            ]
        ),
        html.Div(
            [
                dcc.Graph(
                    id='rt-graph',
                    config={
                        'modeBarButtonsToRemove': [
                            'autoScale2d', 'select2d', 'zoom2d',
                            'pan2d', 'toggleSpikelines',
                            'hoverCompareCartesian',
                            'zoomOut2d', 'zoomIn2d',
                            'hoverClosestCartesian',
                            'resetScale2d'
                        ]
                    }
                ),
                dcc.Markdown('IMPORTANTE: El reporte real de infectados y recuperados a la fecha presenta en promedio un retraso mayor a 7 días por lo que la interpretación de los valores de Rt para fechas menores a 7 días a partir del día de hoy debe ser hecha con precaución'),
                dcc.Graph(
                    id='table-fig',
                    figure={
                        'layout': {
                            'height':400,
                            'margin': {'l': 80, 'r': 50, 't': 40}
                        }
                    }
                ),
                dcc.Graph(
                    id='log_infectados',
                    config={
                        'modeBarButtonsToRemove': [
                            'autoScale2d', 'select2d', 
                            'zoom2d', 'pan2d', 
                            'toggleSpikelines',
                            'hoverCompareCartesian',
                            'zoomOut2d', 'zoomIn2d',
                            'hoverClosestCartesian',
                            'resetScale2d'
                        ]
                    }
                )
            ],
        ),
            dcc.Markdown('Elaborado por:'),
            dcc.Markdown('- Jairo Díaz, División de Ciencias Básicas, Universidad del Norte - Barranquilla'),
            dcc.Markdown('- Jairo Espinosa, Facultad de Minas, Universidad Nacional de Colombia - Medellín'),
            dcc.Markdown('- Héctor López'),
            dcc.Markdown('- Bernardo Uribe, División de Ciencias Básicas, Universidad del Norte - Barranquilla'),
            dcc.Markdown('La información completa de este proyecto se puede consultar en :'),
            dcc.Markdown('https://sites.google.com/site/bernardouribejongbloed/home/RtColombia'),
            dcc.Markdown('Sociedad Colombiana de Matemáticas')
    ],
className='container'
)



@app.callback(
    [
        Output('municipio', 'options'),
    ],
    [
        Input('departamento', 'value'),
    ]
)
def update_combo_municipio(dpto: str):
    municipio_list=np.sort(covid_data[covid_data['departamento']==dpto]['municipio'].unique())
    municipio_list=np.concatenate((np.array(['Todos']),municipio_list))
    municipio_list=[{'label': city, 'value': city} for city in municipio_list],
    return municipio_list

@app.callback(
    [
        Output('rt-graph', 'figure'),
        Output('log_infectados', 'figure'),
        Output('table-fig', 'figure'),
    ],
    [
        Input('fecha', 'start_date'),
        Input('fecha', 'end_date'),
        Input('departamento', 'value'),
        Input('municipio', 'value'),
    ]
)

def update_figure(start_date: datetime, end_date: datetime, dpto: str=None, municipio: str=None) -> list:
    if (dpto is None and municipio is None) or dpto=='Todos':
        df = covid_data
        print(1)
    elif (municipio is None) or (municipio=='Todos'):
        df = covid_data[covid_data['departamento'] == dpto]
        print(2)
    elif dpto is None:
        df = covid_data[covid_data['municipio'] == municipio]
        print(3)
    else:
        if np.sum((covid_data['departamento'] == dpto) & (covid_data['municipio'] == municipio))==0:
            df = covid_data[covid_data['departamento'] == dpto]
            print(4.1)
        else:
            df = covid_data[(covid_data['departamento'] == dpto) & (covid_data['municipio'] == municipio)]
            print(4.2)

    print(municipio)
    print(df)
    #Calcular tiempos de retraso en la notificación
    Wt=covid_data['fecha_reporte']-covid_data['fecha_sintomas']
    Wt=[i.days for i in Wt]
    Wt = np.array(Wt)[~np.isnan(Wt)]
    Wt=Wt.astype('int')
    Wtmedio=np.median(Wt)

    Pretraso=np.zeros(np.max(Wt)+1)
    for d in range(np.max(Wt)+1):
        Pretraso[d]=np.sum(Wt<=d)/len(Wt)

    Pretraso[Pretraso==0]=np.min(Pretraso[Pretraso!=0])

    df_imp = df[df['tipo_contagio'] == 'Importado']
    df_imp['fecha_sintomas'][df_imp['fecha_sintomas'].isnull()]=df_imp['fecha_notificacion'][df_imp['fecha_sintomas'].isnull()]-timedelta(days=int(Wtmedio))

    df_no_imp = df[df['tipo_contagio'] != 'Importado']
    df_no_imp['fecha_sintomas'][df_no_imp['fecha_sintomas'].isnull()]=df_no_imp['fecha_notificacion'][df_no_imp['fecha_sintomas'].isnull()]-timedelta(days=int(Wtmedio))

    df_fall = df[df['atencion'] == 'Fallecido']
    df_no_imp_fall = df_no_imp[df_no_imp['atencion'] == 'Fallecido']
    
    df_recu = df[df['atencion'] == 'Recuperado']
    df_no_imp_recu = df_no_imp[df_no_imp['atencion'] == 'Recuperado']
    
    df_infect = df[df['atencion'].isin(['Hospital', 'Hospital Uci', 'Casa'])]
    df_hosp = df_infect[df_infect['atencion'] == 'Hospital']
    df_uci = df_infect[df_infect['atencion'] == 'Hospital Uci']
    df_casa = df_infect[df_infect['atencion'] == 'Casa']

    
    
    # Si para el DataFrame actual no se tiene información suficiente, se usa la información total del país
    if df['dias'].count() >= 20:
        df_days = df
    else:
        df_days = covid_data
    
    # Calcula media y mediana de tiempo de recuperación 
    d_mean = np.nanmean(df_days['dias'])
    d_hat = np.nanmedian(df_days['dias'])
    d_hat_max = np.nanquantile(df_days['dias'], 0.975)
    d_hat_min = np.nanquantile(df_days['dias'], 0.025)

    print('Media de días = ', d_mean)
    print('Mediana de días = ', d_hat)

    # Número de infectados por fecha
    df1 = df_no_imp.groupby('fecha_sintomas').count()[['id']].rename(columns={'id': 'infectados'})
    
    # Número de recuperados por fecha
    df2 = df_recu.groupby('fecha_recuperacion').count()[['id']].rename(columns={'id': 'recuperados'})

    # Número de fallecidos por fecha
    df3 = df_fall.groupby('fecha_muerte').count()[['id']].rename(columns={'id': 'fallecidos'})

    # Mergea (y ordena) los DataFrames
    df_covid = df1.merge(df2, how='outer', left_index=True, right_index=True).merge(df3, how='outer', left_index=True, right_index=True)

    # Fecha primer caso
    first_date = df_covid.index.min()
    print(municipio)
    print(df_covid)
    # Crea DataFrame de fechas continuas desde el principio de la epidemia
    df_dates = pd.DataFrame(index=pd.date_range(start=first_date, end=current_date))

    # Rellena el DataFrame para que en los días que no hubo casos reportados asignar el valor de 0
    df_covid_diario = df_dates.merge(df_covid, how='left', left_index=True, right_index=True, sort=True).fillna(0)
    
    #Calcular diferencias entre la fecha actual y las fechas del dataframe
    T_t=pd.to_datetime(current_date)-df_covid_diario.index
    T_t=np.array(T_t.astype('timedelta64[D]').astype('int'))

    #Omitir diferencias mayores que sean mayores que el retraso más largo visto
    T_t[T_t>=len(Pretraso)]=len(Pretraso)-1

    #ajustar número de infectados
    df_covid_diario['infectados_sinajuste']=df_covid_diario['infectados']
    df_covid_diario['infectados']=df_covid_diario['infectados']*1/Pretraso[T_t]

    #Calcula acumulados en el dataframe
    df_covid = df_covid_diario.cumsum()

    # Filtra el DataFrame con las fechas indicadas
    df_covid = df_covid[(start_date <= df_covid.index) & (df_covid.index <= end_date)]
    
    time_vector = list(df_covid.index)
    
    # Imprime DataFrame con los infectados, recuperados y fallecidos por día
    #print(df_covid.head())
    #print(df_covid.tail())
    
    # Crea array con el número de infectados acumulado por día
    cum_infectados_sinajuste = df_covid['infectados_sinajuste']
    cum_infectados = df_covid['infectados']
    cum_recu = df_covid['recuperados']
    cumulcases = cum_infectados - cum_recu
    cumulcases_sinajuste = cum_infectados_sinajuste - cum_recu
    print('Infectados acumulados', cum_infectados, sep='\n', end='\n\n')
    print('Recuperados acumulados', cum_recu, sep='\n', end='\n\n')
    print('Actuales acumulados', cumulcases, sep='\n', end='\n\n')

    # Log infectados
    log_infect = np.log(cumulcases.astype('float64'))

    # Estima rt tomando usando los días de contagio promedio
    rt_raw = d_hat * np.diff(np.log(cumulcases.astype('float64')))+1
    if len(rt_raw) > 9:
        rt_filt = sgnl.filtfilt([0.3333, 0.3333, 0.3333], [1.0], rt_raw)
    else:
        rt_filt = rt_raw

    # Estima rt tomando usando los días de contagio promedio para infectados SIN AJUSTE
    rt_raw_sinajuste = d_hat * np.diff(np.log(cumulcases_sinajuste.astype('float64')))+1
    if len(rt_raw_sinajuste) > 9:
        rt_filt_sinajuste = sgnl.filtfilt([0.3333, 0.3333, 0.3333], [1.0], rt_raw_sinajuste)
    else:
        rt_filt_sinajuste = rt_raw_sinajuste

    # rt_max
    rt_raw_max = d_hat_max * np.diff(np.log(cumulcases.astype('float64'))) + 1
    if len(rt_raw) > 9:
        rt_filt_max = sgnl.filtfilt([0.3333, 0.3333, 0.3333], [1.0], rt_raw_max)
    else:
        rt_filt_max = rt_raw_max
    
    # rt_min
    rt_raw_min = d_hat_min * np.diff(np.log(cumulcases.astype('float64'))) + 1
    if len(rt_raw) > 9:
        rt_filt_min = sgnl.filtfilt([0.3333, 0.3333, 0.3333], [1.0], rt_raw_min)
    else:
        rt_filt_min = rt_raw_min

    # rt_1
    rt_1 = np.zeros(len(time_vector)) + 1

    tick_suffix = ' '

    data_infectados = [
        {
            'x': time_vector,
            'y': log_infect,
            'hoverinfo': 'text',
            'type': 'scatter',
            'mode': 'lines',
            'name': 'log(infectados)',
            'line': {
                'color': colors[0],
                'width': 1
            },
            'text': [f'{date}<br>{val:.2f} ' for date, val in zip(time_vector, log_infect)]
        }
    ]
    
    data_rt = [
        {
            'x': time_vector,
            'y': rt_filt,
            'hoverinfo': 'text',
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Rt ajustado',
            'line': {
                'color': 'darkgreen',
                'width': 1
            },
            'text': [f'{date}<br>{val:.2f} ' for date, val in zip(time_vector, rt_filt)]
        },
        {
            'x': time_vector,
            'y': rt_filt_sinajuste,
            'hoverinfo': 'text',
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Rt sin ajuste',
            'line': {
                'color': 'lightgreen',
                'width': 1
            },
            'text': [f'{date}<br>{val:.2f} ' for date, val in zip(time_vector, rt_filt_sinajuste)]
        },
        {
            'x': time_vector,
            'y': rt_1,
            'hoverinfo': 'text',
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Rt = 1',
            'line': {
                'color': 'blue',
                'width': 1,
                'dash': 'dash'
            },
            'text': [f'{date}<br>{val:.2f} ' for date, val in zip(time_vector, rt_1)]
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
        datetime(2020, 5, 11),
    ]

    annotation=list()
    for i, fecha_cuarentena in enumerate(cuarentenas):
        new_dict = deepcopy(annotation_dict)
        if fecha_cuarentena in time_vector:
            new_dict['y'] = rt_filt[time_vector.index(fecha_cuarentena)]
        else:
            continue
        new_dict['x'] = fecha_cuarentena
        new_dict['text'] = f'{i + 1}ᵃ cuarentena'
        annotation.append(new_dict)

    # Actualiza gráfica de infectados
    log_infectados={
        'data': data_infectados,
        'layout': {
            'height':400,
            'legend': {
                'orientation': 'h',
                "x": 0.5,
                'xanchor': 'center'
            },
            'margin': {'l': 80, 'r': 50, 't': 40},
            'hovermode': 'closest',
            'yaxis': {
                'ticksuffix': tick_suffix,
                'title': 'log(infectados)',
                'showgrid': True,
            },
            'xaxis': {
                'range': [start_date, end_date],
                'showgrid': True,
            },
        }
    }

    # Actualiza gráfica de rt
    rt_graph = {
        'data': data_rt,
        'layout': {
            'title': f'Tiempo medio de recuperación: {round(d_hat, 2)} días',
            'legend': {
                'orientation': 'h',
                "x": 0.5,
                'xanchor': 'center'
            },
            'margin': {'l': 80, 'r': 50, 't': 40},
            'annotations': annotation,
            'hovermode': 'closest',
            'yaxis': {
                'ticksuffix': tick_suffix,
                'title': 'Rt',
                'showgrid': True,
            },
            'xaxis': {
                'range': [start_date, end_date],
                'showgrid': True,
            },
        }
    }
    
    table_values = [
        ['Positivos', 'Importados', 'Recuperados','Fallecidos'], 
        list(map(thousand_sep, [df.shape[0], df_imp.shape[0], df_recu.shape[0], df_fall.shape[0]])),
        ['Activos', 'En casa', 'Hospitalizados', 'En UCI'],
        list(map(thousand_sep, [df_infect.shape[0], df_casa.shape[0], df_hosp.shape[0], df_uci.shape[0]]))
    ]

    table = go.Figure(
        data=[
            go.Table(
                columnorder = [1,2,3,4],
                columnwidth = [400,400,400,400],
                header = dict(
                    values = [
                        ['Casos'],
                        ['Número'],
                        ['Infectados'],
                        ['Número']
                    ],
                    line_color='darkslategray',
                    fill_color='gray',
                    align=['center'],
                    font=dict(color='white', size=12),
                    height=30
                ),
                cells=dict(
                    values=table_values,
                    line_color='darkslategray',
                    fill=dict(color=['lightgray', 'white','lightgray', 'white']),
                    align=['center'],
                    font_size=12,
                    height=30
                )
            )
        ]
    )

    return rt_graph, log_infectados, table

if __name__ == '__main__':
    app.run_server(debug=True)