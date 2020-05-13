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


data1 = pd.read_json('https://www.datos.gov.co/resource/gt2j-8ykr.json?$limit=1000000')
data_columns=['fecha_de_notificaci_n','fis','fecha_de_muerte','fecha_diagnostico','fecha_recuperado']
for column in data_columns:
  print(column)
  data1[column]=[i if is_date(str(i)) else np.nan for i in data1[column]]
  data1[column]=pd.to_datetime(data1[column])

dptolist=[{'label':i,'value':i} for i in np.sort(data1['departamento'].unique())]
dptolist.insert(0,{'label':'Colombia','value':'Colombia'})

currentYear = datetime.now().year

fn = 'https://raw.githubusercontent.com/gschivley/climate-life-events/master/iamc_db.csv'
df = pd.read_csv(fn)
df['climate'] = df['Scenario'].str.split('-').str[-1]
climates = df['climate'].unique()
years = pd.to_datetime(df.columns[6:-1], yearfirst=True)

fn = 'https://raw.githubusercontent.com/gschivley/climate-life-events/master/GISS_temps.csv'
hist = pd.read_csv(fn)
hist['datetime'] = pd.to_datetime(hist['datetime'], yearfirst=True)


# Colors from tab10 palette
colors = ['#d62728', '#ff7f0e', '#1f77b4'][::-1]

# for idx, climate in enumerate(['Low', 'Mid', 'High']):
#     trace = {
#         'x': years,
#         'y': df.loc[df['name'] == climate, '2010':'2100'].mean(),
#         'hoverinfo': 'text',#'text+x',
#         'showlegend': False,
#         'type': 'scatter',
#         'mode': 'lines',
#         'name': climate,
#         'line': {'color': 'rgb(33, 33, 33)'}
#     }
#     data.append(trace)

# # for idx, climate in enumerate(climates):
# for idx, climate in enumerate(['Low', 'Mid', 'High']):
#     trace = {
#         'x': years,
#         'y': df.loc[df['name'] == climate, '2010':'2100'].min(),
#         'hoverinfo': 'text',#'text+x',
#         'showlegend': False,
#         'type': 'scatter',
#         'mode': 'lines',
#         'name': 'min {}'.format(climate),
#         'line': {'color': colors[idx],
#                  'width': 0.5}
#     }
#     data.append(trace)

#     trace = {
#         'x': years,
#         'y': df.loc[df['name'] == climate, '2010':'2100'].max(),
#         'hoverinfo': 'text',#'text+x',
#         'type': 'scatter',
#         'fill': 'tonexty',
#         'mode': 'lines',
#         'name': climate,
#         'line': {'color': colors[idx],
#                  'width': 0.5}
#     }
#     data.append(trace)

# Define separate traces for units

external_stylesheets = ['https://cdn.rawgit.com/gschivley/8040fc3c7e11d2a4e7f0589ffc829a02/raw/fe763af6be3fc79eca341b04cd641124de6f6f0d/dash.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# app.scripts.config.serve_locally=True
# app.css.config.serve_locally=True
# app.config.supress_callback_exceptions=True
#app.css.append_css({'external_url':
#                    'https://cdn.rawgit.com/gschivley/8040fc3c7e11d2a4e7f0589ffc829a02/raw/fe763af6be3fc79eca341b04cd641124de6f6f0d/dash.css'
                    # 'https://rawgit.com/gschivley/8040fc3c7e11d2a4e7f0589ffc829a02/raw/8daf84050707365c5e266591d65232607f802a43/dash.css'

#                    })
app.title = 'Rt Colombia'
server = app.server


app.layout = html.Div(children=[
    # html.Link(href='/assets/stylesheet.css', rel='stylesheet'),
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
        # dcc.Input(id='child_birth', value=0, type='number'),
        dcc.Dropdown(
            id='units',
            options=dptolist,
            value='Atlántico'
        )
    ]
    ),
    #html.Div([
    #     html.P(id = "status",
    #        children=["init"],
    #        style={'width': '400px', 'margin-right': 'auto',
    #       'margin-left': 'auto', 'text-align': 'left',"white-space": "pre"})],
    #       className='input-wrapper'),
    html.Div(
    [
        dcc.Graph(
            id='example-graph',
            config={
                'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'zoom2d',
                                           'pan2d', 'toggleSpikelines',
                                           'hoverCompareCartesian',
                                           'zoomOut2d', 'zoomIn2d',
                                           'hoverClosestCartesian',
                                           # 'sendDataToCloud',
                                           'resetScale2d']
            }
        ),
                dcc.Graph(id='table-fig',figure={
            'layout': {'height':400,            'margin': {
                'l': 80,
                'r': 50,
                't': 40
            }}}
        ),
        dcc.Graph(
            id='example-graph0',
            config={
                'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'zoom2d',
                                           'pan2d', 'toggleSpikelines',
                                           'hoverCompareCartesian',
                                           'zoomOut2d', 'zoomIn2d',
                                           'hoverClosestCartesian',
                                           # 'sendDataToCloud',
                                           'resetScale2d']
            }
        )
        ],
        # style={'width': '75%', 'margin-right': 'auto', 'margin-left': 'auto'}
        ),
        dcc.Markdown('Elaborado por:'),
        dcc.Markdown('- Jairo Díaz, División de Ciencias Básicas, Universidad del Norte - Barranquilla'),
        dcc.Markdown('- Jairo Espinosa, Facultad de Minas, Universidad Nacional de Colombia - Medellín'),
        dcc.Markdown('- Bernardo Uribe, División de Ciencias Básicas, Universidad del Norte - Barranquilla'),
        dcc.Markdown('La información completa de este proyecto se puede consultar en :'),
        dcc.Markdown('https://sites.google.com/site/bernardouribejongbloed/home/RtColombia'),
        dcc.Markdown('Sociedad Colombiana de Matemáticas')

],
className='container'
# style={'width': '600px', 'margin-right': 'auto', 'margin-left': 'auto'}
)

@app.callback(
    [dash.dependencies.Output('example-graph', 'figure'),dash.dependencies.Output('example-graph0', 'figure'),dash.dependencies.Output('table-fig', 'figure')],
    [dash.dependencies.Input('units', 'value')])
def update_figure(units):
    Departamento=units
    data_covid=data1.copy()
    #data_covid['fis'] = [row['fis'] if ]


    #if dates_diaf.size>0 :
        
    #    data_covid.loc[(data_covid['fis'] =='Asintomático'),['fis']] = dates_diaf[:,1]
    #    print(123)
    #    print(data_covid['fis'].unique())
    #    data_covid.fis = pd.to_datetime(data_covid.fis)
    #    print(456)
    if Departamento!='Colombia':
        select_department = data_covid.loc[data_covid['departamento'] == Departamento]
        select_department_tipo_Imp = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['tipo']=='Importado')]
        select_department_tipo = data_covid.loc[(data_covid['departamento'] == Departamento)]
        select_department_tipo_sinimportados = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['tipo']!='Importado')]
        select_department_fallec = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['atenci_n'] =='Fallecido')]
        select_department_fallec_sinimportados = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['atenci_n'] =='Fallecido') & (data_covid['tipo']!='Importado')]
        select_department_recup = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['atenci_n'] =='Recuperado')]
        select_department_recup_sinimportados = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['atenci_n'] =='Recuperado') & (data_covid['tipo']!='Importado')]

        
        select_department_hospit = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['atenci_n'] =='Hospital')]
        select_department_UCI = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['atenci_n'] =='Hospital UCI')]
        select_department_En_Casa = data_covid.loc[(data_covid['departamento'] == Departamento) & (data_covid['atenci_n'] =='Casa')]
    else:
        Departamento='Colombia'
        print('colombia')
        select_department = data_covid
        select_department_tipo_Imp = data_covid.loc[data_covid['tipo']=='Importado']
        select_department_tipo = data_covid
        select_department_tipo_sinimportados = data_covid.loc[(data_covid['tipo']!='Importado')]
        select_department_fallec = data_covid.loc[(data_covid['atenci_n'] =='Fallecido')]
        select_department_fallec_sinimportados = data_covid.loc[(data_covid['atenci_n'] =='Fallecido') & (data_covid['tipo']!='Importado')]
        select_department_recup = data_covid.loc[(data_covid['atenci_n'] =='Recuperado')]
        select_department_recup_sinimportados = data_covid.loc[(data_covid['atenci_n'] =='Recuperado') & (data_covid['tipo']!='Importado')]
        
        select_department_hospit = data_covid.loc[(data_covid['atenci_n'] =='Hospital')]
        select_department_UCI = data_covid.loc[(data_covid['atenci_n'] =='Hospital UCI')]
        select_department_En_Casa = data_covid.loc[(data_covid['atenci_n'] =='Casa')]

    #######################################################################################
    #Estimar D con el promedio de los datos

    if select_department_recup.shape[0]!=0:
        select_department_recup['D']=select_department_recup['fecha_recuperado']-select_department_recup['fis']
        select_department_recup['D']=[i.days for i in select_department_recup['D']]
        D_hat=np.nanmean(select_department_recup['D'])
        D_hatmax=np.nanquantile(select_department_recup['D'],0.975)
        D_hatmin=np.nanquantile(select_department_recup['D'],0.025)
        D_Median=np.nanmedian(select_department_recup['D'])
    else:
        select_department_recup_colombia = data_covid.loc[(data_covid['atenci_n'] =='Recuperado')]
        select_department_recup_colombia['D']=select_department_recup_colombia['fecha_recuperado']-select_department_recup_colombia['fis']
        select_department_recup_colombia['D']=[i.days for i in select_department_recup_colombia['D']]
        D_hat=np.nanmean(select_department_recup_colombia['D'])
        D_hatmax=np.nanquantile(select_department_recup_colombia['D'],0.975)
        D_hatmin=np.nanquantile(select_department_recup_colombia['D'],0.025)
        D_Median=np.nanmedian(select_department_recup_colombia['D'])

    print ('D promedio='+str(D_hat))
    print('D Mediana='+str(D_Median))

    #######################################################################################

    #gran_total
    gran_total=select_department_tipo.shape[0]-select_department_recup.shape[0]-select_department_fallec.shape[0]

    #infectados
    select_department_tipo_sinimportados['fis'][select_department_tipo_sinimportados['fis'].isnull()]=select_department_tipo_sinimportados['fis'][select_department_tipo_sinimportados['fis'].isnull()]-timedelta(days=int(D_hat))
    select_department_tipo_sinimportados['fis'][select_department_tipo_sinimportados['fis']<data_covid['fecha_de_notificaci_n'].min()]=data_covid['fecha_de_notificaci_n'].min()
    df_covid = select_department_tipo_sinimportados.groupby(by='fis').count()

    #recuperados
    select_department_recup_sinimportados['fecha_recuperado'][select_department_recup_sinimportados['fecha_recuperado'].isnull()]=select_department_recup_sinimportados['fis'][select_department_recup_sinimportados['fecha_recuperado'].isnull()]+timedelta(days=int(D_hat))
    select_department_recup_sinimportados['fecha_recuperado'][select_department_recup_sinimportados['fecha_recuperado']>data_covid['fecha_de_notificaci_n'].max()]=data_covid['fecha_de_notificaci_n'].max()
    df_covid_rec = select_department_recup_sinimportados.groupby(by='fecha_recuperado').count()

    #fallecidos
    select_department_fallec_sinimportados['fecha_de_muerte'][select_department_fallec_sinimportados['fecha_de_muerte'].isnull()]=select_department_fallec_sinimportados['fis'][select_department_fallec_sinimportados['fecha_de_muerte'].isnull()]+timedelta(days=int(D_hat))
    select_department_fallec_sinimportados['fecha_de_muerte'][select_department_fallec_sinimportados['fecha_de_muerte']>data_covid['fecha_de_notificaci_n'].max()]=data_covid['fecha_de_notificaci_n'].max()
    df_covid_fa = select_department_fallec_sinimportados.groupby(by='fecha_de_muerte').count()

    df_covid_fa=df_covid_fa.reset_index().rename(columns={'fecha_de_muerte':'fecha','id_de_caso':'fallecidos'})[['fecha','fallecidos']]
    df_covid_rec=df_covid_rec.reset_index().rename(columns={'fecha_recuperado':'fecha','id_de_caso':'recuperados'})[['fecha','recuperados']]
    df_covid=df_covid.reset_index().rename(columns={'fis':'fecha','id_de_caso':'infectados'})[['fecha','infectados']]

    df_covid_fa['fecha']=pd.to_datetime(df_covid_fa['fecha']).dt.strftime('%Y-%m-%d')

    df_covid_rec['fecha']=pd.to_datetime(df_covid_rec['fecha']).dt.strftime('%Y-%m-%d')
    df_covid['fecha']=pd.to_datetime(df_covid['fecha']).dt.strftime('%Y-%m-%d')

    df_covid=df_covid.merge(df_covid_rec,'outer').merge(df_covid_fa,'outer').fillna(0)
    df_covid=df_covid.sort_values(by='fecha')

    df_array = df_covid.to_numpy()
    Total_NI_infected_cases = np.cumsum(df_array[:,1],0)-np.cumsum(df_array[:,2],0)-np.cumsum(df_array[:,3],0)
    Total_NI_recovered_cases= np.cumsum(df_array[:,2],0)
    print(df_covid)


    time_vector = pd.to_datetime(df_covid['fecha'])
    print(time_vector)
    #time_vector = datetime(df.index.to_numpy()).fromisoformat()
    tv = (time_vector-time_vector[0])
    cumulcases = Total_NI_infected_cases #-Total_NI_infected_cases[0]+1


    #######################################################################################
    #Estimar R_t tomando D del promedio de los datos

    #Rtraw = 14.0*np.diff(np.log(cumulcases.astype('float64')))+1
    Rtraw = D_hat*np.diff(np.log(cumulcases.astype('float64')))+1
    if len(Rtraw)>9:
        Rfilt = sgnl.filtfilt([0.3333, 0.3333, 0.3333],[1.0],Rtraw)
    else:
        Rfilt=Rtraw

    #Rmax
    Rtraw_max = D_hatmax*np.diff(np.log(cumulcases.astype('float64')))+1
    if len(Rtraw)>9:
        Rfilt_max = sgnl.filtfilt([0.3333, 0.3333, 0.3333],[1.0],Rtraw_max)
    else:
        Rfilt_max=Rtraw_max
    
    #Rmin
    Rtraw_min = D_hatmin*np.diff(np.log(cumulcases.astype('float64')))+1
    if len(Rtraw)>9:
        Rfilt_min = sgnl.filtfilt([0.3333, 0.3333, 0.3333],[1.0],Rtraw_min)
    else:
        Rfilt_min=Rtraw_min


    #########################################################################################

    def hovertext(datetime, temp, suffix):
        hover_year = datetime
        hover_string = '{}<br>{:.2f}{}'.format(hover_year, temp, suffix)

        return hover_string


    # data = []
    # trace = {
    #     'x': hist.loc[(hist['datetime'].dt.year <= 2010) &
    #                 (hist['datetime'].dt.year > 1879), 'datetime'],
    #     'y': hist.loc[(hist['datetime'].dt.year <= 2010) &
    #                 (hist['datetime'].dt.year > 1879), 'temp'],
    #     'hoverinfo': 'text',#'text+x',
    #     'type': 'scatter',
    #     'mode': 'lines',
    #     'name': 'Historical record',
    #     'line': {'color': 'rgb(33, 33, 33)'}
    # }
    # data.append(trace)
    data0=[]
    trace = {
        'x': time_vector,
        'y': np.log(cumulcases.astype('float64')),
        'hoverinfo': 'text',#'text+x',
        'type': 'scatter',
        'mode': 'lines',
        'name': 'log(infectados)',
        'line': {'color': colors[0],
                  'width': 1}
    }
    data0.append(trace)

    data = []
    trace = {
        'x': time_vector[1:],
        'y': Rfilt_min,
        'hoverinfo': 'text',#'text+x',
        'type': 'scatter',
        'mode': 'lines',
        'showlegend':False,
        'line': {'color': 'palegreen',
                  'width': 1}
    }
    data.append(trace)
    trace = {
        'x': time_vector[1:],
        'y': Rfilt_max,
        'fill':'tonexty',
        'fillcolor' : 'palegreen',
        'hoverinfo': 'text',#'text+x',
        'type': 'scatter',
        'mode': 'lines',
        'showlegend':False,
        'line': {'color': 'palegreen',
                  'width': 1}
    }
    data.append(trace)
    trace = {
        'x': time_vector[1:],
        'y': Rfilt,
        'hoverinfo': 'text',#'text+x',
        'type': 'scatter',
        'mode': 'lines',
        'name': 'Rt suavizado',
        'line': {'color': 'darkgreen',
                  'width': 1}
    }
    #data.append(trace)
    #trace = {
    #    'x': time_vector[1:],
    #    'y': Rtraw,
    #    'hoverinfo': 'text',#'text+x',
    #    'type': 'scatter',
    #    'mode': 'lines',
    #    'name': 'Rt diario',
    #    'line': {'color': 'lightgreen',
    #              'width': 1}
    #}
    data.append(trace)
    # trace = {
    #     'x': time_vector[1:],
    #     'y': Rfilt_2,
    #     'hoverinfo': 'text',#'text+x',
    #     'type': 'scatter',
    #     'mode': 'lines',
    #     'name': 'Rt2 suavizado',
    #     'line': {'color': 'darkred',
    #               'width': 1}
    # }
    # data.append(trace)
    # trace = {
    #     'x': time_vector[1:],
    #     'y': Rtraw_2,
    #     'hoverinfo': 'text',#'text+x',
    #     'type': 'scatter',
    #     'mode': 'lines',
    #     'name': 'Rt2 diario',
    #     'line': {'color': 'lightcoral',
    #               'width': 1}
    # }
    #data.append(trace)
    trace = {
        'x': time_vector[1:],
        'y': np.zeros(len(time_vector[1:]))+1,
        'hoverinfo': 'text',#'text+x',
        'type': 'scatter',
        'mode': 'lines',
        'name': 'Rt = 1',
        'line': {'color': 'blue',
                  'width': 1,'dash': 'dash'}
    }
    data.append(trace)


    # Set units on axis and scale number for imperial units
    tick_suffix = ' '
    _data = deepcopy(data)
    for trace in _data:
        # trace['text'] = ['{:.2f}°F'.format(y) for y in trace['y']]
        hover_inputs = zip(trace['x'], trace['y'])
        trace['text'] = [hovertext(x, y, tick_suffix)
                            for (x, y) in hover_inputs]
    for trace in data0:
        # trace['text'] = ['{:.2f}°F'.format(y) for y in trace['y']]
        hover_inputs = zip(trace['x'], trace['y'])
        trace['text'] = [hovertext(x, y, tick_suffix)
                            for (x, y) in hover_inputs]

    annotation = [
            # {
            #     "yanchor": "bottom",
            #     "xref": "x",
            #     "xanchor": "center",
            #     "yref": "y",
            #     "text": "Primer caso",
            #     "y": Rfilt[time_vector.date[1:]==min(time_vector.date[1:])][0], #0.75,
            #     "x": min(time_vector.date[1:]),
            #     "ay": -90,
            #     "ax": 0,
            #     "showarrow": True,
            #     'arrowhead': 2,
            # },
            {
                "yanchor": "bottom",
                "xref": "x",
                "xanchor": "center",
                "yref": "y",
                "text": "1a cuarentena",
                "y": Rfilt[abs(time_vector[1:]-datetime.strptime('2020-03-25', '%Y-%m-%d')).argmin()], #0.75,
                "x": '2020-03-25',
                "ay": -40,
                "ax": 0,
                "showarrow": True,
                'arrowhead': 2,
            },
            {
                "yanchor": "bottom",
                "xref": "x",
                "xanchor": "center",
                "yref": "y",
                "text": "2a cuarentena",
                "y": Rfilt[abs(time_vector[1:]-datetime.strptime('2020-04-11', '%Y-%m-%d')).argmin()], #0.75,
                "x": '2020-04-11',
                "ay": -40,
                "ax": 0,
                "showarrow": True,
                'arrowhead': 2,
            },
            {
                "yanchor": "bottom",
                "xref": "x",
                "xanchor": "center",
                "yref": "y",
                "text": "3a cuarentena",
                "y": Rfilt[abs(time_vector[1:]-datetime.strptime('2020-04-27', '%Y-%m-%d')).argmin()], #0.75,
                "x": '2020-04-27',
                "ay": -40,
                "ax": 0,
                "showarrow": True,
                'arrowhead': 2,
            }
            ]


    # push the lower lim of the xaxis back if needed

    figure0={
        'data': data0,
        'layout': {
            'height':400,
            'legend': {
                'orientation': 'h',
                "x": 0.5,
                'xanchor': 'center'
            },
            'margin': {
                'l': 80,
                'r': 50,
                't': 40
            },
            'hovermode': 'closest',
            'yaxis': {
                'ticksuffix': tick_suffix,#'°C',
                'title': 'log(infectados)',
                'showgrid': True,
            },
            'xaxis': {
                'range': [min(time_vector[1:]), max(time_vector[1:])],
                'showgrid': True,
                 #'title': 'Fecha'
            },
        }
    }
    figure={
        'data': _data,
        'layout': {
            'title':'Tiempo promedio de recuperación: '+str(int(D_hat))+ ' días',
            'legend': {
                'orientation': 'h',
                "x": 0.5,
                'xanchor': 'center'
            },
            'margin': {
                'l': 80,
                'r': 50,
                't': 40
            },
            'annotations': annotation,
            'hovermode': 'closest',
            'yaxis': {
                'ticksuffix': tick_suffix,#'°C',
                'title': 'Rt',
                'showgrid': True,
            },
            'xaxis': {
                'range': [min(time_vector[1:]), max(time_vector[1:])],
                'showgrid': True,
                 #'title': 'Fecha'
            },
            # "font": {
            #     "family": "Roboto",
            #     "size": 14
            # }
        }
    }



    textlabel=  'Casos Positivos = ' + str(select_department.shape[0]) +'\n' + 'Casos Importados  = ' + str(select_department_tipo_Imp.shape[0]) +'\n' + 'Casos Relacionados/en Estudio  = ' + str(select_department_tipo.shape[0])+'\n'+'Recuperados = ' + str(select_department_recup.shape[0])+'\n'+'En Casa = ' + str(select_department_En_Casa.shape[0])+'\n'+'Hospitalizados = ' + str(select_department_hospit.shape[0])+'\n'+'Hospitalizados/UCI = ' + str(select_department_UCI.shape[0])+'\n'+'Fallecidos = ' + str(select_department_fallec.shape[0])
    
    valuestable = [['Positivos', 'Importados', 'Recuperados','Fallecidos'], #1st col
    [select_department.shape[0], select_department_tipo_Imp.shape[0],select_department_recup.shape[0],select_department_fallec.shape[0]],
    ['Activos', 'En casa', 'Hospitalizados', 'En UCI'],
    [gran_total,select_department_En_Casa.shape[0], select_department_hospit.shape[0],select_department_UCI.shape[0]]]


    fig = go.Figure(data=[go.Table(
    columnorder = [1,2,3,4],
    columnwidth = [400,400,400,400],
    header = dict(
        values = [['Casos'],
                    ['Número'],['INFECTADOS'],
                    ['Número']],
        line_color='darkslategray',
        fill_color='gray',
        align=['center'],
        font=dict(color='white', size=12),
        height=30
    ),
    cells=dict(
        values=valuestable,
        line_color='darkslategray',
        fill=dict(color=['lightgray', 'white','lightgray', 'white']),
        align=['center'],
        font_size=12,
        height=30)
        )
    ])

    return figure,figure0,fig

if __name__ == '__main__':
    app.run_server(debug=True)
