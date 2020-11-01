import pandas as pd
import numpy as np
from datetime import timedelta

class CovidData:
    def __init__(self, covid_data: pd.DataFrame):
        self.covid_data = covid_data
        self.d_hat = None
        self.w_hat = None

    
    def preprocessing_data(self):
        self.rename_columns()
        self.standarize_values()
        self.dates_to_datetime()
        self.get_delay()
        self.deal_asymptomatic()
        self.deal_deaths()
        self.assign_recovery_date(9)
        self.get_recovery_days()
        self.covid_data.drop(columns=['dummy_date'])


    def rename_columns(self):
        '''Renombre las columnas para mejorar legibilidad.'''
        rename_dict = {
            'id_de_caso': 'id',
            'fecha_de_notificaci_n': 'fecha_notificacion',
            'ciudad_municipio_nom': 'municipio',
            'departamento_nom': 'departamento',
            'recuperado': 'atencion',
            'edad': 'edad',
            'sexo': 'sexo',
            'fuente_tipo_contagio': 'tipo_contagio',
            'estado': 'estado_salud',
            'pais_viajo_1_nom': 'pais_procedencia',
            'fecha_inicio_sintomas': 'fecha_sintomas',
            'fecha_muerte': 'fecha_muerte',
            'fecha_diagnostico': 'fecha_diagnostico',
            'fecha_recuperado': 'fecha_recuperacion',
            'fecha_reporte_web': 'fecha_reporte'
        }
        self.covid_data = self.covid_data.rename(columns=rename_dict)


    def standarize_values(self):
        '''Cambia los valores de las columas indicadas a "formato título".'''
        columnas_corregir = ['municipio', 'departamento', 'atencion', 'sexo',
                            'tipo_contagio', 'estado_salud', 'pais_procedencia']
        for col in columnas_corregir:
            self.covid_data[col] = self.covid_data[col].fillna('-').apply(lambda x: x.title())


    def dates_to_datetime(self):
        '''Convierte las columnas de fechas de covid_data a datetime.

        Fechas con valores '-   -' o 'Asintomático' son reemplazados
        por np.datetime64('NaT')
        '''
        fechas = ['fecha_notificacion', 'fecha_diagnostico', 'fecha_sintomas', 
                'fecha_muerte', 'fecha_recuperacion', 'fecha_reporte']

        for fecha in fechas:
            self.covid_data[fecha] = self.covid_data[fecha].replace(['-   -', 'Asintomático', 'asintomático'], np.datetime64('NaT'))
            try:
                self.covid_data[fecha] = pd.to_datetime(self.covid_data[fecha], dayfirst=True).dt.normalize()
            except Exception as e:
                print('Hay una fecha en formato incorrecto: ', e)
                self.covid_data[fecha] = pd.to_datetime(self.covid_data[fecha], errors='coerce')

    
    def get_delay(self):
        '''Agrega la columna 'dias_retraso', que consiste en el número 
        de días entre fecha_sintomas y fecha_reporte.
        '''
        self.covid_data['dias_retraso'] = (self.covid_data['fecha_reporte'] - self.covid_data['fecha_sintomas']).apply(lambda x: x.days) 
        self.w_hat = self.covid_data['dias_retraso'].median(skipna=True)


    def deal_asymptomatic(self):
        '''Asigna valor a fecha_sintomas a los infectados que no la tienen

        Para los infectados sin fecha de síntomas, se les asigna un valor
        basándose en la estimación del Instituto Robert Koch de Alemania.
        Para más información ver el documento: 
        '''
        departamentos = self.covid_data['departamento'].unique()
        # Condición 0: el infectado es asintomático
        cond0 = self.covid_data['fecha_sintomas'].isna()
        for dpto in departamentos:
            # Condición 1: el departamento es 'dpto'
            cond1 = self.covid_data['departamento'] == dpto
            if self.covid_data[cond0 & cond1].empty:
                continue
            df_filter = self.covid_data[(cond1)]
            n, w_raw = df_filter['dias_retraso'].count(), df_filter['dias_retraso'].median(skipna=True)
            if n >= 20:
                w = w_raw
            elif n == 0:
                w = self.w_hat
            else:
                w = w_raw * n / 20 + self.w_hat * (20-n) / 20
            self.covid_data['dummy_date'] = self.covid_data['fecha_reporte'] - timedelta(days=int(w))
            self.covid_data.loc[cond0 & cond1, 'fecha_sintomas'] = self.covid_data.loc[(cond0) & (cond1), ['fecha_recuperacion', 'dummy_date', 'fecha_muerte']].min(axis=1, skipna=True) 


    def deal_deaths(self):
        '''Para los fallecidos, se asigna su fecha de recuperación 
        como su fecha de muerte.
        '''

        cond = self.covid_data['fecha_muerte'].isna()
        self.covid_data.loc[~cond, 'fecha_recuperacion'] = \
            self.covid_data[~cond]['fecha_muerte']


    def assign_recovery_date(self, rd=13):
        '''Asigna una fecha de recuperación asumiendo que todos
        los pacientes tardan máximo rd días en recuperarse
        '''
        limit_date = pd.to_datetime('now') - timedelta(hours=(24*rd + 5))
        cond1 = self.covid_data['fecha_sintomas'] < limit_date
        cond2 = self.covid_data['fecha_muerte'].isna()

        self.covid_data['dummy_date'] = self.covid_data['fecha_sintomas'] + timedelta(days=rd)
        self.covid_data.loc[cond1 & cond2,'fecha_recuperacion'] = self.covid_data[cond1 & cond2][['fecha_recuperacion', 'dummy_date']].min(axis=1, skipna=True)


    def get_recovery_days(self):
        ''' Agrega la columna 'días', que consiste en el número
        de días que una persona duró infectada.
        '''
        self.covid_data['dias'] = (self.covid_data['fecha_recuperacion'] - self.covid_data['fecha_sintomas']).apply(lambda x: x.days)
        self.d_hat = self.covid_data['dias'].median(skipna=True)
