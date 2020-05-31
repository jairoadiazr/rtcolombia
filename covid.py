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
        self.assign_recovery_date()
        self.get_recovery_days()


    def rename_columns(self):
        '''Renombre las columnas para mejorar legibilidad.'''
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
                self.covid_data[fecha] = pd.to_datetime(self.covid_data[fecha])
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
            df_filter = self.covid_data[(cond1)]
            n, w_raw = df_filter['dias_retraso'].count(), df_filter['dias_retraso'].median(skipna=True)
            if n >= 20:
                w = w_raw
            elif n == 0:
                w = self.w_hat
            else:
                w = w_raw * n / 20 + self.w_hat * (20-n) / 20
            self.covid_data.loc[(cond0) & (cond1), 'fecha_sintomas'] = self.covid_data[(cond0) & (cond1)]['fecha_reporte'] - timedelta(days=w)


    def assign_recovery_date(self): 
        '''Para los fallecidos, se asigna su fecha de recuperación 
        como su fecha de muerte.
        '''
        self.covid_data.loc[self.covid_data['estado_salud'] == 'Fallecido', 'fecha_recuperacion'] = \
            self.covid_data[self.covid_data['estado_salud'] == 'Fallecido']['fecha_muerte']

    
    def get_recovery_days(self):
        ''' Agrega la columna 'días', que consiste en el número
        de días que una persona duró infectada.
        '''
        self.covid_data['dias'] = (self.covid_data['fecha_recuperacion'] - self.covid_data['fecha_sintomas']).apply(lambda x: x.days)
        self.d_hat = self.covid_data['dias'].median(skipna=True)