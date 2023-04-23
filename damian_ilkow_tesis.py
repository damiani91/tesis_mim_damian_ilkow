# Codigo para cargar los datasets utilizados
import os
import datetime
import random
import time

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import uniform
from scipy.stats import randint

import hashlib 

from sklearn import set_config
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
import warnings
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc
from lifelines import KaplanMeierFitter, CoxPHFitter
from sksurv.preprocessing import OneHotEncoder
from xgbse.converters import convert_to_structured

################ 01 - LOAD ####################
# funcion creada para cargar cada uno de los datasets.
def data_load_riders(path):
    # creamos el parh que contiene los datasets
    dataset_path = path + '/datasets'
    
    # lista que contiene cada uno de los nombre de los datasets
    datasets = os.listdir(dataset_path)
    
    # creamos una lista vacÃ­a para almacenar todos los datasets
    df_list = []
    
    # iteramos sobre cada uno de los datasets y los appendeamos
    for file in datasets:
        dataframe = pd.read_csv(dataset_path + '/' + file, index_col = None, header = 0)
        df_list.append(dataframe)
    
    # concatenamos todos los datasets para unirlos en uno
    df = pd.concat(df_list, axis = 0, ignore_index = True)
    
    # cambiamos el tipo de dato de las columnas year y week
    df['year'] = df['year'].astype(int)
    df['week'] = df['week'].astype(int)
    
    # cargamos un dataset original con la fecha del primer envío de repartidores para calcular antiguedad
    df_fo = pd.read_csv(path + '/dataset_adicional/rider_fisrt_order.csv')
    
    # traemos la primera orden para cada repartidor
    df = df.merge(df_fo, on = 'rider_id')
    
    # anonimizamos la información sensible, relacionada al rider_id, luego de cambiar su tipo
    df['rider_id'] = df['rider_id'].astype(str)
    df['city_id'] = df['city_id'].astype(str)
    df['rider_id'] = [hashlib.sha256(x.encode()).hexdigest() for x in df['rider_id']]
    
    # eliminamos la columna first_order por poseer información incorrecta
    df.drop(columns = 'first_order', inplace = True)
    
    
    return df

################ 02 - CLEANING & ANÁLISIS EXPLORATORIO DE DATOS ####################
def transform_data (df):
    
    # usar métodos para describir el df
    print(df.shape)
    print(df.describe().transpose())
    print(df.info())
    
    # graficamos la media de órdenes para entender si detectamos algún valor extraño rápidamente
    graph_oders = df.groupby(['year', 'week'], as_index = False).agg({'completed_deliveries' : np.mean})
    fig, axs = plt.subplots(3, sharex = True, figsize=(10, 10))
    axs[0].plot(graph_oders[graph_oders['year'] == 2020]['week'], graph_oders[graph_oders['year'] == 2020]['completed_deliveries'], linewidth = 4, color = 'red')
    axs[1].plot(graph_oders[graph_oders['year'] == 2021]['week'], graph_oders[graph_oders['year'] == 2021]['completed_deliveries'], linewidth = 4, color = 'blue')
    axs[2].plot(graph_oders[graph_oders['year'] == 2022]['week'], graph_oders[graph_oders['year'] == 2022]['completed_deliveries'], linewidth = 4, color = 'green')
    axs[0].title.set_text('year 2020')
    axs[1].title.set_text('year 2021')
    axs[2].title.set_text('year 2022')
    plt.title('Media de órdenes según semana del año', loc = 'right')
    plt.show()
    
    # quitamos la semana 0 de los dos años y la máxima semana del año 2022
    df = df[df['week'] != 0]
    df = df[~((df['week'] == 53) & (df['year'] == 2021))]
    df = df[~((df['week'] == 52) & (df['year'] == 2022))]
    max_2022_week = max(df[df['year'] == 2022]['week'])
    df = df[~((df['week'] == max_2022_week) & (df['year'] == 2022))]
    
    # quitamos todas las filas que contengan valores de completed_orders iguales a cero
    # porque solo vamos a considerar riders activos
    df = df[df['completed_deliveries'] > 0]
    
    # entendemos la cantidad de valores Na que posee el DF 
    na_values = df.isna().sum().reset_index(name = "variable").sort_values(by = 'variable', ascending = False)
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.barh(na_values['index'], 
            na_values['variable'])
    plt.title('Cantidad de Na Values para cada variable')
    plt.show()
    
    # quitamos del df las observaciones que posean NA value para batch number porque no podemos determinar
    # a qué batch pertenecen esos riders
    df = df[df['batch_number'].notna()]
    
    # reemplazo los valores de batch 7 y 8 por 5 y 6 (ya que por decisiones del negocio, son iguales)
    df.loc[df['batch_number'] == 7, 'batch_number'] = 5
    df.loc[df['batch_number'] == 8, 'batch_number'] = 6
    
    # converitr batch_number a str
    df.loc[:,'batch_number'] = df.loc[:,'batch_number'].astype(str)
    
    # reemplazamos Na Values por el valor cero (son valores nulos que deberían ser cero)
    df.fillna(0, inplace = True)
    
    # convertir semana año a día para luego calcular la antiguedad en días para cada repartidor
    df['year_week'] = df['year'].astype(str) + df['week'].astype(str) + '0'
    df['year_week_day'] = pd.to_datetime(df['year_week'], format='%Y%U%w')
    
    print(df.shape)
    print(df.describe().transpose())
    print(df.info())
    
    return df
    
def set_y_label(df):
    # filtramos para cada rider_id
    df.sort_values(by = ['rider_id', 'year', 'week'], ascending = True, inplace = True)
    
    # creamos las columnas auxiliares que nos servirán para determinar
    compare_year = df['year'][1:].tolist()
    compare_year.append(0)
    compare_week = df['week'][1:].tolist()
    compare_week.append(0)
    compare_rider_id = df['rider_id'][1:].to_list()
    compare_rider_id.append(0)
    
    # agregamos las columnas al df original
    df.loc[:,'compare_year'] = compare_year
    df.loc[:,'compare_week'] = compare_week
    df.loc[:,'compare_rider_id'] = compare_rider_id
    
    # definimos el label churn / nochurn
    # condicion: más de cuatro semanas sin repartir
    df.loc[
        ((df['rider_id'] == df['compare_rider_id']) &
        (((df['year'] == df['compare_year']) & (df['compare_week'] - df['week'] > 4))
            | (((df['compare_year'] - df['year']) == 1) & ((df['compare_week'] - df['week']) > (-48)))
            | ((df['compare_year'] == 0) & (df['compare_week'] == 0) & (df['year'] < 2022)))),
        'churn_condition'] = 'churn'
    # condicion dos: máximo año en que repartió
    df.loc[(df['rider_id'] != df['compare_rider_id']) & (df['year'] < 2022), 'churn_condition'] = 'churn'
    # condicion tres: máximo semana del año en curso en que repartió
    df.loc[(df['rider_id'] != df['compare_rider_id']) & ((df['year'] == 2022) & (df['week'] <= (max(df[df['year'] == 2022]['week']) - 4))), 'churn_condition'] = 'churn'
    
    # definición de no_chun como contrapartida a churn
    df.loc[df['churn_condition'] != 'churn', 'churn_condition'] = 'no_churn'
    
    # eliminamos las columnas auxiliares
    df.drop(['compare_year', 'compare_week', 'compare_rider_id'], axis = 1, inplace = True)
            
    return df

def data_analysis(df):
       
    # graficamos la cantidad de datos para cada label
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.barh(df.groupby('churn_condition').agg({'rider_id' : 'count'}).index,
            df.groupby('churn_condition').agg({'rider_id' : 'count'})['rider_id'], color = ['#00337C', '#8EA7E9'])
    #ax.label('')
    plt.title('Cantidad de observaciones para cada label')
    plt.show()
    
    # acomodamos el dataset para crear la variable churn ratio
    churn_ratio = df.groupby(['year','week','churn_condition'], as_index = False).agg({'rider_id' : 'count'})
    churn_ratio.loc[churn_ratio['churn_condition'] == 'churn','churn'] = churn_ratio['rider_id']
    churn_ratio.loc[churn_ratio['churn_condition'] != 'churn','no_churn'] = churn_ratio['rider_id']
    churn_ratio = churn_ratio.groupby(['year', 'week'], as_index = False).agg({'churn' : max, 'no_churn' : max})
    # creamos la variable churn ratio como cantidad de churn sobre cantidad de no churn
    churn_ratio.loc[:,'churn_ratio'] = churn_ratio['churn'] / churn_ratio['no_churn']
    # filtramos las semanas que no poseen churn ratio porque no pudimos definir la cantidad de riders churn
    churn_ratio = churn_ratio[churn_ratio['churn_ratio'].notnull()]
    
    # graficamos la evolución del churn ratio para cada año
    fig, axs = plt.subplots(3, sharex = True, figsize=(10, 10))
    axs[0].plot(churn_ratio[churn_ratio['year'] == 2020]['week'], churn_ratio[churn_ratio['year'] == 2020]['churn_ratio'], linewidth = 4, color = '#ff6961')
    axs[1].plot(churn_ratio[churn_ratio['year'] == 2021]['week'], churn_ratio[churn_ratio['year'] == 2021]['churn_ratio'], linewidth = 4, color = '#84b6f4')
    axs[2].plot(churn_ratio[churn_ratio['year'] == 2022]['week'], churn_ratio[churn_ratio['year'] == 2022]['churn_ratio'], linewidth = 4, color = '#77dd77')
    axs[0].title.set_text('year 2020')
    axs[1].title.set_text('year 2021')
    axs[2].title.set_text('year 2022')
    plt.title('Churn ratio según semana del año', loc = 'right')
    plt.show()
    
    # quitamos las semanas anteriores anteriores a la semana 11 del 2020 porque el efecto de la pandemia
    # y quitar una sola semana nos alteraría el resto de los valores (en el gráfico se ve nítidamente)
    df = df.loc[~((df['year'] == 2020) & (df['week'] < 11)),:]
    
    # graficamos la evolución del churn ratio para cada país
    stacked_graph = pd.crosstab(index = df['country_code'].astype(str),
                                columns = df['churn_condition'],
                                normalize = "index")
    #graficamos el stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 10))
    stacked_graph.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab10')
    plt.legend(loc = "upper left", ncol = 2)
    plt.xlabel("Country Code")
    plt.ylabel("Proportion Churn / No Churn")
    plt.title('churn proportion by Country Code')
    #plt.show()
    

    # crear paleta para seaborn
    my_palet = sns.color_palette("pastel")
    # grficamos un boxplot de los valores de órdenes para churn y no churn
    sns.boxplot(x = df['churn_condition'], y = df['completed_deliveries'], palette = my_palet)
    plt.title('Boxplot de órdenes para cada tipo de label')
    #plt.show()
    
    # graficamos un boxplot de los valores de working hours para churn y no churn
    sns.boxplot(x = df['churn_condition'], y = df['working_hours'], palette = my_palet)
    plt.title('Boxplot de working hours según cada tipo de label')
    #plt.show()
    
    # graficamos barras apiladas al 100% para entender la proporción de churn para cada número de batch
    # armamos la estructura de datos a graficar
    stacked_graph_batch = pd.crosstab(index = df['batch_number'].astype(str),
                                columns = df['churn_condition'],
                                normalize = "index")
    #graficamos el stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 10))
    stacked_graph_batch.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab10')
    plt.legend(loc = "upper left", ncol = 2)
    plt.xlabel("Batch Number")
    plt.ylabel("Proportion Churn / No Churn")
    plt.title('churn proportion by batch number')
    #plt.show()
    
    
    # scatterplot con working hours y órdenes
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.lmplot(y = 'completed_deliveries', 
               x = 'working_hours', 
               data = df, 
               fit_reg = False,
               hue = 'churn_condition', 
               legend = False)
    plt.legend(loc = 'upper left')
    plt.title('scatterplot para órdenes y hours según variable dependiente')
    plt.show()
      
    # Necesito un gráfico para UTR
    
        
    
    return df
    

################ 03 - FEATURE ENGINEERING ####################
def feature_engineering (df):
    
    # creamos una variable que sea la antiguedad
    first_order = df.groupby('rider_id', as_index = False).agg({'year_week_day' : min})
    first_order.rename(columns = {'year_week_day' : 'first_order'}, inplace = True)
    # calculamos la antiguedad en semana de los repartidores para cada semana
    df = df.merge(first_order, on = 'rider_id')
    df['seniority_week'] = abs((df['year_week_day'].astype('datetime64[ns]') + datetime.timedelta(days = 6) - df['first_order'].astype('datetime64[ns]')).dt.days // 7)
     
    
    # creamos otra variable donde realizamos la suma cuadrática de la cantidad de órdenes y la working hours
    df.loc[:,'sqr_orders_working_hours'] = df['completed_deliveries']**2 + df['working_hours']**2
    # creamos otra variable donde realizamos la suma cuadrática de la cantidad de órdenes y la antiguedad
    df.loc[:,'sqr_orders_seniority'] = df['completed_deliveries']**2 + df['seniority_week']**2
    # creamos una variables que sea el log de las dos anteriores (porque quedan con cola pesada)
    # utilizando el LOG, la distribución de asemeja más a una normal, favoreciendo el trabajo de los Algoritmos.
    df.loc[:,'log_sqr_orders_working_hours'] = np.log(df['sqr_orders_working_hours'])
    df.loc[:,'log_sqr_orders_seniority'] = np.log(df['sqr_orders_seniority'])
    
    # Creamos nuevas variables con los componentes principales
        # Estandarizamos los valores del df para obtener mejores resultados en PCA
    std_scaler = preprocessing.StandardScaler()
    standarized_df = std_scaler.fit_transform(df[list(df.select_dtypes(include = float))])
        # Generamos las componentes principales
    pca = PCA(n_components = 3)
    pca.fit(standarized_df)
    #pca.fit(df[list(df.select_dtypes(include = float))])
        # incorporamos los valores de las componentes principales al df
    df.loc[:,'pca_1'] = pd.DataFrame(pca.transform(standarized_df))[0]
    df.loc[:,'pca_2'] = pd.DataFrame(pca.transform(standarized_df))[1]
    df.loc[:,'pca_3'] = pd.DataFrame(pca.transform(standarized_df))[2]
        # analizamos la proporción de la varianza 
    explained_variance = round(sum(pca.explained_variance_ratio_), 4) * 100
    print('PCA de tres componentes, explica el ', explained_variance, '% de la variación de la muestra')
    
    # Estandarizar valores de algunas variables (MinMax Scaler)
    min_max_scaler = preprocessing.MinMaxScaler()
    df[['minmax_deliveries', 'minmax_working_hours', 'minmax_cancelled_deliveries']] = min_max_scaler.fit_transform(df[['completed_deliveries', 'working_hours', 'cancelled_deliveries']])
    
    
    # creamos una nueva variable que sea la variación semana a semana de la cantidad de órdenes
    compare_id = df['rider_id'][:-1].tolist()
    compare_id = [0] + compare_id
    df.loc[:,'compare_id'] = compare_id
    compare_deliveries = df['completed_deliveries'][:-1].tolist()
    compare_deliveries = [0] + compare_deliveries
    df.loc[:,'compare_deliveries'] = compare_deliveries
    compare_hours = df['working_hours'][:-1].tolist()
    compare_hours = [0] + compare_hours
    df.loc[:,'compare_hours'] = compare_hours
    compare_utr = df['utr'][:-1].tolist()
    compare_utr = [0] + compare_utr
    df.loc[:,'compare_utr'] = compare_utr
    compare_churn = df['churn_condition'][:-1].tolist()
    compare_churn = [0] + compare_churn
    df.loc[:,'compare_churn'] = compare_churn
    # definimos el label churn / nochurn
    # condicion uno: diferencia de días entre órdenes
    df.loc[((df['compare_churn'] == 'no_churn') & (df['rider_id'] == df['compare_id'])), 'orders_variation'] = (df['completed_deliveries'] / df['compare_deliveries']) - 1
    df.loc[((df['compare_churn'] == 'no_churn') & (df['rider_id'] == df['compare_id'])), 'hours_variation'] = (df['working_hours'] / df['compare_hours']) - 1
    df.loc[((df['compare_churn'] == 'no_churn') & (df['rider_id'] == df['compare_id'])), 'utr_variation'] = (df['utr'] / df['compare_utr']) - 1
    df.loc[(df['orders_variation']).isna(),'orders_variation'] = 999
    df.loc[(df['hours_variation']).isna(),'hours_variation'] = 999
    df.loc[(df['hours_variation'] == np.inf),'hours_variation'] = 999
    df.loc[(df['utr_variation']).isna(),'utr_variation'] = 999
    df.loc[(df['utr_variation'] == np.inf),'utr_variation'] = 999
    # eliminamos las columnas auxiliares utilizadas para los cálculos
    df.drop(columns = ['compare_churn', 'compare_deliveries', 'compare_id', 'compare_hours'], inplace = True)
    
    
    # creamos una nueva variable que sea el decil que ocupa un repatidor/a en para una ciudad y
    # un período de tiempo determinado
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_
    
    percentiles = df.groupby(['year','week', 'city_id'], as_index = False).agg(percentile_10 = ('completed_deliveries', percentile(10)),
                                                                 percentile_20 = ('completed_deliveries', percentile(20)),
                                                                 percentile_30 = ('completed_deliveries', percentile(30)),
                                                                 percentile_40 = ('completed_deliveries', percentile(40)),
                                                                 percentile_50 = ('completed_deliveries', percentile(50)),
                                                                 percentile_60 = ('completed_deliveries', percentile(60)),
                                                                 percentile_70 = ('completed_deliveries', percentile(70)),
                                                                 percentile_80 = ('completed_deliveries', percentile(80)),
                                                                 percentile_90 = ('completed_deliveries', percentile(90)))
    
    df = df.merge(percentiles, on = ['year', 'week', 'city_id'])
    
    df.loc[df['completed_deliveries'] <= df['percentile_10'], 'decil'] = 1
    df.loc[(df['completed_deliveries'] <= df['percentile_20']) & (df['decil'].isna()), 'decil'] = 2
    df.loc[(df['completed_deliveries'] <= df['percentile_30']) & (df['decil'].isna()), 'decil'] = 3
    df.loc[(df['completed_deliveries'] <= df['percentile_40']) & (df['decil'].isna()), 'decil'] = 4
    df.loc[(df['completed_deliveries'] <= df['percentile_50']) & (df['decil'].isna()), 'decil'] = 5
    df.loc[(df['completed_deliveries'] <= df['percentile_60']) & (df['decil'].isna()), 'decil'] = 6
    df.loc[(df['completed_deliveries'] <= df['percentile_70']) & (df['decil'].isna()), 'decil'] = 7
    df.loc[(df['completed_deliveries'] <= df['percentile_80']) & (df['decil'].isna()), 'decil'] = 8
    df.loc[(df['completed_deliveries'] <= df['percentile_90']) & (df['decil'].isna()), 'decil'] = 9
    df.loc[(df['completed_deliveries'] > df['percentile_90']) & (df['decil'].isna()), 'decil'] = 10
    
    df.drop(columns = ['percentile_10',
                       'percentile_20',
                       'percentile_30',
                       'percentile_40',
                       'percentile_50',
                       'percentile_60',
                       'percentile_70',
                       'percentile_80',
                       'percentile_90'], inplace = True)
    
    
    # generamos el label 1 y 0 para churn == 1 / no_churn == 0
    df['binary_churn_label'] = [1 if x == 'churn' else 0 for x in df['churn_condition']]
    
    """
    # calculamos la media móvil para las últimas tres semanas para cada rider
    rolling_average = []
    test = df[['year', 'week', 'rider_id', 'completed_deliveries']].to_numpy()
    
    for i in tqdm.tqdm(test['rider_id'].unique().tolist()):
        aux_test = test.loc[test['rider_id'] == i, :]
        rollin_avg = aux_test['completed_deliveries'].rolling(3, min_periods=1).mean().tolist()
        rolling_average = rolling_average + rollin_avg
    
    df['rolling_average'] = rolling_average
    
    rolling_average = [test.loc[test['rider_id'] == i, 'completed_deliveries'].rolling(3, min_periods=1).mean().tolist() for i in tqdm.tqdm(test['rider_id'].unique().tolist())]
    """
    
    return df


def data_analysis_feature_engineering(df): 
    
    # crear paleta para seaborn
    my_palet = sns.color_palette("pastel")
    # graficamos un boxplot de los valores de working hours para churn y no churn
    sns.boxplot(x = df['churn_condition'], y = df['seniority_week'], palette = my_palet)
    plt.title('Boxplot de seniority (weeks) según cada tipo de label')
    plt.show()
    
    # graficamos barras apiladas al 100% para entender la proporción de churn para cada antiguedad
    # armamos la estructura de datos a graficar
    stacked_graph_2 = pd.crosstab(index = df['seniority_week'].astype(int),
                                columns = df['churn_condition'],
                                normalize = "index")
    stacked_graph_2.loc[:,'churn_ratio'] = stacked_graph_2['churn'] / stacked_graph_2['no_churn']
    #graficamos el stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 10))
    stacked_graph_2[['churn', 'no_churn']].plot(kind = 'area',
                         stacked = True,
                        colormap = 'tab10')
    plt.legend(loc = "upper left", ncol = 2)
    plt.xlabel("seniority")
    plt.ylabel("Proportion Churn / No Churn")
    plt.title('proporción de churn según antiguedad')
    plt.show()
    #graficamos el stacked line plot churn ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(stacked_graph_2['churn_ratio'][:130], color = '#F94892', linewidth = 2)
    plt.xlabel("seniority")
    plt.ylabel("churn ratio")
    plt.title('churn ratio según antiguedad')
    plt.show()
    
    # scatterplot con working hours y seniority
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.lmplot(y = 'working_hours', 
               x = 'seniority_week', 
               data = df, 
               fit_reg = False,
               hue = 'churn_condition', 
               legend = False)
    plt.legend(loc = 'upper left')
    plt.title('scatterplot para órdenes y antiguedad según variable dependiente')
    plt.show()
    
    # histograma para churn y no churn según antiguedad
    fig, axs = plt.subplots(2, figsize=(10, 10))
    axs[0].hist(df[df['churn_condition'] == 'churn']["seniority_week"], color = "#B5EAEA", bins = 150)
    axs[0].vlines(np.median(df[df['churn_condition'] == 'churn']["seniority_week"]), color = '#3AB4F2', ymin = 0, ymax = 19000)
    axs[1].hist(df[df['churn_condition'] != 'churn']["seniority_week"], color = "#F38BA0", bins = 150)
    axs[1].vlines(np.median(df[df['churn_condition'] != 'churn']["seniority_week"]), color = '#E64848', ymin = 0, ymax = 150000)
    axs[0].title.set_text('Churn')
    axs[1].title.set_text('No Churn')
    plt.title('histograma antiguedad según y-label', loc = 'right')
    plt.show()
    
        # scatterplot para componenetes principales
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.lmplot(y = 'pca_1', 
               x = 'pca_2', 
               data = df, 
               fit_reg = False,
               hue = 'churn_condition', 
               legend = False)
    plt.legend(loc = 'upper left')
    plt.title('scatterplot para pcas según variable dependiente')
    plt.show()
    
    
    # Separación sobre log sqr orders seniority
    plt.rcParams["figure.figsize"] = 12, 8
    sns.histplot(x = np.log(df[df['churn_condition'] == 'churn']['sqr_orders_seniority']), stat = "density", bins = 100, edgecolor = 'black')
    heights, bins = np.histogram(np.log(df[df['churn_condition'] != 'churn']['sqr_orders_seniority']), density = True, bins = 100) 
    heights *= -1
    bin_width = np.diff(bins)[0]
    bin_pos =( bins[:-1] + bin_width / 2) * -1
    plt.bar(bin_pos, heights, width = bin_width, edgecolor = 'black')
    plt.title('distribución log sqr orders seniority')
    plt.legend(['churn', 'no churn'])
    plt.show()
    
    # Separación sobre log sqr orders working hours
    plt.rcParams["figure.figsize"] = 12, 8
    sns.histplot(x = np.log(df[df['churn_condition'] == 'churn']['sqr_orders_working_hours']), stat = "density", bins = 100, edgecolor = 'black')
    heights, bins = np.histogram(np.log(df[df['churn_condition'] != 'churn']['sqr_orders_working_hours']), density = True, bins = 100) 
    heights *= -1
    bin_width = np.diff(bins)[0]
    bin_pos =( bins[:-1] + bin_width / 2) * -1
    plt.bar(bin_pos, heights, width = bin_width, edgecolor = 'black')
    plt.title('distribución log sqr orders working hours')
    plt.legend(['churn', 'no churn'])
    plt.show()
    
    # graficamos la distribución de variación órdenes para cada condición de churn
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.kdeplot(x = df[(df['churn_condition'] == 'churn') & (df['orders_variation'] < 4)]['orders_variation'], shade = True, color = 'r')
    sns.kdeplot(x = df[(df['churn_condition'] != 'churn') & (df['orders_variation'] < 4)]['orders_variation'], shade = True, color = 'b')
    plt.title('distribución variación de órdenes según condición de churn')
    plt.legend(['churn', 'no churn'])
    plt.show()
    
    # graficamos la distribución de variación working hours para cada condición de churn
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.kdeplot(x = df[(df['churn_condition'] == 'churn') & (df['utr_variation'] < 4)]['utr_variation'], shade = True, color = 'r')
    sns.kdeplot(x = df[(df['churn_condition'] != 'churn') & (df['utr_variation'] < 4)]['utr_variation'], shade = True, color = 'b')
    plt.title('distribución variación de utr según condición de churn')
    plt.legend(['churn', 'no churn'])
    plt.show()
    
    # graficamos la distribución de variación órdenes para cada condición de churn
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.kdeplot(x = df[(df['churn_condition'] == 'churn') & (df['hours_variation'] < 4)]['hours_variation'], shade = True, color = 'r')
    sns.kdeplot(x = df[(df['churn_condition'] != 'churn') & (df['hours_variation'] < 4)]['hours_variation'], shade = True, color = 'b')
    plt.title('distribución variación de working hours según condición de churn')
    plt.legend(['churn', 'no churn'])
    plt.show()
    
    # graficamos barras apiladas al 100% para entender la proporción de churn para cada decil de órdenes
    # armamos la estructura de datos a graficar
    stacked_graph_decil = pd.crosstab(index = df['decil'].astype(int),
                                columns = df['churn_condition'],
                                normalize = "index")
    #graficamos el stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 10))
    stacked_graph_decil.plot(kind='bar', 
                        stacked=True, 
                        colormap='tab10')
    plt.legend(loc = "upper left", ncol = 2)
    plt.xlabel("Decile Number")
    plt.ylabel("Proportion Churn / No Churn")
    plt.title('churn proportion by order decil')
    plt.show()
    
    pass

################ 04 - CLASS BALANCING ####################
# definimos algún método para eliminar ciertas observaciones y compensar el desbalance de clases
def class_balance(df):
    # TESTEANDO
    # agrupo para eliminar aquellos que nunca hicieron churn a lo largo de su lifecycle
    riders_to_delete = df.groupby('rider_id', as_index = False).agg({'churn_condition' : min})
    riders_to_delete = riders_to_delete[riders_to_delete['churn_condition'] == 'no_churn']
    # eliminamos los riders que nunca hicieron churn
    df = df[~df['rider_id'].isin(riders_to_delete['rider_id'])]
    
    # Calculamos el ratio de semanas en churn sobre no churn, con el objetivo de eliminar
    # aquellas observaciones que se encuentran sobrerepresentadas en churn
    rep = df.groupby(['rider_id', 'churn_condition'], as_index = False).agg({'country_code' : 'count'})
    #  
    rep = pd.crosstab(rep.rider_id, rep.churn_condition, values = rep.country_code, aggfunc = 'sum')
    # calculamos la proporción de churn/no_churn
    rep['prop_to_delete'] = rep['churn'] / rep['no_churn']
    # eliminamos los datos vacíos
    #rep.dropna(inplace = True)
    rep.loc[rep['prop_to_delete'].isna(), 'prop_to_delete'] = 1
    # definimos aquellos riders a eliminar por tener sobrerepresentación
    # entendemos por sobrerepresentación a una muy mayor cantidad de observaciones para una clase
    # por sobre otra
    rep.loc[rep['prop_to_delete'] > np.quantile(rep['prop_to_delete'], 0.2) , 'delete'] = 0
    rep.loc[rep['delete'] != 0, 'delete'] = 1
    # llevamos los valores al df original
    df = pd.merge(df, rep['delete'], on = 'rider_id')
    # nos quedamos únicamente con aquellos que cumplen la condición de corte según prop_to_delete
    df = df.loc[df['delete'] == 0, :]

    return df


################ 05 - TEST / TRAIN SEPARATION ####################
def test_train_separation(df):
    
    # seteamos una semilla para poder replicar los resultados en el futuro
    np.random.seed(777)
    # decido entrenar el modelo con el 95% de los rider_id ya que me genera suficiente cantidad de observaciones
    # como para testear el modelo e igual mantengo la cantidad de observaciones 
    # selecciono los ids para cada caso
    #train_ids = random.choices(df['rider_id'].unique(), k = round(len(df['rider_id'].unique()) * 0.95))
    train_ids = random.sample(df['rider_id'].unique().tolist(), k = round(len(df['rider_id'].unique()) * 0.95))
    test_ids = df[~df['rider_id'].isin(train_ids)]['rider_id'].unique().tolist()
    
    # verificamos la proporción de observaciones con cada label para train y test
    print('la proporción de de churn para train, es: ', sum(df[df['rider_id'].isin(train_ids)]['binary_churn_label']) / len(df[df['rider_id'].isin(train_ids)]['binary_churn_label']))
    print('la proporción de de churn para test, es: ', sum(df[df['rider_id'].isin(test_ids)]['binary_churn_label']) / len(df[df['rider_id'].isin(test_ids)]['binary_churn_label']))
    
    # creamos un index que contenga las variables year, week y id para ganar trazabilidad en el dataset
    df['new_index'] = df['year'].astype(str) + df['week'].astype(str) + df['rider_id'].astype(str)
    df.set_index('new_index', inplace = True)
    
    # realizamos one hot encoder sobre variables categóricas
    one_hot_encode_country = pd.get_dummies(df['country_code'], sparse = False)
    one_hot_encode_city = pd.get_dummies(df['city_id'], sparse = False)
    one_hot_encode_batch_number = pd.get_dummies(df['batch_number'], sparse = False)
    
    # agregamos las columnas One Hot Encode 
    df = df.join(one_hot_encode_country)
    df = df.join(one_hot_encode_city)
    df = df.join(one_hot_encode_batch_number)
    
    # creamos los dataset para x_train e y_train
    x_train = df.drop(columns = ['year','year_week_day', 'year_week', 'year_week', 'week', 'first_order', 'churn_condition', 'binary_churn_label', 'country_code', 'city_id', 'rider_id', 'batch_number']).loc[df['rider_id'].isin(train_ids)]
    y_train = df.loc[df['rider_id'].isin(train_ids), 'binary_churn_label']
    # convertimos el x_train en una sparse matrix
    
    # creamos los dataset para x_train e y_train
    x_test = df.drop(columns = ['year','year_week_day', 'year_week', 'year_week', 'week', 'first_order', 'churn_condition', 'binary_churn_label', 'country_code', 'city_id', 'rider_id', 'batch_number']).loc[df['rider_id'].isin(test_ids)]
    y_test = df.loc[df['rider_id'].isin(test_ids), 'binary_churn_label']
    
    return x_train, y_train, x_test, y_test


################ 06 - CREACIÓN Y MODELOS DE BENCHMARK ####################
def benchmark_logistic_model(x_train, y_train, x_test, y_test):
    
    random.seed(777)
    # Entrenamos un modelos de Regresi�n Log�stica para utilizar como Benchmark
    clf_log = LogisticRegression(random_state = 37).fit(x_train, y_train)
    y_train_predict_proba = clf_log.predict_proba(x_train)[:,1]
    y_test_predict_proba = clf_log.predict_proba(x_test)[:,1]
    
    # calculamos la curva roc
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_predict_proba)
    
    # AUC train
    auc_train = metrics.roc_auc_score(y_train, y_train_predict_proba)
    print('el Área bajo la curva ROC para Regresión logística en train es: ', round(auc_train, 4))
    
    # crear ROC curve
    plt.plot(fpr,tpr, label = "AUC: " + str(round(auc_train, 4)), color = '#3E54AC')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Curva ROC para Regresión logística Benchmark')
    plt.legend(loc=4)
    plt.show()
    
    # buscamos el mejor threshold en train
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_benchmark = thresholds[optimal_idx]
    
    # predecimos cagtegóricamente según threshold
    y_train_predict = [1 if x > optimal_threshold_benchmark  else 0 for x in y_train_predict_proba]
    y_test_predict = [1 if x > optimal_threshold_benchmark  else 0 for x in y_test_predict_proba]
    
    # calculamos el accuracy para cada uno de los modelos
    train_accuracy = sum(y_train_predict == y_train) / len(y_train)
    test_accuracy = sum(y_test_predict == y_test) / len(y_test)
    print('El accuracy para Regresión logística en train es: ', round(train_accuracy, 4) * 100, '%')
    print('El accuracy para Regresión logística en test es: ', round(test_accuracy, 4) * 100, '%')
    
    # confusion matrix 
    cm = metrics.confusion_matrix(y_test, y_test_predict).astype(int)
    print('confusion matrix para Regresión logística: ', cm)
    
    # AUC para test
    auc_test = metrics.roc_auc_score(y_test, y_test_predict_proba)
    print('el Área bajo la curva ROC para Regresión logística en test es: ', round(auc_test, 4))
    
    # graficamos el accuracy del modelo - confusion matrix
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, annot = True, fmt="d", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Accuracy del Modelo para Regresión logística Benchmark')
    plt.show()
    
    return auc_test



def benchmark_randomforest_model(x_train, y_train, x_test, y_test):
    
    random.seed(777)
    
    # entrenamos un modelo randomforest con las variables estandarizadas.
    pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators = 10, max_depth = 3, random_state = 10))
    pipe.fit(x_train, y_train)
    
    # predecimos la variable dependiente para cada uno de los tipos de muestra que tenemos
    y_train_predict_proba = pipe.predict(x_train)
    y_test_predict_proba = pipe.predict(x_test)
    
    # calculamos la curva roc
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_predict_proba)
    
    # AUC train
    auc_train = metrics.roc_auc_score(y_train, y_train_predict_proba)
    print('el Área bajo la curva ROC para Random Forest en train es: ', round(auc_train, 4))
    
    # crear ROC curve
    plt.plot(fpr,tpr, label = "AUC: " + str(round(auc_train, 4)), color = '#ADA2FF')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Curva ROC para Random Forest Benchmark ')
    plt.legend(loc=4)
    plt.show()
    
    # buscamos el mejor threshold en train
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_benchmark = thresholds[optimal_idx]
    
    # predecimos cagtegóricamente según threshold
    y_train_predict = [1 if x > optimal_threshold_benchmark  else 0 for x in y_train_predict_proba]
    y_test_predict = [1 if x > optimal_threshold_benchmark  else 0 for x in y_test_predict_proba]
    
    # calculamos el accuracy para cada uno de los modelos
    train_accuracy = sum(y_train_predict == y_train) / len(y_train)
    test_accuracy = sum(y_test_predict == y_test) / len(y_test)
    print('El accuracy para Random Forest en train es: ', round(train_accuracy, 4) * 100, '%')
    print('El accuracy para Random Forest en test es: ', round(test_accuracy, 4) * 100, '%')
    
    # confusion matrix 
    cm = metrics.confusion_matrix(y_test, y_test_predict).astype(int)
    print('confusion matrix para Random Forest: ', cm)
    
    # AUC para test
    auc_test = metrics.roc_auc_score(y_test, y_test_predict_proba)
    print('el Área bajo la curva ROC para Random Forest en test es: ', round(auc_test, 4))
    
    # graficamos el accuracy del modelo - confusion matrix
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, annot = True, fmt="d", linewidths=.5, square = True, cmap = sns.cubehelix_palette(as_cmap = True))
    # sns.color_palette("YlOrBr", as_cmap=True)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Accuracy del Modelo para Random Forest Benchmark')
    plt.show()
    
    return auc_test




def features_randomforest_model(x_train, y_train, x_test, y_test):
    
    # entrenamos un modelo randomforest con las variables estandarizadas.
    pipe = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators = 10, max_depth = 3, random_state = 10))
    pipe.fit(x_train, y_train)
    
    # feature importance
    importance = pipe.steps[1][1].feature_importances_
    # dataframe con la importancia de cada feature
    var_importance = pd.DataFrame(index = x_test.columns.to_list(), data = abs(importance))
    var_importance.rename(columns = {0 : 'importance'}, inplace = True)
    var_importance.sort_values(by = 'importance', inplace = True, ascending = False)
    print(var_importance.head(10))
    # guardamos las variables más importantes para utilizarlas en un modelo deep learning luego
    variables_deep = var_importance.loc[var_importance['importance'] > 0.0].index.to_list()
    
    
    return variables_deep


################ 07 - RANDOM GRID SEARCH & XGBOOST MODEL ####################
def random_grid_search_hyperparameters(x_train, y_train):
    
    clf = xgb.XGBClassifier(random_state = 37)

    hyperparameter_grid = {
        'max_depth': randint(5, 35),     #OK
        'eta': uniform(0.01, 0.17),   #OK
        'colsample_bytree': uniform(0.83, 0.25),     #OK
        'min_child_weight': randint(1, 73),    #OK    
        'max_delta_step' : randint(0, 13),    #hiperparámetro para balancear la clase
        'gamma': uniform(1, 7.0),    #OK
        'subsample' : uniform(0.83, 0.25)}
    
    # seteamos una semilla para replicar resultados
    random.seed(777)
    # creamos los kflods
    kfold = StratifiedKFold(n_splits = 3, shuffle = True)
    # creamos el modelo para buscar la mejor combinación de hiperparámetros
    rs_clf = RandomizedSearchCV(estimator = clf, 
                                param_distributions = hyperparameter_grid, 
                                n_iter = 47,
                                n_jobs = 1, 
                                verbose = 3, 
                                cv = kfold,
                                scoring = 'roc_auc',
                                refit = True, 
                                random_state = 73)
    
    # buscamos la mejor combinación de hiperparámetros
    print("Randomized search..")
    search_time_start = time.time()
    rs_clf.fit(x_train, y_train)
    print("Randomized search time:", round((time.time() - search_time_start) / 60, 2), 'min.')
    
    print(" Results from Random Search " )
    print("\n The best estimator across ALL searched params:\n", rs_clf.best_estimator_)
    print("\n The best score across ALL searched params:\n", rs_clf.best_score_)
    print("\n The best parameters across ALL searched params:\n", rs_clf.best_params_)
    
    # guardamos en una variable el mejor score y la mejor combinación de hiperparámetros
    best_score = rs_clf.best_score_
    best_params = rs_clf.best_params_
    print("Best score: {}".format(best_score))
    print("Best params: ")
    for param_name in sorted(best_params.keys()):
        print('%s: %r' % (param_name, best_params[param_name]))
    
    # Best score: 0.8019827049247947
    """
    # PRIMERA ITERACIÓN
    
    best_params = {'colsample_bytree': 0.710513222270019,
                    'eta': 0.08035985346443755,
                    'gamma': 1.7749542059016372,
                    'max_delta_step': 6,
                    'max_depth': 10,
                    'min_child_weight': 28,
                    'subsample': 0.9827203626796498}

    The best estimator across ALL searched params:
     XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1,
                  colsample_bytree=0.710513222270019, early_stopping_rounds=None,
                  enable_categorical=False, eta=0.08035985346443755,
                  eval_metric=None, gamma=1.7749542059016372, gpu_id=-1,
                  grow_policy='depthwise', importance_type=None,
                  interaction_constraints='', learning_rate=0.0803598538,
                  max_bin=256, max_cat_to_onehot=4, max_delta_step=6, max_depth=10,
                  max_leaves=0, min_child_weight=28, missing=nan,
                  monotone_constraints='()', n_estimators=100, n_jobs=0,
                  num_parallel_tree=1, predictor='auto', random_state=0,
                  reg_alpha=0, ...)
    
     The best score across ALL searched params:
     0.8019827049247947
    
     The best parameters across ALL searched params:
     {'colsample_bytree': 0.710513222270019, 
      'eta': 0.08035985346443755, 
      'gamma': 1.7749542059016372, 
      'max_delta_step': 6, 
      'max_depth': 10, 
      'min_child_weight': 28, 
      'subsample': 0.9827203626796498}
     
     
     ######## SEGUNDA ITERACIÓN DEL GRID SEARCH 
     
     The best estimator across ALL searched params:
     XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                 colsample_bylevel=1, colsample_bynode=1,
                 colsample_bytree=0.9853842854945174, early_stopping_rounds=None,
                 enable_categorical=False, eta=0.1269505748829578,
                 eval_metric=None, gamma=3.7420674866297565, gpu_id=-1,
                 grow_policy='depthwise', importance_type=None,
                 interaction_constraints='', learning_rate=0.126950577,
                 max_bin=256, max_cat_to_onehot=4, max_delta_step=1, max_depth=8,
                 max_leaves=0, min_child_weight=45, missing=nan,
                 monotone_constraints='()', n_estimators=100, n_jobs=0,
                 num_parallel_tree=1, predictor='auto', random_state=37,
                 reg_alpha=0, ...)
        
     The best score across ALL searched params:
     0.8031167404395876
        
     The best parameters across ALL searched params:
        Best score: 0.8031167404395876
        Best params: 
        colsample_bytree: 0.9853842854945174
        eta: 0.1269505748829578
        gamma: 3.7420674866297565
        max_delta_step: 1
        max_depth: 8
        min_child_weight: 45
        subsample: 0.8811604021117182
        """
     
    
    return best_params


def xgboost_model(x_train, y_train, x_test, y_test, best_params):
    
    # creamos el modelo XGB
    xgb_model = xgb.XGBClassifier(random_state = 37)
    # seteamos los parámetros del modelo según los hallados con el grid
    xgb_model.set_params(**best_params)
    
    
    # entrenamos el modelo
    xgb_model.fit(x_train, y_train)
    
    # predecimos la variable dependiente para cada uno de los tipos de muestra que tenemos
    y_train_predict_proba = xgb_model.predict_proba(x_train)[:,1]
    y_test_predict_proba = xgb_model.predict_proba(x_test)[:,1]
    
    # calculamos la curva roc
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_predict_proba)
    
    # AUC train
    auc_train = metrics.roc_auc_score(y_train, y_train_predict_proba)
    print('Para xgboost el Área bajo la curva ROC para XGBOOST en train es: ', round(auc_train, 4))
    
    # crear ROC curve
    plt.plot(fpr,tpr, label = "AUC: " + str(round(auc_train, 4)), color = '#EB455F')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Curva ROC Train para XGBOOST Modelo final')
    plt.legend(loc=4)
    plt.show()
    
    # buscamos el mejor threshold en train
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_xgb = thresholds[optimal_idx]
    
    # predecimos cagtegóricamente según threshold
    y_train_predict = [1 if x > optimal_threshold_xgb else 0 for x in y_train_predict_proba]
    y_test_predict = [1 if x > optimal_threshold_xgb else 0 for x in y_test_predict_proba]
    
    # calculamos el accuracy para cada uno de los modelos
    train_accuracy = sum(y_train_predict == y_train) / len(y_train)
    test_accuracy = sum(y_test_predict == y_test) / len(y_test)
    print('El accuracy para XGBOOST en train es: ', round(train_accuracy, 4) * 100, '%')
    print('El accuracy para XGBOOST en test es: ', round(test_accuracy, 4) * 100, '%')
    
    # confusion matrix 
    cm = metrics.confusion_matrix(y_test, y_test_predict).astype(int)
    print('confusion matrix para XGBOOST: ', cm)
    
    # AUC para test
    auc_test = metrics.roc_auc_score(y_test, y_test_predict_proba)
    print('Para xgboost el área bajo la curva ROC en test es: ', auc_test)
    
    # graficamos el accuracy del modelo - confusion matrix
    plt.figure(figsize = (10,10))
    sns.heatmap(cm, annot = True, fmt="d", linewidths=.5, square = True, cmap = 'Reds_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Accuracy en Test para XGBOOST Modelo final')
    plt.show()
    
    return auc_test


################ 09 - CREACIÓN DEL DF PARA EL MODELO DE SURVIVAL ####################
def survival_analysis_df(survival_df):
    
    # agrupamos 
    churned_riders = survival_df.groupby('rider_id', as_index = False).agg({'binary_churn_label' : max})
    rider_seniority = survival_df.groupby('rider_id', as_index = False).agg({'seniority_week' : max})
    
    # nos quedamos unicamente con las semanas 0, 1 y 2 de cada 
    survival_df = survival_df[survival_df['seniority_week'] <= 2]
    
    survival_df = survival_df.groupby('rider_id', as_index = False).agg({'completed_deliveries' : sum,
                                                          'cancelled_deliveries' : sum,
                                                          'avg_distance_pickup_to_vendor_m' : 'mean',
                                                          'working_hours' : sum,
                                                          'stacked_orders' : sum, 
                                                          'rider_delivery_time' : sum,
                                                          'late_shift_ratio' : sum})
    
    survival_df = pd.merge(survival_df, churned_riders, on = 'rider_id', how = 'left')
    survival_df = pd.merge(survival_df, rider_seniority, on = 'rider_id', how = 'left')
    
    # quitamos los que poseen los mayores valores - riesgo de outlier
    survival_df = survival_df[survival_df['seniority_week'] < max(survival_df['seniority_week'] - 3)]
    
    # creamos la variable censored que determina si una observación se encuentra censurada
    survival_df['censored'] = [0 if x == 0 else 1 for x in survival_df['binary_churn_label']]
    
    graph_survival_probability = survival_df.groupby('seniority_week', as_index = False).agg({'rider_id' : 'count'})
    graph_survival_probability['probability'] = [sum(graph_survival_probability['rider_id'][j:])/sum(graph_survival_probability['rider_id']) for j in graph_survival_probability.index]
    
    # graficamos la survival probability
    fig, axs = plt.subplots(figsize=(15, 10))    
    axs.plot(graph_survival_probability['seniority_week'], graph_survival_probability['probability'], linewidth = 4, color = '#ff6961')
    plt.title('survival probability', loc = 'center')
    plt.show()
    
    # crear variables log y sqr orders + working hours
    survival_df.loc[:,'sqr_orders_working_hours'] = survival_df['completed_deliveries']**2 + survival_df['working_hours']**2
    survival_df.loc[:,'log_sqr_orders_working_hours'] = np.log(survival_df['sqr_orders_working_hours'])
    
    # segmentamos rápidamente según estos valores para entender si pueden resultar predictivos
    graph_two = survival_df[['rider_id', 'seniority_week', 'sqr_orders_working_hours']]
    graph_two.loc[graph_two['sqr_orders_working_hours'] <= np.quantile(graph_two['sqr_orders_working_hours'], 1), 'percentile'] = 100
    graph_two.loc[graph_two['sqr_orders_working_hours'] <= np.quantile(graph_two['sqr_orders_working_hours'], 0.75), 'percentile'] = 75
    graph_two.loc[graph_two['sqr_orders_working_hours'] <= np.quantile(graph_two['sqr_orders_working_hours'], 0.50), 'percentile'] = 50
    graph_two.loc[graph_two['sqr_orders_working_hours'] <= np.quantile(graph_two['sqr_orders_working_hours'], 0.25), 'percentile'] = 25
    
    graph_two = graph_two.groupby(['seniority_week', 'percentile'], as_index = False).agg({'rider_id' : 'count'})
    
    
    # ordenamos para luego definir valores de la nueva variable
    graph_two.sort_values(by = ['percentile', 'seniority_week'], ascending = True, inplace = True)
    
    # agrupamos y calculamos la probabilidad para segmento según cuartil
    prob = []
    for i in graph_two['percentile'].unique():
        subset = graph_two[graph_two['percentile'] == i]
        subset.reset_index(inplace = True)
        prob_ii = [sum(subset['rider_id'][j:])/sum(subset['rider_id']) for j in subset.index]
        prob.extend(prob_ii)
    
    # incorporamos la lista como columna al df
    graph_two['probability'] = prob 
    
    # graficamos cada segmento buscando diferencias 
    # graficamos la survival probability
    fig, axs = plt.subplots(figsize=(15, 10))    
    for i, j in zip(graph_two['percentile'].unique(), ['#B5EAEA', '#EDF6E5', '#FFBCBC', '#B6E2A1']):
        axs.plot(graph_two[graph_two['percentile'] == i]['seniority_week'], graph_two[graph_two['percentile'] == i]['probability'], linewidth = 4, color = j)
    plt.legend(graph_two['percentile'].unique())
    plt.title('survival probability', loc = 'center')
    plt.show()
    
    # convertimos a bool 
    survival_df['censored'] = survival_df['censored'].astype(bool)
    
    
    # cambianmos los nombres de las variables a predecir
    survival_df.rename(columns = {'censored' : 'event',
                                   'seniority_week' : 'duration'}, inplace = True)
    
    survival_df.set_index('rider_id', inplace = True)  
    survival_df.drop(columns = 'binary_churn_label', inplace = True)
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations = survival_df["duration"], event_observed = survival_df["event"])
    kmf.plot_survival_function()
    plt.ylabel('est. probability of survival')
    plt.xlabel('time')
    plt.title('Survival Curves')    
    plt.show()
    
    return survival_df



def survival_cox_model_base(survival_df):
    
    # https://github.com/himchopra/survivalanalysis/blob/main/dsti_survivalanalysis.ipynb
    
    # Cox Model - statistics
    cph = CoxPHFitter()
    cph.fit(survival_df, 'duration', event_col = 'event')
    cph.print_summary()
    
    # https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxnetSurvivalAnalysis.html
    
    # spliteamos el df en x & y
    x = survival_df.drop(['duration', 'event'], axis = 1)
    y = convert_to_structured(survival_df['duration'], survival_df['event'])
    # spliteamos en test y train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 5)
    
    # creamo el estimador
    estimator = CoxnetSurvivalAnalysis(l1_ratio=0.99, fit_baseline_model=True)
    estimator.fit(x_train, y_train)
    
    train_c_index = estimator.score(x_train, y_train)
    # train_c_index = 0.6950000105629817
    
    test_c_index = estimator.score(x_test, y_test)
    # test_c_index = 0.6114263849064957

    # graficamos la survival function para los cuatro mayores alpha
    surv_funcs = {}
    for alpha in estimator.alphas_[:5]:
        surv_funcs[alpha] = estimator.predict_survival_function(
            x.iloc[:1], alpha=alpha)
 
    for alpha, surv_alpha in surv_funcs.items():
        for fn in surv_alpha:
            plt.step(fn.x, fn(fn.x), where="post",
                     label="alpha = {:.3f}".format(alpha))
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    
    return train_c_index, test_c_index



def survival_improved_model(survival_df):
    
    # https://github.com/sebp/scikit-survival/blob/master/doc/user_guide/coxnet.ipynb
    
    # https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.linear_model.CoxPHSurvivalAnalysis.html
        
    set_config(display="text")  # displays text representation of estimators
    
    # spliteamos el df en x & y
    x = survival_df.drop(['duration', 'event'], axis = 1)
    y = convert_to_structured(survival_df['duration'], survival_df['event'])
    # spliteamos en test y train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 37)
    
    
    # creamos el pipeline del modelo
    coxnet_pipe = make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(l1_ratio = 0.9, alpha_min_ratio = 0.01, max_iter = 100))
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(x_train, y_train)
    
    
    # buscamos los mejores parámetros del modelo
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits = 5, shuffle = True, random_state = 0)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio = 0.9)),
        param_grid = {"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv = cv,
        error_score = 0.5,
        n_jobs = 1).fit(x_train, y_train)
    
    cv_results = pd.DataFrame(gcv.cv_results_)
    
    
    #cv_results = pd.read_csv('cv_results_cox.csv') # solo para no correr toda la búsqueda otra nuevamente - después eliminar
    
    # graficamos el c_index para mostrar el mejor valor encontrado para el parámetro alpha
    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)
    
    # creamos el modelo predictivo cox net con el mejor valor de alpha
    coxnet_pred = make_pipeline(
        StandardScaler(),
        CoxnetSurvivalAnalysis(l1_ratio=0.9, fit_baseline_model=True))
    coxnet_pred.set_params(**gcv.best_params_)
    coxnet_pred.fit(x_train, y_train)
    
    # predecimos con el modelo entrenado
    train_c_index = coxnet_pred.score(x_train, y_train)
    # train_c_index = 0.697848081530137
    
    # predecimos con el modelo entrenado
    test_c_index = coxnet_pred.score(x_test, y_test)
    # test_c_index = 0.6994348040507276
    # graficamos la survival function para cada observación en y_test
    surv_fns_test = coxnet_pred.predict_survival_function(x_test)
    surv_fns_test = coxnet_pred.predict(x_test)
    
    
    # https://towardsdatascience.com/how-to-evaluate-survival-analysis-models-dd67bc10caae
    
    # Times <t> at which to calculate the AUC
    va_times = np.arange(1, 119, 1)
    # where max(<t>) is chosen arbitrarily and < of follow-up time
    # Risk scores <f(xi)> on test data
    cph_risk_scores = coxnet_pred.predict(x_test)
    # AUC at times <t> and its average
    cph_auc, cph_mean_auc = cumulative_dynamic_auc(
      y_train, y_test, cph_risk_scores, va_times
    )
    
    # Plot AUC Dinámico
    plt.figure(figsize = (12,8))
    plt.plot(va_times, cph_auc, marker = 'o')
    plt.axhline(cph_mean_auc, linestyle = '--')
    plt.title('AUC para datos de Test en diferentes puntos de tiempo')
    plt.xlabel('Períodos de tiempo')
    plt.ylabel('AUC Dinámico')
    # incoroporamos al gráfico la media de AUC
    textbox = 'AUC promedio: {:.3f}'.format(cph_mean_auc)
    props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.9)
    plt.text(80, 0.75, textbox, fontsize = 20, bbox = props)
    plt.grid(True)
    plt.show()
    
    pred_surv = coxnet_pred.predict_survival_function(x_test)
    time_points = np.arange(1, 120)
    for i, surv_func in enumerate(pred_surv):
        plt.step(time_points, surv_func(time_points), where="post")
    plt.ylabel("probabilidad de supervivencia $\hat{S}(t)$")
    plt.xlabel("períodos de tiempo $t$")
    plt.show()
    
    
    return train_c_index, test_c_index, surv_fns_test



def main():
        
    # definimos la ubicación del directorio principal
    path = 'C:/Users/damian.ilkow/Desktop/dami/mim/materias/99_Seminario/00_code'
    # seteamos el working directory
    os.chdir(path)
    # cargamos el df
    df = data_load_riders(path)
    
    # limpiamos el dataset
    df = transform_data(df)
    
    # creamos los labels necesarios del modelo
    df = set_y_label(df)
    # analizamos el dataset y creamos algunos gráficos
    df = data_analysis(df)
    
    # analizamos los datos y buscamos variables que puedan ser relelvantes
    df = feature_engineering(df)
    # analizamos las features creadas y graficamos
    data_analysis_feature_engineering(df)
    
    # creamos una copia del df para luego modificarla para survivial analysis
    survival_df = df.copy()
    
    
    # balanceamos las clases
    df = class_balance(df)
    
    # separamos el dataset en train y test
    x_train, y_train, x_test, y_test = test_train_separation(df)
    
    # creamos un modelo de Regresión Logística que utilizaremos como Benchmark
    b_auc = benchmark_logistic_model(x_train, y_train, x_test, y_test)
    
    # creamos un modelo de Regresión Logística que utilizaremos como Benchmark
    b2_auc = benchmark_randomforest_model(x_train, y_train, x_test, y_test)
    
    # realizamos un random grid search para definir los mejores hiperparámetros
    #best_params = random_grid_search_hyperparameters(x_train, y_train)
    best_params = {'colsample_bytree': 0.710513222270019,
                    'eta': 0.08035985346443755,
                    'gamma': 1.7749542059016372,
                    'max_delta_step': 6,
                    'max_depth': 10,
                    'min_child_weight': 28,
                    'subsample': 0.9827203626796498}
    # creamos un modelo de boosting XGB
    xgb_auc = xgboost_model(x_train, y_train, x_test, y_test, best_params)
    ################ 08 - COMPARACIÓN DE MODELOS ####################
    # diferencia entre modelos
    print('el AUC nuevo modelo, es mejor que el benchmark, en: ', xgb_auc - b_auc)
    
    ############### SEGUNDA PARTE - PROBAMOS UN MODELO DE SURVIVAL ANALYSIS
    
    
    # Calculamos la importancia de las variables a través de Random Forest
    features_deep = features_randomforest_model(x_train, y_train, x_test, y_test)
    # consideraremos sólo las dos primeras semanas de cada repartidor para definir
    survival_df = survival_analysis_df(df)
    # corremos el primer modelo 
    first_train_index, first_test_index = survival_cox_model_base(survival_df)
    # corremos el segundo modelo, buscando hiperparámetros
    second_train_index, second_test_index, surv_fns = survival_improved_model(survival_df)
    
    pass



main()
