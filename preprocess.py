import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import os
from datetime import datetime
from tensorflow.keras import backend as K

target_column = ['Sale','SalesAmountInEuro','time_delay_for_conversion']
feature_column=['click_timestamp','nb_clicks_1week','product_price','product_age_group','device_type','audience_id','product_gender','product_brand','product_category1','product_category2','product_category3','product_category4','product_category5','product_category6','product_category7','product_country','product_id','product_title','partner_id','user_id']

ohe_columns = ['device_type','product_gender','product_age_group','product_country','weekday','hour']
integer_columns = ['nb_clicks_1week']
model_path = './model/model.dat'
model_update_time = 0

def l1_reg(weight_matrix):
    return 0.001 * K.sum(K.abs(weight_matrix))

l1_reg = l1_reg

model = []
#tf.keras.models.load_model(model_path,custom_objects={'l1_reg': l1_reg})

# data로 DataFrame 생성
def makeDataFrame(data) :
    df = pd.DataFrame(data)
    df = df.transpose()
    df.columns = feature_column
    return df

def makeCrossTermFeature(df,cross_term_left,cross_term_right) :
    df[cross_term_left+'_'+cross_term_right] = df[cross_term_left]+'_'+df[cross_term_right]
    if cross_term_left+'_'+cross_term_right not in ohe_columns :
        ohe_columns.append(cross_term_left+'_'+cross_term_right)
    return df

def makeTimeFeature(df) :
    df['click_timestamp'] = df['click_timestamp'].astype(int)
    df['datetime'] = df['click_timestamp'].map(lambda x : datetime.fromtimestamp(x).strftime('%Y %m %d %H %M %S'))
    #df['datetime'] = df['click_timestamp'].map(lambda x :print('-----------------',type(x)))

    df['weekday'] = df['click_timestamp'].map(lambda x : str(datetime.fromtimestamp(x).weekday()))
    df[['year','month','day','hour','minute','second']] = pd.DataFrame(df.datetime.str.split(' ',6).tolist())
    return df

## 범주형 데이터 one hot encoding
def ohe(df,df_preprocess,use_saved_columns=False):
    if use_saved_columns==True :
        col_df = pd.read_csv('saved_ohe_columns.csv',header=None)
        save_columns = col_df[1].to_list()

    product_category_columns = ['product_category'+str(i)for i in range(1,8)]
    for column in ohe_columns :
        ohe_df = pd.get_dummies(df[column],prefix=column)
        df_preprocess = pd.concat( [df_preprocess,ohe_df] ,axis=1)
    
    if use_saved_columns==True :
        return df_preprocess.reindex(columns=save_columns,fill_value=0)
    else :
        return df_preprocess

def makeDataFrameFromDataSet(size) :
    chunksize=size
    df=pd.DataFrame(columns = target_column+feature_column)
    data = []
    for cnt,chunk in enumerate(pd.read_csv('./data/Criteo_Conversion_Search/CriteoSearchData',header=None,delimiter='\t',chunksize=chunksize)) :
        chunk.columns = target_column+feature_column
        df = pd.concat([df,chunk],axis=0)
        break 
    return df

# # size 까지의 데이터로 ohe columns을 미리 생성
# def makeOheColumns(size,use_saved_columns=False):
#     df=makeDataFrameFromDataSet(size)
#     df=missingValueProcessCategorical(df)
#     return ohe(df,pd.DataFrame(),use_saved_columns)

# 최빈값과 그 비율을 구함.
def getMode(series) :
    total = 0
    max_value = 0
    max_index = 0
    for index in series.value_counts().index :
        if index!=str(-1):
            value = series.value_counts()[index]
            total += value
            if  value > max_value :
                max_value = value
                max_index = index
    return max_value,total,max_index

# 정수형 데이터 결측값 처리
def missingValueProcessInteger(df):
    save_mean = pd.read_csv("save_df_mean.csv")

    save_mean.index = integer_columns

    #nb_clicks_1week 의 -1 값을 모두 평균으로 대체
    for integer_column in integer_columns : 
        df[integer_column] = df[integer_column].astype(float)
        meanValue = save_mean.loc[integer_column]
        df[integer_column][df[integer_column]==-1] = meanValue['mean']
    return df

# 범주형 데이터 결측값 처리
def missingValueProcessCategorical(df):
    save_mode=pd.read_csv('save_mode.csv',header=0)
    
    save_mode.index = ohe_columns

    for ohe_column in ohe_columns :
        modeResult = save_mode.loc[ohe_column]
        if modeResult['total']>0 and modeResult['max_value']/modeResult['total'] >= 0.5 :
            df[ohe_column] = df[ohe_column].map(lambda x : modeResult['max_index'] if x==str(-1) else x)
    return df

# 서버에서 사용할 데이터를 csv 형식으로 저장
def save_value_by_csv(value_name,value,df_name):
    save_mean = pd.DataFrame(columns=[value_name])
    for integer_column in integer_columns :
        save_mean.loc[integer_column,:] = [value]
    save_mean.to_csv("./save_"+df_name+"_"+value_name+".csv")
    save_mean

# 정수형 데이터 normalization
def normalization(df_preprocess) :
    save_df_preprocess_mean=pd.read_csv('save_df_preprocess_mean.csv',header=0,index_col=0)
    save_df_preprocess_std=pd.read_csv('save_df_preprocess_std.csv',header=0,index_col=0)

    for integer_column in integer_columns:
        mean = save_df_preprocess_mean['mean'][integer_column]
        std = save_df_preprocess_std['std'][integer_column]
        df_preprocess[integer_column] = (df_preprocess[integer_column]-mean)/std

    save_df_preprocess_min=pd.read_csv('save_df_preprocess_min.csv',header=0,index_col=0)
    save_df_preprocess_max=pd.read_csv('save_df_preprocess_max.csv',header=0,index_col=0)

    for integer_column in integer_columns:
        min = save_df_preprocess_min['min'][integer_column]
        max = save_df_preprocess_max['max'][integer_column]
        df_preprocess[integer_column] = (df_preprocess[integer_column]-min)/(max-min)
    
    return df_preprocess

def scale(df_preprocess,scaler) :

    scaler.fit(df_preprocess[integer_columns])
    data = scaler.transform(df_preprocess[integer_columns])
    df_integer_preprocess = pd.DataFrame(data,columns=integer_columns)

    for integer_column in integer_columns:
        df_preprocess[integer_column] = df_integer_preprocess[integer_column]

    return df_preprocess

def totalProcess(df):
    global ohe_columns
    ohe_columns = ['device_type','product_gender','product_age_group','product_country','weekday','hour']
    df = missingValueProcessCategorical(df)
    
    makeCrossTermFeature(df,'product_country','hour')    
    makeCrossTermFeature(df,'product_country','weekday')
    makeCrossTermFeature(df,'device_type','product_gender')

    df = missingValueProcessInteger(df)

    df_preprocess = pd.DataFrame()
    df_preprocess = ohe(df,df_preprocess,use_saved_columns=True)
    df_preprocess = pd.concat([df['nb_clicks_1week'],df_preprocess],axis=1)
    df_preprocess = df_preprocess.astype(float)
    df_preprocess = normalization(df_preprocess)
    return df_preprocess

def result_to_txt_file(df,cvr_result,filePath) :
    list_cvr = cvr_result.tolist()
    list_cvr = list(map(lambda x : x[0],list_cvr))
    list_cvr = list(map(str,list_cvr))
    cnt=0
    for i in df.index :
        list_cvr[cnt] = str(i+1)+','+str(df[i])+','+list_cvr[cnt]
        cnt+=1

    with open(filePath, 'w+') as lf:
        lf.write('\n'.join(list_cvr))