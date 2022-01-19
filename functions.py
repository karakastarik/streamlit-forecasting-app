import requests
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit,train_test_split
from catboost import CatBoostRegressor

def select_period(period):
    periods={"1 gün":24,"2 gün":48,"3 gün":72,"1 hafta":168,"2 hafta":336}
    return periods[period]

def get_consumption_data(start_date,end_date):
    url = "https://seffaflik.epias.com.tr/transparency/service/consumption/real-time-consumption?startDate="+f'{start_date}'+"&endDate="+f'{end_date}'
    response = requests.get(url,verify=False)
    json_data = json.loads(response.text.encode('utf8'))
    df = pd.DataFrame(json_data["body"]["hourlyConsumptions"])
    df['date']=pd.to_datetime(df.date.str[:16])
    return df

def date_features(df):
    df_c=df.copy()
    df_c['month'] = df_c['date'].dt.month
    df_c['year'] = df_c['date'].dt.year
    df_c['hour'] = df_c['date'].dt.hour
    df_c['quarter'] = df_c['date'].dt.quarter
    df_c['dayofweek'] = df_c['date'].dt.dayofweek
    df_c['dayofyear'] = df_c['date'].dt.dayofyear
    df_c['dayofmonth'] = df_c['date'].dt.day
    df_c['weekofyear'] = df_c['date'].dt.weekofyear
    return df_c

def rolling_features(df,fh):
    df_c=df.copy()
    rolling_windows=[fh,fh+3,fh+10,fh+15,fh+20,fh+25]
    lags=[fh,fh+5,fh+10,fh+15,fh+20,fh+30]
    for a in rolling_windows:
        df_c['rolling_mean_'+str(a)]=df_c['consumption'].rolling(a,min_periods=1).mean().shift(1)
        df_c['rolling_std_'+str(a)]=df_c['consumption'].rolling(a,min_periods=1).std().shift(1)
        df_c['rolling_min_'+str(a)]=df_c['consumption'].rolling(a,min_periods=1).min().shift(1)
        df_c['rolling_max_'+str(a)]=df_c['consumption'].rolling(a,min_periods=1).max().shift(1)
        df_c['rolling_var_'+str(a)]=df_c['consumption'].rolling(a,min_periods=1).var().shift(1)
    for l in lags:
        df_c['consumption_lag_'+str(l)]=df_c['consumption'].shift(l)

    return df_c

def forecast_func(df,fh):
    fh_new=fh+1
    date=pd.date_range(start=pd.to_datetime(df.date).tail(1).iloc[0], periods=fh_new, freq='H')
    date=pd.DataFrame(date).rename(columns={0:"date"})
    df_fe=pd.merge(df,date,how='outer')

    #feature engineering
    df_fe=rolling_features(df_fe,fh_new)
    df_fe=date_features(df_fe)
    df_fe=df_fe.iloc[fh_new+30:].reset_index(drop=True)

    #train/test split
    split_date = pd.to_datetime(df_fe.date).tail(fh_new).iloc[0]
    print(split_date)
    historical = df_fe.loc[df_fe.date <= split_date]
    y=historical[['date','consumption']].set_index('date')
    X=historical.drop('consumption',axis=1).set_index('date')
    forecast_df=df_fe.loc[df_fe.date > split_date].set_index('date').drop('consumption',axis=1)


    tscv = TimeSeriesSplit(n_splits=3,test_size=fh_new*20)
    
    score_list = []
    fold = 1
    unseen_preds = []
    importance = []
    #cross validation step
    for train_index, test_index in tscv.split(X,y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        print(X_train.shape,X_val.shape)

        cat = CatBoostRegressor(iterations=1000,eval_metric='MAE',allow_writing_files=False)
        cat.fit(X_train,y_train,eval_set=[(X_val,y_val)],early_stopping_rounds=150,verbose=50)

        forecast_predicted=cat.predict(forecast_df)
        unseen_preds.append(forecast_predicted)

        score = mean_absolute_error(y_val,cat.predict(X_val))
        print(f"MAE Fold-{fold} : {score}")
        score_list.append(score)
        importance.append(cat.get_feature_importance())
        fold+=1
    print("CV Mean Score:",np.mean(score_list))

    forecasted=pd.DataFrame(unseen_preds[2],columns=['forecasting']).set_index(forecast_df.index)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_fe.date.iloc[-fh_new*5:],y=df_fe.consumption.iloc[-fh_new*5:],name='Tarihsel Veri',mode='lines'))
    fig1.add_trace(go.Scatter(x=forecasted.index,y=forecasted["forecasting"],name='Öngörü',mode='lines'))
    fig1.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    f_importance=pd.concat([pd.Series(X.columns.to_list(),name="Feature"),pd.Series(np.mean(importance,axis=0),name="Importance")],axis=1).sort_values(by="Importance",ascending=True)
    fig2 = px.bar(f_importance.tail(10), x='Importance', y='Feature')
    return fig1,fig2
