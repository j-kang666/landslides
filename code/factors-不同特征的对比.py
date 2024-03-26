# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:09:01 2024

@author: kangjia
"""

# factors

import xarray as xr
import pandas as pd
import os
import rasterio
import rioxarray as rxr
# 读取Excel文件
excel_file_path = 'D:/ei_lanslide/updated_traindata2016.xlsx'
data = pd.read_excel(excel_file_path)

# In[]



import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score,train_test_split
from bayes_opt import BayesianOptimization



columns = ['precipitation_day_0','Pa','aspect', 'DEM', 'GPM', 'Lithology', 'NDVI', 'plants', 'slope', 'soil']
data = data.dropna()
# Use the cumulative precipitation as the feature
X = data[columns]
y = data['label']


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size=0.3,random_state=42)#分训

# XGBoost训练
def xgb_cv(learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree):
    val = cross_val_score(xgb.XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_child_weight=int(min_child_weight),
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42), Xtrain, Ytrain.ravel(), scoring='accuracy', cv=10).mean()
    return val

xgb_bo = BayesianOptimization(
        xgb_cv,
        {'learning_rate': (0.01, 0.3),
         'n_estimators': (50, 300),
         'max_depth': (3, 100),
         'min_child_weight': (1, 10),
         'gamma': (0, 1),
         'subsample': (0.8, 1),
         'colsample_bytree': (0.8, 1)})
xgb_bo.maximize()

# SVM训练
def svm_cv(C, gamma):
    val = cross_val_score(SVC(
            C=C,
            gamma=gamma,
            
            random_state=42), Xtrain, Ytrain.ravel(), scoring='accuracy', cv=10).mean()
    return val

svm_bo = BayesianOptimization(
        svm_cv,
        {'C': (0.1, 10),
         'gamma': (0.01, 1)})
svm_bo.maximize()

# KNN训练
def knn_cv(n_neighbors):
    val = cross_val_score(KNeighborsClassifier(
            n_neighbors=int(n_neighbors)),
            Xtrain, Ytrain.ravel(), scoring='accuracy', cv=10).mean()
    return val

knn_bo = BayesianOptimization(
        knn_cv,
        {'n_neighbors': (2, 20)})
knn_bo.maximize()

# Random Forest训练
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(RandomForestClassifier(
            n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=int(max_features),
            max_depth=int(max_depth),
            random_state=42), Xtrain, Ytrain.ravel(), scoring='accuracy', cv=10).mean()
    return val

rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (50, 300),
         'min_samples_split': (2, 10),
         'max_features': (1, 10),
         'max_depth': (5, 300)})
rf_bo.maximize()

# 逻辑回归训练
def lr_cv(C):
    val = cross_val_score(LogisticRegression(
            C=C,
            max_iter=1000,  # 增加迭代次数以确保收敛
            random_state=42), Xtrain, Ytrain.ravel(), scoring='accuracy', cv=10).mean()
    return val

lr_bo = BayesianOptimization(
        lr_cv,
        {'C': (0.01, 10)})
lr_bo.maximize()


# 获取各模型最优参数
xgb_best_params = {'colsample_bytree': 0.8708400390255239,
 'gamma': 0.8626702404751789,
 'learning_rate': 0.12135244414607452,
 'max_depth': 15,
 'min_child_weight': 2.708389562392272,
 'n_estimators': 280,
 'subsample': 0.996737101892489}
svm_best_params = {k: v for k, v in svm_bo.max['params'].items() if k != 'params'}
knn_best_params = {k: int(v) for k, v in knn_bo.max['params'].items() if k != 'params'}
rf_best_params = {k: int(v) for k, v in rf_bo.max['params'].items() if k != 'params'}
lr_best_params = {'C': lr_bo.max['params']['C']}

# 训练各模型
xgb_model = xgb.XGBClassifier(**xgb_best_params, random_state=42)
svm_model = SVC(**svm_best_params,probability=True, random_state=42)
knn_model = KNeighborsClassifier(**knn_best_params)
rf_model = RandomForestClassifier(**rf_best_params, random_state=42)
lr_model = LogisticRegression(**lr_best_params, max_iter=1000, random_state=42)


xgb_model.fit(Xtrain, Ytrain.ravel())
svm_model.fit(Xtrain, Ytrain.ravel())
knn_model.fit(Xtrain, Ytrain.ravel())
rf_model.fit(Xtrain, Ytrain.ravel())
lr_model.fit(Xtrain, Ytrain.ravel())

# 预测并计算指标
xgb_Ytest_predict = xgb_model.predict(Xtest)
svm_Ytest_predict = svm_model.predict(Xtest)
knn_Ytest_predict = knn_model.predict(Xtest)
rf_Ytest_predict = rf_model.predict(Xtest)

xgb_accuracy = accuracy_score(Ytest, xgb_Ytest_predict)
svm_accuracy = accuracy_score(Ytest, svm_Ytest_predict)
knn_accuracy = accuracy_score(Ytest, knn_Ytest_predict)
rf_accuracy = accuracy_score(Ytest, rf_Ytest_predict)

# 预测概率并计算AUC
xgb_probabilities = xgb_model.predict_proba(Xtest)[:, 1]
svm_probabilities = svm_model.predict_proba(Xtest)[:, 1]
knn_probabilities = knn_model.predict_proba(Xtest)[:, 1]
rf_probabilities = rf_model.predict_proba(Xtest)[:, 1]

xgb_auc = roc_auc_score(Ytest, xgb_probabilities)
svm_auc = roc_auc_score(Ytest, svm_probabilities)
knn_auc = roc_auc_score(Ytest, knn_probabilities)
rf_auc = roc_auc_score(Ytest, rf_probabilities)

lr_Ytest_predict = lr_model.predict(Xtest)
lr_accuracy = accuracy_score(Ytest, lr_Ytest_predict)
lr_probabilities = lr_model.predict_proba(Xtest)[:, 1]
lr_auc = roc_auc_score(Ytest, lr_probabilities)


# 输出各模型的评价指标
print("XGBoost 模型评价指标：")
print(f"Accuracy: {xgb_accuracy}")
print(f"AUC: {xgb_auc}")

print("\nSVM 模型评价指标：")
print(f"Accuracy: {svm_accuracy}")
print(f"AUC: {svm_auc}")

print("\nKNN 模型评价指标：")
print(f"Accuracy: {knn_accuracy}")
print(f"AUC: {knn_auc}")

print("\nRandom Forest 模型评价指标：")
print(f"Accuracy: {rf_accuracy}")
print(f"AUC: {rf_auc}")

print("\nLogistic Regression 模型评价指标：")
print(f"Accuracy: {lr_accuracy}")
print(f"AUC: {lr_auc}")

from sklearn.metrics import confusion_matrix

def calculate_metrics(Y_true, Y_pred):
    # 计算混淆矩阵
    TN, FP, FN, TP = confusion_matrix(Y_true, Y_pred).ravel()
    
    # 计算POD, POFD, EI
    POD = TP / (TP + FN)
    POFD = FP / (FP + TN)
    EI = POD - POFD
    
    return POD, POFD, EI

# 计算每个模型的指标
xgb_metrics = calculate_metrics(Ytest, xgb_Ytest_predict)
svm_metrics = calculate_metrics(Ytest, svm_Ytest_predict)
knn_metrics = calculate_metrics(Ytest, knn_Ytest_predict)
rf_metrics = calculate_metrics(Ytest, rf_Ytest_predict)
lr_metrics = calculate_metrics(Ytest, lr_Ytest_predict)

# 输出指标结果
print("XGBoost Metrics: POD = {:.2f}, POFD = {:.2f}, EI = {:.2f}".format(*xgb_metrics))
print("SVM Metrics: POD = {:.2f}, POFD = {:.2f}, EI = {:.2f}".format(*svm_metrics))
print("KNN Metrics: POD = {:.2f}, POFD = {:.2f}, EI = {:.2f}".format(*knn_metrics))
print("Random Forest Metrics: POD = {:.2f}, POFD = {:.2f}, EI = {:.2f}".format(*rf_metrics))
print("Logistic Regression Metrics: POD = {:.2f}, POFD = {:.2f}, EI = {:.2f}".format(*lr_metrics))

# In[]
#验证集
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, accuracy_score,roc_auc_score


data = pd.read_excel('D:/ei_lanslide/landslide2014.xlsx')
columns = ['precipitation_day_0','Pa','aspect', 'DEM', 'GPM', 'Lithology', 'NDVI', 'plants', 'slope', 'soil']
data = data.dropna()
# Use the cumulative precipitation as the feature
X = data[columns]
Y = data['label']





knn_Y_predict = knn_model.predict(X)
rf_Y_predict = rf_model.predict(X)
lr_Y_predict = lr_model.predict(X)

knn_Y_accuracy = accuracy_score(Y, knn_Y_predict)
rf_Y_accuracy = accuracy_score(Y, rf_Y_predict)
lr_Y_accuracy = accuracy_score(Y, lr_Y_predict)

knn_Y_probabilities = knn_model.predict_proba(X)[:, 1]
rf_Y_probabilities = rf_model.predict_proba(X)[:, 1]
lr_Y_probabilities = lr_model.predict_proba(X)[:, 1]

print("\nKNN 验证机模型评价指标：")
print(f"Accuracy: {knn_Y_accuracy}")
print(f"AUC: {knn_auc}")

print("\nRandom Forest 验证集模型评价指标：")
print(f"Accuracy: {rf_Y_accuracy}")
print(f"AUC: {rf_auc}")

print("\nlogistic 验证集模型评价指标：")
print(f"Accuracy: {lr_Y_accuracy}")
print(f"AUC: {lr_auc}")

# In[]
#验证7月6日
import xarray as xr
import pandas as pd
import os
from scipy.interpolate import interp2d

# 文件夹路径
folder_path = 'D:/ei_lanslide/factors/'

# 获取文件夹下所有tif文件
tif_files = [file for file in os.listdir(folder_path) if file.endswith('.tif')]

# 创建一个空的DataFrame
df = pd.DataFrame()

n=1
# 遍历tif文件
for tif_file in tif_files:
    n=n+1
    # 读取tif文件
    ds = xr.open_rasterio(os.path.join(folder_path, tif_file))
    ds = ds.where(ds != ds.nodatavals,np.nan)[0,0:973,0:1040]
    # 将数据转换为一维数组
    data_array = ds.values.flatten()
    
    # 创建DataFrame的列名
    column_name = columns[n]
    
    # 将一维数组添加到DataFrame中
    df[column_name] = data_array
 
lon = ds.x.values
lat = ds.y.values    

lon_grid, lat_grid = np.meshgrid(lon, lat) 
# 创建一个空的DataFrame
df_pre = pd.DataFrame()

# 日期范围
start_date = '2014-06-21'
end_date = '2014-07-06'

# 打开NC文件
nc_ds = xr.open_dataset('D:/ei_lanslide/GPM-2014-day.nc')

# 根据日期范围选择数据
nc_data = nc_ds['precipitationCal'].sel(time=slice(start_date, end_date))


# 循环处理每一天的数据
for i, date in enumerate(pd.date_range(start=start_date, end=end_date)):
    i = 15-i
    # 获取单天的数据
    daily_data = nc_data[15-i,:,:]
    pre = daily_data.sel(lat=lat,lon=lon,method='nearest')
    pre = pre.T
        
    # 创建DataFrame的列名
    column_name = f'precipitation_day_{i}'
    
    # 将一维数组添加到DataFrame中
    df_pre[column_name] = pre.values.flatten()

k = 0.84
n_days = 11

def calculate_Pa(row, K, n):
    Pa = 0
    for i in range(n + 1):
        Pa += (K ** i) * row[f'precipitation_day_{i}']
        
    return Pa

df['Pa'] = df_pre.apply(lambda row: calculate_Pa(row, k,n_days), axis=1)
df['precipitation_day_0'] = df_pre['precipitation_day_0']

df = df[columns]
df_data = df.dropna(subset=(df.columns))

rf_Y_result = rf_model.predict_proba(df_data)[:, 1]
lr_Y_reslut = lr_model.predict_proba(df_data)[:, 1]

df_data['label'] = rf_Y_result

df['label'] = df_data['label']

lon2d, lat2d = np.meshgrid(lon, lat)
label = np.array(df['label']).reshape(973, 1040)

df_nc = xr.Dataset({'label':(('lat', 'lon'), label)},
                   coords={'lat':('lat',lat2d[:,0]),
                           'lon': ('lon',lon2d[0,:])}
                   )
df_nc.to_netcdf(r'D:/ei_lanslide/result/rf_c.nc')


df_data['label'] = lr_Y_reslut

df['label'] = df_data['label']

lon2d, lat2d = np.meshgrid(lon, lat)
label = np.array(df['label']).reshape(973, 1040)

df_nc = xr.Dataset({'label':(('lat', 'lon'), label)},
                   coords={'lat':('lat',lat2d[:,0]),
                           'lon': ('lon',lon2d[0,:])}
                   )
df_nc.to_netcdf(r'D:/ei_lanslide/result/lr_factors.nc')
