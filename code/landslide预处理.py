# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:57:00 2023

@author: kangjia
"""

import pandas as pd
import numpy as np
import os
import xarray as xr
import pytz

ds = pd.read_excel('D:/ei_lanslide/landslide2016.xlsx')
#将时间规整到一列
ds['date'] = 0
for n in range(len(ds)):
    if pd.isna(ds['发生日E'][n]):
        ds['date'][n] =  str(ds['Year'][n]) + '/' + str(ds['Month'][n]) + '/' + str(ds['Day'][n])
    else:
        ds['date'][n] =  ds['发生日E'][n]

ds['date'] = pd.to_datetime(ds['date'])

 
#统一时间，将北京时统一为世界时
'''
# 处理时间列，将'unknow'和空值填充为'00点00分'
ds_yunnan = ds[ds['省'] == '云南']
ds_yunnan['发生时E'] = ds_yunnan['发生时E'].str.split('点').str[0]

#322和209行有问题删除
#ds_yunnan = ds_yunnan.drop([209,322])
ds_yunnan = ds_yunnan[ds_yunnan['发生时E'] != '24']


ds_yunnan['date'] = ds_yunnan['date'].astype(str) + ' ' + ds_yunnan['发生时E'].astype(str) + ':00' 
ds_yunnan['date'] = pd.to_datetime(ds_yunnan['date'], format='%Y-%m-%d %H:%M')
# 创建时区对象，北京时区为'Asia/Shanghai'，目标时区为UTC
beijing_tz = pytz.timezone('Asia/Shanghai')
utc_tz = pytz.timezone('UTC')

# 将日期时间列从北京时间转换为世界时间
ds_yunnan['date'] = ds_yunnan['date'].apply(lambda x: beijing_tz.localize(x).astimezone(utc_tz))
'''
df = pd.DataFrame()
df[['date','lon','lat']] = ds[['date','经度','纬度']]
df.drop_duplicates(subset=['date', 'lon', 'lat'], keep='first', inplace=True)


# In[]

import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# 加载NetCDF文件
nc_file_path = 'D:/ei/GPM-2016-day.nc'  # 请替换为实际的文件路径
ds = xr.open_dataset(nc_file_path, decode_times=False)

# 将NetCDF中的时间转换为日期
ref_date = datetime(1970, 1, 1)
ds['time'] = pd.to_datetime(ds['time'].values, unit='D', origin=ref_date)


# 添加降水量数据的新列
for i in range(16):
    df[f'precipitation_day_{i}'] = None

# 遍历DataFrame中的每一行
for index, row in df.iterrows():
    # 获取经纬度和日期
    lon = row['lon']
    lat = row['lat']
    date = pd.to_datetime(row['date']).date()

    # 计算前15天的日期范围
    start_date = date - timedelta(days=15)
    date_range = pd.date_range(start=start_date, end=date)

    # 使用线性插值提取特定位置和时间的降水量
    for i, date in enumerate(date_range):
        interpolated_data = ds.sel(time=date, method='nearest').interp(lat=lat, lon=lon)
        precipitation = interpolated_data['precipitationCal'].values.item()
        df.at[index, f'precipitation_day_{15-i}'] = precipitation

sum_columns = df.iloc[:, 4:15].sum(axis=1)



# In[] 
import geopandas as gpd
import random
import datetime
import pandas as pd
from shapely.geometry import Point

df = df_select
# 您原始的代码部分
positive_samples = pd.DataFrame()
positive_samples['lon'], positive_samples['lat'], positive_samples['date'] = df['lon'], df['lat'], df['date']
positive_samples['label'] = 1
num_positive_samples = len(positive_samples)

# 读取 Shapefile 文件
shapefile = gpd.read_file('D:/map/yunnan.shp')
bbox = shapefile.total_bounds

# 收集原始数据中的所有日期
existing_dates = set(positive_samples['date'])

# 创建空列表来存储负样本
random_negative_samples = []

# 时间范围
start_date = datetime.datetime(2016, 1, 1)
end_date = datetime.datetime(2016, 12, 31)

# 生成随机负样本
while len(random_negative_samples) < num_positive_samples:
    random_longitude = random.uniform(bbox[0], bbox[2])
    random_latitude = random.uniform(bbox[1], bbox[3])
    random_point = Point(random_longitude, random_latitude)

    if shapefile.geometry.contains(random_point).any():
        # 随机生成时间，并确保它不在原始数据中
        while True:
            random_date = start_date + datetime.timedelta(days=random.randint(0, (end_date - start_date).days))
            if random_date not in existing_dates:
                break

        random_negative_samples.append((random_date, random_longitude, random_latitude))

# 创建负样本 DataFrame
negative_samples = pd.DataFrame(random_negative_samples, columns=['date', 'lon', 'lat'])
negative_samples['label'] = 0

# 合并正样本和负样本
training_data = pd.concat([positive_samples, negative_samples], ignore_index=True)

# 调整时间格式并保存
training_data['date'] = training_data['date'].apply(lambda x: x.replace(tzinfo=None))
training_data.to_csv('D:/ei_lanslide/traindata2016.csv')

# In[]
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
# 加载NetCDF文件
nc_file_path = 'D:/ei/GPM-2016-day.nc'  # 请替换为实际的文件路径
ds = xr.open_dataset(nc_file_path, decode_times=False)

# 将NetCDF中的时间转换为日期
ref_date = datetime(1970, 1, 1)
ds['time'] = pd.to_datetime(ds['time'].values, unit='D', origin=ref_date)

# 加载Excel文件
excel_file_path = 'D:/ei_lanslide/traindata2016.csv'  # 请替换为实际的文件路径
df = pd.read_csv(excel_file_path)

# 添加降水量数据的新列
for i in range(16):
    df[f'precipitation_day_{i}'] = None

# 遍历DataFrame中的每一行
for index, row in df.iterrows():
    # 获取经纬度和日期
    lon = row['lon']
    lat = row['lat']
    date = pd.to_datetime(row['date']).date()

    # 计算前15天的日期范围
    start_date = date - timedelta(days=15)
    date_range = pd.date_range(start=start_date, end=date)

    # 使用线性插值提取特定位置和时间的降水量
    for i, date in enumerate(date_range):
        interpolated_data = ds.sel(time=date, method='nearest').interp(lat=lat, lon=lon)
        precipitation = interpolated_data['precipitationCal'].values.item()
        df.at[index, f'precipitation_day_{15-i}'] = precipitation


# In[]
#易发性指数
#得到易发性指数
import xarray as xr
import pandas as pd
import os
import rasterio
import rioxarray as rxr

data= df
easy_ds = rxr.open_rasterio('D:/ei_lanslide/yunnan_FR_lat.tif')
data['easy_value'] = None
# 遍历DataFrame中的每一行
for index, row in data.iterrows():
    # 获取经纬度和日期
    lon = row['lon']
    lat = row['lat']
    temp_data = easy_ds.sel(y=lat, x=lon, method='nearest')
    easy_value = temp_data.values
    data.at[index,'easy_value'] = easy_value

data['easy_value'] = data['easy_value'].astype(float)

data = data[data['easy_value'] != easy_ds._FillValue]


k = 0.84
n_days = 11

def calculate_Pa(row, K, n):
    Pa = 0
    for i in range(n + 1):
        Pa += (K ** i) * row[f'precipitation_day_{i}']
        
    return Pa

data['Pa'] = data.apply(lambda row: calculate_Pa(row, k,n_days), axis=1)

#添加原始数据
# 获取所有tif文件
points_df = data
tif_folder_path = 'D:/ei_lanslide/factors/'
tif_files = [f for f in os.listdir(tif_folder_path) if f.endswith('.tif')]

# 创建一个空的DataFrame来存储结果
result_df = pd.DataFrame()

for tif_file in tif_files:
    tif_file_full_path = os.path.join(tif_folder_path, tif_file)

    # 打开tif文件
    ds = xr.open_rasterio(tif_file_full_path)


    for index, point in points_df.iterrows():
        # 获取点坐标
        lon, lat = point['lon'], point['lat']

        # 读取该点的值
        tif_value = ds.sel(x=lon, y=lat, method='nearest').values.item()

        # 将结果添加到DataFrame中
        result_df.at[index, tif_file.split('.')[0]] = tif_value


data = pd.concat([points_df, result_df], axis=1)


#获得FR指数
# 获取所有tif文件
points_df = data
tif_folder_path = 'D:/ei_lanslide/factors_FR/'
tif_files = [f for f in os.listdir(tif_folder_path) if f.endswith('lon.tif')]

# 创建一个空的DataFrame来存储结果
result_df = pd.DataFrame()

for tif_file in tif_files:
    tif_file_full_path = os.path.join(tif_folder_path, tif_file)

    # 打开tif文件
    ds = xr.open_rasterio(tif_file_full_path)


    for index, point in points_df.iterrows():
        # 获取点坐标
        lon, lat = point['lon'], point['lat']

        # 读取该点的值
        tif_value = ds.sel(x=lon, y=lat, method='nearest').values.item()

        # 将结果添加到DataFrame中
        result_df.at[index, tif_file.split('_')[0]+'_FR'] = tif_value


data = pd.concat([points_df, result_df], axis=1)







# 保存更新后的DataFrame到一个新的Excel文件
output_excel_file_path = 'D:/ei_lanslide/updated_traindata2016.xlsx'
data.to_excel(output_excel_file_path, index=False)























