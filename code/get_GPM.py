
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

# 保存更新后的DataFrame到一个新的Excel文件
output_excel_file_path = 'D:/ei_lanslide/updated_traindata2016.xlsx'
df.to_excel(output_excel_file_path, index=False)


