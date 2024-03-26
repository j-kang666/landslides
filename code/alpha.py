# In[]
#确定n值
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取Excel文件
excel_file_path = 'D:/ei_lanslide/updated_traindata2016.xlsx'  # 替换成你的Excel文件路径
df = pd.read_excel(excel_file_path)
df = df[df['label'] == 1]
# 假设Excel文件中有前15天的降水数据，列名从'day1'到'day15'
precipitation_columns = ['precipitation_day_0',
       'precipitation_day_1', 'precipitation_day_2', 'precipitation_day_3',
       'precipitation_day_4', 'precipitation_day_5', 'precipitation_day_6',
       'precipitation_day_7', 'precipitation_day_8', 'precipitation_day_9',
       'precipitation_day_10', 'precipitation_day_11', 'precipitation_day_12',
       'precipitation_day_13', 'precipitation_day_14', 'precipitation_day_15']
for n in range(1,17):
    # 计算前n天的累计降水量
    df['cumulative_precipitation'] = df[precipitation_columns[0:n]].sum(axis=1)
    
    # 统计各样本在各降水等级中出现的次数
    precipitation_levels = [0, 25,50,75,125,150, 200,300]
    df['precipitation_level'] = pd.cut(df['cumulative_precipitation'], bins=precipitation_levels)
    histogram = df['precipitation_level'].value_counts(sort=False)
    
    max_disaster_index = np.argmax(histogram)
    
    print(n)
    print(precipitation_levels[max_disaster_index], precipitation_levels[max_disaster_index + 1])
    # 打印结果
    print(histogram.max())
    
    # 绘制直方图
    plt.bar(precipitation_levels[:-1], histogram, width=20)
    plt.xlabel('Cumulative Precipitation Levels')
    plt.ylabel('Number of Disasters')
    plt.title('Histogram of Cumulative Precipitation for Landslides')
    plt.show()
'''
# In[]

import numpy as np
import pandas as pd
# 假设的降雨量数据，这里用随机数代替，实际应用中应使用实际降雨量数据
n = 11  # 时间间隔数
file_path = 'D:/ei_lanslide/updated_traindata2016.xlsx'
data = pd.read_excel(file_path)
data = data[data['label'] == 1]
rainfall_columns = ['precipitation_day_{}'.format(i) for i in range(0, 15)]
rainfall = data[rainfall_columns].to_numpy()




# 计算给定α时的累积降雨量Rc的函数
def calculate_Rc(alpha, rainfall):
    Rc = rainfall[:,0] + rainfall[:,1] + rainfall[:,2]
    for i in range(3, n):
        Rc += alpha**(i-2) * rainfall[:,i]
    return Rc

