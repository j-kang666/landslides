# In[]
#ȷ��nֵ
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ��ȡExcel�ļ�
excel_file_path = 'D:/ei_lanslide/updated_traindata2016.xlsx'  # �滻�����Excel�ļ�·��
df = pd.read_excel(excel_file_path)
df = df[df['label'] == 1]
# ����Excel�ļ�����ǰ15��Ľ�ˮ���ݣ�������'day1'��'day15'
precipitation_columns = ['precipitation_day_0',
       'precipitation_day_1', 'precipitation_day_2', 'precipitation_day_3',
       'precipitation_day_4', 'precipitation_day_5', 'precipitation_day_6',
       'precipitation_day_7', 'precipitation_day_8', 'precipitation_day_9',
       'precipitation_day_10', 'precipitation_day_11', 'precipitation_day_12',
       'precipitation_day_13', 'precipitation_day_14', 'precipitation_day_15']
for n in range(1,17):
    # ����ǰn����ۼƽ�ˮ��
    df['cumulative_precipitation'] = df[precipitation_columns[0:n]].sum(axis=1)
    
    # ͳ�Ƹ������ڸ���ˮ�ȼ��г��ֵĴ���
    precipitation_levels = [0, 25,50,75,125,150, 200,300]
    df['precipitation_level'] = pd.cut(df['cumulative_precipitation'], bins=precipitation_levels)
    histogram = df['precipitation_level'].value_counts(sort=False)
    
    max_disaster_index = np.argmax(histogram)
    
    print(n)
    print(precipitation_levels[max_disaster_index], precipitation_levels[max_disaster_index + 1])
    # ��ӡ���
    print(histogram.max())
    
    # ����ֱ��ͼ
    plt.bar(precipitation_levels[:-1], histogram, width=20)
    plt.xlabel('Cumulative Precipitation Levels')
    plt.ylabel('Number of Disasters')
    plt.title('Histogram of Cumulative Precipitation for Landslides')
    plt.show()
'''
# In[]

import numpy as np
import pandas as pd
# ����Ľ��������ݣ���������������棬ʵ��Ӧ����Ӧʹ��ʵ�ʽ���������
n = 11  # ʱ������
file_path = 'D:/ei_lanslide/updated_traindata2016.xlsx'
data = pd.read_excel(file_path)
data = data[data['label'] == 1]
rainfall_columns = ['precipitation_day_{}'.format(i) for i in range(0, 15)]
rainfall = data[rainfall_columns].to_numpy()




# ���������ʱ���ۻ�������Rc�ĺ���
def calculate_Rc(alpha, rainfall):
    Rc = rainfall[:,0] + rainfall[:,1] + rainfall[:,2]
    for i in range(3, n):
        Rc += alpha**(i-2) * rainfall[:,i]
    return Rc

