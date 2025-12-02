import numpy as np
import  pandas as  pd
from datetime import datetime
def data_preprocessing():
    data=pd.read_csv('C:/Users/何/Desktop/工作文件/计算机/demo/load_predict_project/data/train.csv')
    # data.info()
    # print(data.head())

    #格式化时间
    data['time'] = pd.to_datetime(data['time'])
    # print(data.head())

    #肾虚
    data.sort_values('time',ascending=True,inplace=True)
    # by：要排序的列名或列名列表
    # ascending：是否升序排列（True = 升序，False = 降序）
    # inplace：是否在原DataFrame上修改（True = 直接修改，False = 返回新DataFrame）

    #去重
    data.drop_duplicates(inplace=True)
    # subset：基于哪些列判断重复（默认所有列）
    # keep：保留哪个重复项（'first' = 第一个，'last' = 最后一个，False = 全部删除）
    # inplace：是否在原DataFrame上修改
    return data








if __name__=='__main__':
    data_preprocessing()