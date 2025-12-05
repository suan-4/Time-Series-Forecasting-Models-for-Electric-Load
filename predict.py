import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from utils.log import Logger
from utils.common import data_preprocessing


class PowerLoadPredictor:
    def __init__(self):
        self.model = None
        self.logger = Logger('../', 'predict').get_logger()
        
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(project_root, 'model', 'xgb_model.pkl')
            
        try:
            self.model = joblib.load(model_path)
            self.logger.info(f'成功加载模型: {model_path}')
            print(f'成功加载模型: {model_path}')
        except Exception as e:
            self.logger.error(f'加载模型失败: {e}')
            print(f'加载模型失败: {e}')
            return False
        return True
    
    def prepare_features_for_prediction(self, predict_time, historical_data):
        historical_data['time'] = pd.to_datetime(historical_data['time'])
        historical_data = historical_data.sort_values('time').reset_index(drop=True)
        

        historical_data['month'] = historical_data['time'].dt.month
        historical_data['day'] = historical_data['time'].dt.day
        historical_data['hour'] = historical_data['time'].dt.hour
        historical_data['weekday'] = historical_data['time'].dt.weekday
        historical_data['year'] = historical_data['time'].dt.year 

        features = {}
        
        predict_timestamp = pd.Timestamp(predict_time)
        features['hour'] = predict_timestamp.hour
        features['weekday'] = predict_timestamp.weekday()
        features['month'] = predict_timestamp.month
        features['day'] = predict_timestamp.day
        features['is_weekend'] = 1 if predict_timestamp.weekday() >= 5 else 0
        
        # 季节特征
        if predict_timestamp.month in [3, 4, 5]:
            features['season'] = 1  # 春季
        elif predict_timestamp.month in [6, 7, 8]:
            features['season'] = 2  # 夏季
        elif predict_timestamp.month in [9, 10, 11]:
            features['season'] = 3  # 秋季
        else:
            features['season'] = 4  # 冬季
            
        # 时间段特征
        if 6 <= predict_timestamp.hour < 12:
            features['time_period'] = 1  # 上午
        elif 12 <= predict_timestamp.hour < 18:
            features['time_period'] = 2  # 下午
        elif 18 <= predict_timestamp.hour < 24:
            features['time_period'] = 3  # 晚上
        else:
            features['time_period'] = 4  # 凌晨
        
        latest_time = historical_data['time'].max()
        if latest_time < predict_timestamp:
            self.logger.warning('历史数据时间晚于预测时间，可能影响预测准确性')
            
        # 滞后特征
        # lag1_time = predict_timestamp - pd.Timedelta(hours=1)
        # lag24_time = predict_timestamp - pd.Timedelta(hours=24)
        # lag168_time = predict_timestamp - pd.Timedelta(hours=168)  
        
        # 在历史数据中查找最接近的记录
        def find_closest_load(target_time):
            if len(historical_data[historical_data['time'] <= target_time]) > 0:
                closest_row = historical_data[historical_data['time'] <= target_time].iloc[-1]
                return closest_row['power_load']
            else:
                self.logger.warning(f'找不到最接近的记录于{target_time}')
                # 如果找不到，使用全局平均值
                return historical_data['power_load'].mean()
                
        features['load_lag1'] = find_closest_load(lag1_time)
        features['load_lag24'] = find_closest_load(lag24_time)
        features['load_lag168'] = find_closest_load(lag168_time)
        
        # 关键：计算负荷变化率特征
        lag2_time = lag1_time - pd.Timedelta(hours=1)
        lag25_time = lag24_time - pd.Timedelta(hours=1)
        
        load_lag2 = find_closest_load(lag2_time)
        load_lag25 = find_closest_load(lag25_time)
        
        # 负荷变化率特征（提高突发峰值预测能力）
        features['load_change_rate'] = (features['load_lag1'] - load_lag2) / (load_lag2 + 1e-8)
        features['load_change_rate_24h'] = (features['load_lag1'] - load_lag25) / (load_lag25 + 1e-8)
        
        # 历史同期特征（往年同月同日同时刻负荷）
        target_month = predict_timestamp.month
        target_day = predict_timestamp.day
        target_hour = predict_timestamp.hour
        
        # 查找历史同期数据
        historical_same_period = historical_data[
            (historical_data['month'] == target_month) & 
            (historical_data['day'] == target_day) & 
            (historical_data['hour'] == target_hour)
        ]
        
        if len(historical_same_period) > 0:
            # 使用最近一年的同期数据
            features['load_same_period_last_year'] = historical_same_period.iloc[-1]['power_load']
        else:
            # 如果没有找到历史同期数据，使用全局平均值
            features['load_same_period_last_year'] = historical_data['power_load'].mean()
        
        feature_columns = [
            'hour', 'weekday', 'month', 'season', 'is_weekend', 'time_period',
            # 'load_lag1', 'load_lag24', 'load_lag168', 
            'load_same_period_last_year',
            'load_change_rate', 'load_change_rate_24h'  # 关键的峰值预测特征
        ]
        
        # 构造特征向量
        feature_vector = np.array([features[col] for col in feature_columns]).reshape(1, -1)
        
        return feature_vector, feature_columns
    

    def predict(self, year, month, day, hour):

        if self.model is None:
            self.logger.error('模型未加载')
            print('错误: 模型未加载，请先调用load_model()方法')
            return None
        # 构造预测时间点2
        try:
            predict_time = datetime(year, month, day, hour)
        except ValueError as e:
            self.logger.error(f'无效的日期时间: {year}-{month}-{day} {hour}:00')
            print(f'错误: 无效的日期时间: {e}')
            return None
            
        self.logger.info(f'开始预测 {predict_time} 的电力负荷')
        print(f'开始预测 {predict_time.strftime("%Y-%m-%d %H:%M")} 的电力负荷')
        
        # 加载历史数据用于特征构造
        try:
            historical_data = data_preprocessing()
        except Exception as e:
            self.logger.error(f'加载历史数据失败: {e}')
            print(f'错误: 加载历史数据失败: {e}')
            return None
            
        # 准备特征
        try:
            feature_vector, feature_names = self.prepare_features_for_prediction(predict_time, historical_data)
        except Exception as e:
            self.logger.error(f'特征准备失败: {e}')
            print(f'错误: 特征准备失败: {e}')
            return None
            
        # 进行预测
        try:
            prediction = self.model.predict(feature_vector)[0]
            print(prediction)
            self.logger.info(f'预测完成，结果: {prediction:.2f}')
            print(f'{predict_time.strftime("%Y-%m-%d %H:%M")} 的预测负荷: {prediction:.2f} MW')
            return prediction
        except Exception as e:
            self.logger.error(f'预测过程出错: {e}')
            print(f'错误: 预测过程出错: {e}')
            return None
    

def main():
    predictor = PowerLoadPredictor()

    if not predictor.load_model():
        return

    try:
        print('请输入预测的年份:')
        year = int(input())
        print('请输入预测的月份:')
        month = int(input())
        print('请输入预测的日期:')
        day = int(input())
        print('请输入预测的小时:')
        hour = int(input())
        
        result = predictor.predict(year, month, day, hour)
        if result is not None:
            print(f'预测的功率为: {result:.2f} MW')
    except ValueError:
        print("输入格式错误，请确保输入的都是数字")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


def predict_hour_of_day(year, month, day):

    
    # 创建预测器实例
    predictor = PowerLoadPredictor()
    
    # 加载模型
    if not predictor.load_model():
        print("模型加载失败，无法进行预测")
        return
    
    # 存储预测结果
    hours = list(range(24))  # 0-23小时
    predictions = []
    
    print(f"正在预测{year}年{month}月{day}日24小时的电力负荷...")
    # 对每个小时进行预测
    for hour in hours:
        try:
            pred = predictor.predict(year, month, day, hour)
            if pred is not None:
                predictions.append(pred)
                print(f"{hour:02d}:00: {pred:.2f} MW")
            else:
                predictions.append(np.nan)  # 如果预测失败，添加NaN值
                print(f"{hour:02d}:00: 预测失败")
        except Exception as e:
            print(f"{hour:02d}:00 预测出错: {e}")
            predictions.append(np.nan)
    
    # 过滤有效预测结果
    valid_predictions = [p for p in predictions if not np.isnan(p)]
    
    if not valid_predictions:
        print("所有预测都失败了，无法生成趋势图")
        return
    
    # 检查预测结果是否都相同
    if len(set(valid_predictions)) == 1 and len(valid_predictions) > 1:
        print("警告：所有预测结果都相同，这可能表明模型存在问题或特征工程需要改进")
    
    # 绘制24小时负荷曲线图
    plt.figure(figsize=(14, 7))
    plt.plot(hours, predictions, marker='o', linewidth=2, markersize=6, color='blue')
    plt.title(f'{year}年{month}月{day}日24小时电力负荷预测曲线')
    plt.xlabel('时间 (小时)')
    plt.ylabel('电力负荷 (MW)')
    plt.grid(True, alpha=0.3)
    
    # 设置x轴刻度为整数小时
    plt.xticks(hours, [f"{h:02d}:00" for h in hours], rotation=45)
    
    # 添加数值标签
    for i, (hour, pred) in enumerate(zip(hours, predictions)):
        if not np.isnan(pred):
            plt.annotate(f'{pred:.0f}', (hour, pred), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=7)
    
    # 标记峰值和谷值
    valid_preds_with_index = [(i, pred) for i, pred in enumerate(predictions) if not np.isnan(pred)]
    if valid_preds_with_index:
        max_index, max_value = max(valid_preds_with_index, key=lambda x: x[1])
        min_index, min_value = min(valid_preds_with_index, key=lambda x: x[1])
        
        plt.annotate(f'最大值: {max_value:.0f} MW', 
                    xy=(max_index, max_value), 
                    xytext=(10, 20), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.annotate(f'最小值: {min_value:.0f} MW', 
                    xy=(min_index, min_value), 
                    xytext=(10, -30), 
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(f'picture/daily_load_prediction_{year}_{month:02d}_{day:02d}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # # 打印统计信息
    # if valid_predictions:
    #     print(f"\n{year}年{month}月{day}日预测统计信息:")
    #     print(f"平均负荷: {np.mean(valid_predictions):.2f} MW")
    #     print(f"最高负荷: {np.max(valid_predictions):.2f} MW")
    #     print(f"最低负荷: {np.min(valid_predictions):.2f} MW")
    #     print(f"负荷变化范围: {np.max(valid_predictions) - np.min(valid_predictions):.2f} MW")
    #     print(f"标准差: {np.std(valid_predictions):.2f} MW")
        
    #     # 计算负荷率(平均负荷/最高负荷)
    #     load_rate = np.mean(valid_predictions) / np.max(valid_predictions) * 100
    #     print(f"负荷率: {load_rate:.2f}%")
def predict_hour_of_day_interactive():

    try:
        print("请输入要预测的日期：")
        year = int(input("年份: "))
        month = int(input("月份: "))
        day = int(input("日期: "))
        
        # 调用预测函数
        predict_hour_of_day(year, month, day)
    except ValueError:
        print("输入格式错误，请确保输入的都是数字")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == '__main__':

    predict_hour_of_day_interactive()
    
