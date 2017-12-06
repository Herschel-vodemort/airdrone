# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os


def get_city_list():
    city_list = [(142,328), (84,203), (199,371), (140,234), (236,241), (315,281), (358,207), (363,237), (423,266), (125,375),
                 (189,274)]
    return city_list


def load_data_from_csv(dataset_index=1):
    """
    :param dataset_index: 0, 1, 2, 3 for forecast_training/measurement_data/forecast_testing/city_data
    :return: panda data frame or city list in python tuple
    """
    path_head = '/home/v5/pycharm_projects/airdrone/data/'
    dataset_list = ['ForecastDataforTraining_20171124.csv', 'In-situMeasurementforTraining_20171124.csv',
                    'ForecastDataforTesting_20171124.csv', 'CityData.csv']
    print 'importing dataset:{}...'.format(dataset_list[dataset_index])
    # take almost 1 min to import !!!
    be_time = time.time()
    df = pd.read_csv(path_head + dataset_list[dataset_index])
    print 'data imported, elapse:{} s'.format(time.time() - be_time)
    return df


def get_hour_wind_map(dataset, day_id, hour_id, map_type='danger', danger_range_bias=0, model_id=None):
    """
    根据具体参数提取某个时间的wind map
    :param danger_range_bias: 当输出map type为danger时，定义danger的安全距离，如为1，则安全阀值为15-1=14
    :param save_image:
    :param data: dataset
    :param model_id: None in measure data set
    :param day_id:
    :param hour_id:
    :param map_type: 'value' or 'danger'
    :return: image of map
    """
    d_threshold = 15 - danger_range_bias
    if model_id is not None:
        dataset = dataset[dataset['model'] == model_id]
    df = dataset[dataset['hour'] == hour_id]
    df = df[df['date_id'] == day_id]
    x_max = int(np.max(df['xid']))
    y_max = int(np.max(df['yid']))
    data = df['wind'].reshape(x_max, y_max)
    if map_type != 'value':
        # dangerous zone
        data[data >= d_threshold] = 256
        # safe zone
        data[data < d_threshold] = 0

    return data/256


def generate_map_for_forecast():
    """
    将预测数据集以hour为单位可视化
    :return:
    """
    data_set = load_data_from_csv(2)
    for model_iter in range(1, 11):
        for day_iter in range(1, 6):
            for hour_iter in range(3, 21):
                data = get_hour_wind_map(data_set, day_id=day_iter+5, hour_id=hour_iter, model_id=model_iter)
                path_head = '/home/v5/pycharm_projects/airdrone/data/visualization_image/forecast_data_testing'
                img_name = path_head + '/model_'+str(model_iter)+'-day_'+str(day_iter)+'-hour_'+str(hour_iter)
                print 'generating map of model:{}, day:{}, hour:{}'.format(model_iter, day_iter, hour_iter)
                plt.imsave(img_name, data*256)
    return


class DayMap(object):
    """
    提取某一天的全部时间天气map
    """
    def __init__(self, day_id, data_set, danger_bias=0, model_id=None, training_data_type=True):
        self.hour_map = {}
        # data_set = load_data_from_csv(1)
        if not training_data_type:
            day_id += 5
        for hour_iter in range(3, 21):
            self.hour_map[str(hour_iter)] = get_hour_wind_map(
                data_set, hour_id=hour_iter, day_id=day_id, map_type='danger', model_id=model_id, danger_range_bias=danger_bias)


if __name__ == '__main__':
    generate_map_for_forecast()