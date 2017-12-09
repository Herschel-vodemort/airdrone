# coding=utf-8
import heapq
import matplotlib.pyplot as plt
import numpy as np
from data_util import DayMap, load_data_from_csv, get_city_list
from src.data_utils.data_util import load_data_from_csv, get_hour_wind_map


def evaluate_model_with_predict_acc(top_n=5, verbose=True):
    """
    计算模型在所有区域点上所有时间内的平均准确率
    :param top_n: top n best model
    :return: index of top n model e.g.[2, 4, 5,6,7], evaluate value of all model
    """
    f_train = load_data_from_csv(0)
    m_train = load_data_from_csv(1)

    def _map_diff(map_0, map_1):
        diff_result = np.sum(np.abs(map_0 - map_1)) / float(map_0.size)
        return diff_result
    # params
    day_len = 5
    model_num = 10
    diff = []
    for model_iter in range(1, model_num+1):
        diff_per_model = 0
        for day_iter in range(1, day_len+1):
            if verbose:
                print 'calculating, model:{}    day:{}'.format(model_iter, day_iter)
            for hour_iter in range(3, 21): # 3 am to 21 pm
                map_forecast = get_hour_wind_map(f_train, model_id=model_iter, hour_id=hour_iter, day_id=day_iter, map_type='danger')
                map_measure = get_hour_wind_map(m_train, model_id=None, hour_id=hour_iter, day_id=day_iter, map_type='danger')
                diff_per_model += _map_diff(map_forecast, map_measure)
        diff_per_model /= float(day_len * (21 - 3))
        diff.append(diff_per_model)
        if verbose:
            print 'model:{} Acc:{}'.format(model_iter, 1 - diff_per_model)
    diff = 1 - np.array(diff)
    top_n_index = heapq.nlargest(top_n, range(diff.size), diff.take)
    # np.save('diff.npy', diff)
    # [ 0.9184361   0.91837508  0.92180187  0.9136529   0.91532755  0.91711875  0.91641257  0.91531493  0.91385686  0.9196105 ]
    return top_n_index, diff


def generate_and_map_day(map_day):
    """
    and operation between all maps of hour in a day
    :return:
    """
    map_hour_init = map_day.hour_map['3']
    for hour_iter in range(4, 21):
        map_hour_init = np.logical_and(map_hour_init, map_day.hour_map[str(hour_iter)])*1
    return map_hour_init*75


def city_location_visulization(wind_map):
    city_list = get_city_list()
    map_value = 256
    i = 0
    for item in city_list:
        x, y = item
        if i == 0:
            map_value = 256
        else:
            map_value = 125
        wind_map[x-5:x+5, y-5:y+5] = map_value
        i += 1
    return wind_map


if __name__ == '__main__':
    data_set = load_data_from_csv(2)
    for day_iter in range(1, 6):
        map_day = DayMap(data_set=data_set, day_id=day_iter, model_id=1, training_data_type=False)
        wind_map = generate_and_map_day(map_day)
        wind_map = city_location_visulization(wind_map)
        plt.imsave('mean_map_with_cities_day_{}'.format(day_iter), wind_map)
        plt.imshow(wind_map, 'gray')
        plt.title('map of day:{}'.format(day_iter))
        plt.show()