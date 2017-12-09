# coding=utf-8
"""
main function implementation
"""
from src.data_utils import data_util
from src.path_planning import path_planning
import numpy as np
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt


class Config(object):
    """
    路径生成的参数
    """
    data_set_id = 2 # or 0 for training data set
    model_id = 2
    if data_set_id == 1:
        model_id = None
    danger_bias = 0
    if data_set_id == 2:
        testing = True
    else:
        testing = False

config = Config()

# global data set
measure_data = data_util.load_data_from_csv(config.data_set_id)


def generate_result(day, verbose=True):
    """
    以天为单位, 从预测真值天气生成路径
    :param day:
    :param verbose:
    :return:
    """
    fail_list = []
    # prepare data
    city_list = data_util.get_city_list()
    df = pd.DataFrame(data=None, index=None, columns=['CityId', 'Day', 'Hour', 'x', 'y'])
    # pack map of a day unit
    for day_iter in range(day, day+1):
        map_day = data_util.DayMap(data_set=measure_data, day_id=day_iter, model_id=config.model_id, danger_bias=config.danger_bias, testing_data_set=config.testing)
        for city_iter in range(1, 11):
            # dynamic path planning
            if verbose:
                print 'Generating trajectory for Day:{} City:{}'.format(day_iter, city_iter)
            trajectory_result = path_planning.dynamic_planning(map_day, destination=city_list[city_iter])
            # insert csv data frame
            if trajectory_result is not False:
                trajectory = trajectory_result
                for traj_iter in range(len(trajectory)):
                    (x, y), (hour, min) = trajectory[traj_iter]
                    time_stamp = '{:02d}:{:02d}'.format(hour, min)
                    df_index_len = len(df.index)
                    df.loc[df_index_len] = [np.int(city_iter), np.int(day_iter+5), time_stamp, np.int(x), np.int(y)]
            else:
                # insert NaN when no trajectory available
                # df_index_len = len(df.index)
                # df.loc[df_index_len] = [city_iter, day_iter+5, np.nan, np.nan, np.nan]
                fail_list.append('city:{}-day:{}'.format(city_iter, day_iter+5))
                pass
    csv_name = '/home/v5/pycharm_projects/airdrone/data/result/result_day_{}.csv'.format(day)
    df.to_csv(csv_name, index=False)
    return


def generate_result_multi_p():
    pool = Pool(5)
    pool.map(generate_result, range(1, 6))


def result_merge():
    """
    merger result from all days
    :return:
    """
    result_dir = '/home/v5/pycharm_projects/airdrone/data/result/'
    df_all = []
    for day_iter in range(1, 6):
        file_name = result_dir + 'result_day_{}.csv'.format(day_iter)
        df = pd.read_csv(file_name)
        df_all.append(df)
    df = pd.concat(df_all)
    to_name = result_dir + 'result_all.csv'
    df.to_csv(to_name, header=False, index=False)


if __name__ == '__main__':
    generate_result(1)
    pass
