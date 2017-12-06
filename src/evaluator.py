# coding=utf-8
from src.data_utils.data_util import load_data_from_csv, get_hour_wind_map, get_city_list, DayMap
from src.path_planning.path_planning import dynamic_planning
import numpy as np
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt


def visualization_path_from_result():
    """
    将最终生成的路径可视化
    :return:
    """
    result_dir_head = '/home/v5/pycharm_projects/airdrone/data/result/'
    for day_iter in range(1, 6):
        measure_data = load_data_from_csv()
        map_day = DayMap(data_set=measure_data, day_id=day_iter)
        result_dir = result_dir_head + 'result_day_{}.csv'.format(day_iter)
        day_df = pd.read_csv(result_dir)
        for city_iter in range(1, 11):
            map_visualization = np.zeros_like(map_day.hour_map['3'])
            result_df = day_df[day_df['CityId'] == city_iter]
            time_list = np.array(result_df['Hour'])
            start_hour = int(time_list[0].split(':')[0])
            end_hour = int(time_list[-1].split(':')[0])
            for hour_iter in range(start_hour, end_hour+1):
                s = '{:02d}:00'.format(start_hour)
                next_hour = '{:02d}:00'.format(start_hour+1)
                zone_trajectory = result_df[result_df['Hour']<next_hour]
                zone_trajectory = zone_trajectory[zone_trajectory['Hour']>=s]
                zone_trajectory_x = np.array(zone_trajectory['x'])
                zone_trajectory_y = np.array(zone_trajectory['y'])
                zone_x_min = int(np.min(zone_trajectory_x)-1)
                zone_x_max = int(np.max(zone_trajectory_x)+1)
                zone_y_min = int(np.min(zone_trajectory_y)-1)
                zone_y_max = int(np.max(zone_trajectory_y)+1)
                # wind map
                map_visualization[zone_x_min:zone_x_max, zone_y_min:zone_y_max] \
                    = 125*map_day.hour_map[str(start_hour)][zone_x_min:zone_x_max, zone_y_min:zone_y_max]
                # trajectory
                zone_traj_len = len(zone_trajectory_x)
                for i in range(zone_traj_len):
                    x_item = int(zone_trajectory_x[i])
                    y_item = int(zone_trajectory_y[i])
                    map_visualization[x_item, y_item] = 256
            plt.imshow(map_visualization, 'gray')
            plt.show()


def result_evaluator(verbose=True):
    """
    将生成的path csv文件中的路径与真实天气进行对比，计算最终得分
    计分公式：
        score = 24*60*(count of crash drone) + sum of time(drone reach goal)
    :return: score
    """
    score = 0
    crash_sum = 0
    # measure_data = load_data_from_csv(1)
    measure_data = load_data_from_csv(2)
    result_dir_head = '/home/v5/pycharm_projects/airdrone/data/result/'
    for day_iter in range(1, 6):
        map_day = DayMap(data_set=measure_data, day_id=day_iter, model_id=1, training_data_type=False)
        result_dir = result_dir_head + 'result_day_{}.csv'.format(day_iter)
        day_result = pd.read_csv(result_dir)
        for city_iter in range(1, 11):
            crash_flag = False
            city_result = day_result[day_result['CityId'] == city_iter]
            time_list = np.array(city_result['Hour'])
            # 判断是否生成了路径
            if time_list.size > 10:
                start_hour = int(time_list[0].split(':')[0])
                end_hour = int(time_list[-1].split(':')[0])
                for hour_iter in range(start_hour, end_hour + 1):
                    # 提取生成轨迹中某个时间段内的轨迹
                    s_string = '{:02d}:00'.format(start_hour)
                    next_sting = '{:02d}:00'.format(start_hour + 1)
                    trajectory_in_hour = city_result[city_result['Hour'] < next_sting]
                    trajectory_in_hour = trajectory_in_hour[trajectory_in_hour['Hour'] >= s_string]
                    zone_trajectory_x = np.array(trajectory_in_hour['x'], np.int16)
                    zone_trajectory_y = np.array(trajectory_in_hour['y'], np.int16)
                    # 提取真值天气在轨迹中的天气状况（是否大于15）
                    map_hour = map_day.hour_map[str(start_hour)]
                    weather_list = [map_hour[zone_trajectory_x[i], zone_trajectory_y[i]] for i in xrange(zone_trajectory_x.size)]
                    # 判断是否crash
                    if np.sum(weather_list) > 0:
                        score += 24*60
                        crash_sum += 1
                        crash_flag = True
                        break
                    else:
                        score += 2*len(weather_list)
            else:
                crash_sum += 1
                score += 24 * 60
                crash_flag = True
            if verbose:
                print 'path evaluating for day:{} city:{} crash:{}'.format(day_iter, city_iter, crash_flag)
    print 'final score:{}, {}/10*5'.format(score, crash_sum)
    return score


if __name__ == '__main__':
    result_evaluator()