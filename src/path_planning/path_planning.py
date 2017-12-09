# coding=utf-8
"""
path planning算法实现
"""
from heapq import *
import matplotlib.pyplot as plt
import numpy as np
from src.data_utils import data_util
import matplotlib.animation as animation
from multiprocessing import Process, Pipe, Queue
import matplotlib.pyplot as plt
j = 0
i = 1


def visualization_path(map_day, path, start, goal):
    fig = plt.figure()
    # visualization

    def updatefig(*args):
        global i
        traj_so_far = path[:i]
        i += 1
        if i > len(path):
            i = 1
        (x, y), h = traj_so_far[-1]
        print 'time:{}'.format(h)
        map_data = map_day.hour_map[str(h)]
        map_data = map_data * 125
        map_data[start[0]-5:start[0]+5, start[1]-5:start[1]+5] = 256
        map_data[goal[0]-5:goal[0]+5, goal[1]-5:goal[1]+5] = 256
        map_data[x-5:x+5, y-5:y+5] = 256
        return plt.imshow(map_data, animated=True),
    ani = animation.FuncAnimation(fig, updatefig, interval=20, blit=True)
    plt.show()
    return


def visualization_process(map_day, buffer_data, start, goal):
    fig = plt.figure()
    # visualization

    def updatefig(*args):
        global j
        j += 1
        if j > len(buffer_data):
            j = 0
        current, open_list, close_list = buffer_data[j]
        # map_data = map_day.hour_map[str(h)]
        map_data = map_day
        map_data = map_data * 125
        map_data[start[0]-5:start[0]+5, start[1]-5:start[1]+5] = 256
        map_data[goal[0]-5:goal[0]+5, goal[1]-5:goal[1]+5] = 256
        map_data[current[0][0]-5:current[0][0]+5, current[0][1]-5:current[0][1]+5] = 256
        for item in open_list:
            (x, y) = item
            map_data[x-2:x+2, y-2:y+2] = 256
        for item in close_list:
            (x, y) = item
            map_data[x - 2:x + 2, y - 2:y + 2] = 200
        return plt.imshow(map_data, animated=True),
    ani = animation.FuncAnimation(fig, updatefig, interval=20, blit=True)
    plt.show()
    return


def heuristic(a, b):

    """
    A star的启发性函数, 估计cost to go
    :param a: 点1
    :param b: 点2
    :return: 定义的两点距离
    """
    # return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def distance_between(current, neighbor):
    """
    计算两点之间的距离
    :param current:
    :param neighbor:
    :return:
    """
    return abs(current[0] - neighbor[0]) + abs(current[1] - neighbor[1])*1.5


def astar(input_map, start, destination, h_p=1.5):
    """
    A* algorithm implementation
    usage:
        nmap = numpy()/ 0 for reachable 1 for obstacles
        astar(nmap, (0, 0), (10, 13))
    :param input_map: map grid
    :param start: start point
    :param destination: terminal point
    :return: path/false if no path found
    """
    # to consider diagonal neighbors
    # neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, destination)}
    oheap = []
    heappush(oheap, (fscore[start], start))
    # buffer for visualization
    while oheap:
        current = heappop(oheap)[1]
        # reach goal
        if current == destination:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            # reverse trajectory order
            data.append(start)
            data_reverse = [None]*len(data)
            for i in range(len(data)):
                data_reverse[i] = data[-i-1]
            # return time_stamp, True
            return data_reverse, close_set, True
        close_set.add(current)
        # buffer
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + distance_between(current, neighbor)
            if 0 <= neighbor[0] < input_map.shape[0]:
                if 0 <= neighbor[1] < input_map.shape[1]:
                    if input_map[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # input_map bound y walls
                    continue
            else:
                # input_map bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + h_p*heuristic(neighbor, destination)
                heappush(oheap, (fscore[neighbor], neighbor))


    return None, close_set, False


def astar_with_g_constraint(input_map, start, destination, h_p=1.5, g_max=30):
    """
    A* algorithm implementation
    usage:
        nmap = numpy()/ 0 for reachable 1 for obstacles
        astar(nmap, (0, 0), (10, 13))
    :param input_map: map grid
    :param start: start point
    :param destination: terminal point
    :return: path/false if no path found
    """
    # to consider diagonal neighbors
    # neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, destination)}
    oheap = []
    heappush(oheap, (fscore[start], start))
    # buffer for visualization
    while oheap:
        current = heappop(oheap)[1]
        # reach goal
        if current == destination:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            # reverse trajectory order
            data.append(start)
            data_reverse = [None]*len(data)
            for i in range(len(data)):
                data_reverse[i] = data[-i-1]
            # return time_stamp, True
            return data_reverse, close_set, True
        close_set.add(current)
        # buffer
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + distance_between(current, neighbor)
            if 0 <= neighbor[0] < input_map.shape[0]:
                if 0 <= neighbor[1] < input_map.shape[1]:
                    if input_map[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # input_map bound y walls
                    continue
            else:
                # input_map bound x walls
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + h_p*heuristic(neighbor, destination)
                heappush(oheap, (fscore[neighbor], neighbor))


    return None, close_set, False



def dynamic_planning(day_map, destination, verbose=True):
    """
    动态A*算法，每小时根据天气重新进行一次A*规划
    :param day_map: 当天所有时段天气
    :param destination:
    :param verbose:
    :return:
    """
    start_point = (142, 328)
    global_trajectory = []
    # local_trajectory = None
    trajectory_time = []
    max_step_in_an_hour = 60 / 2
    for hour_iter in range(3, 21):
        hour_map = day_map.hour_map[str(hour_iter)]
        local_trajectory, ret = astar(hour_map, start_point, destination)
        # 一个小时内是否能走完路径
        # 确保每个独立时辰内都能重新规划到一条可通达路径
        if ret:
            # 记录轨迹时间
            trajectory_time.append(hour_iter)
            if verbose:
                print 'local path planning at hour:{}'.format(hour_iter)
            if len(local_trajectory) > max_step_in_an_hour:
                for item_iter in xrange(max_step_in_an_hour):
                    path_point, min = local_trajectory[item_iter]
                    global_trajectory.append((path_point, (hour_iter, min)))
                start_point = local_trajectory[30][0]
            else:
                for item_iter in local_trajectory:
                    path_point, min = item_iter
                    global_trajectory.append((path_point, (hour_iter, min)))
                # start_time = trajectory_time[0]
                # return (global_trajectory, start_time)
                return global_trajectory, True
        else:
            # TODO增加当所有单独小时内都没有可能达路径的解决方法
            if verbose:
                print 'cant not find available path in hour:{}'.format(hour_iter)
            # return False
    return global_trajectory, False


def time_vary_astar(map_day, destination, queue=None):
    """
    time-varying A* algorithm implementation
    usage:
        nmap = numpy()/ 0 for reachable 1 for obstacles
        astar(nmap, (0, 0), (10, 13))
    :param input_map: map grid
    :param start: start point
    :param destination: terminal point
    :return: path/false if no path found
    """
    start_point = (142, 328)
    # to consider diagonal neighbors
    # neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    start = (start_point, 3)
    # destination = (destination, None)
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start[0], destination)}
    oheap = []
    heappush(oheap, (fscore[start], start))
    process_buffer = []
    while oheap:
        current = heappop(oheap)[1]
        # print 'time elapse:{}, hour:{}'.format(time_elaps, time_hour)
        if current == destination:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            data_reverse = [None]*len(data)
            for i in range(len(data)):
                data_reverse[i] = data[-i-1]
            return data_reverse, process_buffer, True
            # return time_stamp

        close_set.add(current)
        time_elapse = gscore[current]*2 + 2
        print 'time——elapse：{}'.format(time_elapse)
        time_hour = 3 + time_elapse // 60
        time_min = time_elapse % 60
        current_map = map_day.hour_map[str(int(time_hour))]
        process_buffer.append((current, time_hour, oheap))
        if time_hour >= 21:
            return None, process_buffer, False
        for i, j in neighbors:
            neighbor = ((current[0][0] + i, current[0][1] + j), time_hour)
            tentative_g_score = gscore[current] + distance_between(current[0], neighbor[0])
            # 无视越界好障碍
            if 0 <= neighbor[0][0] < current_map.shape[0]:
                if 0 <= neighbor[0][1] < current_map.shape[1]:
                    # obstacle
                    if current_map[neighbor[0][0]][neighbor[0][1]] == 1:
                        continue
                else:
                    # input_map bound y walls
                    continue
            else:
                # input_map bound x walls
                continue
            # 判断是否在close set中
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            # 更新的两种情况:不在open或者在则检查g值
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor[0], destination)
                heappush(oheap, (fscore[neighbor], neighbor))

    return None, None, False


def close_list_visualization(map, close_list, start, goal):
    close_list = list(close_list)
    map = map * 125
    for item in close_list:
        x, y = item
        map[x, y] = 200
    map[start[0] - 5:start[0] + 5, start[1] - 5:start[1] + 5] = 256
    map[goal[0] - 5:goal[0] + 5, goal[1] - 5:goal[1] + 5] = 256
    plt.imshow(map)
    plt.show()
    return

if __name__ == '__main__':
    # parent_conn, child_conn = Pipe()
    city_list = data_util.get_city_list()
    measure_data = data_util.load_data_from_csv(1)
    map_day = data_util.DayMap(data_set=measure_data, day_id=3)
    # path, buffer, ret = time_vary_astar(map_day, city_list[3])
    # time_stamp, ret = dynamic_planning(map_day, city_list[5])
    for city_iter in xrange(1, 11):
        path, buffer_list, ret = astar(map_day.hour_map['3'], city_list[0], city_list[city_iter])
        close_list_visualization(map_day.hour_map['3'], buffer_list, city_list[0], city_list[city_iter])
        # visualization_process(map_day, buffer_list, city_list[0], city_list[2])
    print path
    print ret
    # time_stamp = dynamic_planning(map_day, city_list[1])
