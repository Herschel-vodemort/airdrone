# coding=utf-8
"""
path planning算法实现
"""
from heapq import *
import matplotlib.pyplot as plt


def heuristic(a, b):
    """
    A star的启发性函数
    :param a: 点1
    :param b: 点2
    :return: 定义的两点距离
    """
    # return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(input_map, start, goal, visualization=False):
    """
    A* algorithm implementation
    usage:
        nmap = numpy()/ 0 for reachable 1 for obstacles
        astar(nmap, (0, 0), (10, 13))
    :param input_map: map grid
    :param start: start point
    :param goal: terminal point
    :return: path/false if no path found
    """
    # to consider diagonal neighbors
    # neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))

    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            if visualization:
                input_map[input_map > 0] = 125 # obstacle
                for item in data:
                    input_map[item] = 256
                plt.imshow(input_map, 'gray')
                plt.show()
            # reverse trajectory order
            data_reverse = [None]*len(data)
            for i in range(len(data)):
                data_reverse[i] = data[-i-1]
            return data_reverse

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
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
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))

    return None


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
        local_trajectory = astar(hour_map, start_point, destination)
        # 一个小时内是否能走完路径
        # 确保每个独立时辰内都能重新规划到一条可通达路径
        if local_trajectory is not None:
            # 记录轨迹时间
            trajectory_time.append(hour_iter)
            if verbose:
                print 'local path planning at hour:{}'.format(hour_iter)
            if len(local_trajectory) > max_step_in_an_hour:
                global_trajectory += [start_point]
                global_trajectory += local_trajectory[:30]
                start_point = local_trajectory[30]
            else:
                global_trajectory += local_trajectory
                start_time = trajectory_time[0]
                return (global_trajectory, start_time)
        else:
            # TODO增加当所有单独小时内都没有可能达路径的解决方法
            if verbose:
                print 'cant not find available path in hour:{}'.format(hour_iter)
            # return False
    return False