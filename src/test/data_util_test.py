import pandas as pd
# import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# data dir
city_data_dir = '../data/CityData.csv'
forecast_data_for_training_dir = '../data/ForecastDataforTraining_20171124.csv'
forecast_data_for_testing_dir = '../data/ForecastDataforTesting_20171124.csv'
true_data_dir = '../data/In-situMeasurementforTraining_20171124.csv'


def sample_windmap_by_day_test(day_index, map_type='value'):
    """
    plot specific day wind map
    :param day_index: day index in range 1 to 5
    :param map_type: 'value', 'danger'(wind>15)
    :return: none
    """
    # load data
    true_data = pd.read_csv(true_data_dir)
    # danger threshold
    d_threshold = 15
    # sample by time
    # 3 am to 21 pm
    for time_itr in range(3, 21):
        df = true_data[true_data['hour'] == time_itr]
        df = df[df['date_id'] == day_index]
        x_max = np.max(df['xid'])
        y_max = np.max(df['yid'])
        data = df['wind'].reshape(x_max, y_max)
        if map_type != 'value':
            # dangerous zone
            data[data >= d_threshold] = 255
            # safe zone
            data[data < d_threshold] = 0
            center = 125
        else:
            center = d_threshold
        plt.subplot(2, 9, time_itr-2)
        plt.title('wind map of day-{} time-{}'.format(day_index, time_itr))
        sns.heatmap(data, center=center)
        plt.xlabel('yid')
        plt.ylabel('xid')
    plt.show()
    # def location_visualize():


def evaluate_model(top_n=5, visualization=True):
    """
    evaluate 10 predictive model by analyse measure data for training and measurement for training
    some visualization
    :param top_n:
    :param visualization:
    :return:
    """


if __name__ == '__main__':
    sample_windmap_by_day(1, map_type='danger')