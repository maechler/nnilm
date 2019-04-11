import numpy as np
import itertools
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime,timedelta
from keras.models import load_model
import os
import yaml
import time


def predict(device_name, input_data_path, start_date, end_date):
    date_format = '%d/%m/%Y %H:%M:%S'
    input_date_format = '%d-%m-%Y'
    file_dir = os.path.dirname(__file__)
    save_path = os.path.join(file_dir, '../../experiments/' + device_name + '/')
    conf_filepath = os.path.join(file_dir, '../device_config/{}.json'.format(device_name))
    start_date_obj = datetime.strptime(start_date, input_date_format)
    end_date_obj = datetime.strptime(end_date, input_date_format)
    sampling_rate = 0.1

    with open(conf_filepath) as config_file:
        conf = yaml.safe_load(config_file)

    with open(input_data_path) as file_stream:
        first_row = np.genfromtxt(itertools.islice(file_stream, 0, 1), delimiter=',', names=['datetime', 'power'], dtype=[('datetime', 'S19'), ('power', float)])
        first_date = datetime.strptime(str(first_row['datetime']), date_format)
        samples_start = int((start_date_obj - first_date).total_seconds() * sampling_rate)
        samples_end = int((end_date_obj - first_date).total_seconds() * sampling_rate)

        data = np.genfromtxt(itertools.islice(file_stream, samples_start, samples_end), delimiter=',', names=['datetime', 'power'], dtype=[('datetime', 'S19'), ('power', float)])
        x = [datetime.strptime(date, date_format) for date in data['datetime']]
        figure = plt.figure()
        axes = figure.add_subplot(1, 1, 1)

        plt.plot(x, data['power'], color='#2A638C')

        model = load_model(save_path + 'checkpoints/{}-BEST.hdf5'.format(device_name))
        total_std = 0
        total_power = 0
        total_prediction_time = 0
        positive_samples = 0
        nr_of_data_points = samples_end - samples_start
        nr_of_windows = int(nr_of_data_points / conf['seq_length'])

        if not os.path.exists('out'):
            os.makedirs('out')

        print('Start date: ' + str(start_date_obj))
        print('End date: ' + str(end_date_obj))
        print('# of windows: ' + str(nr_of_windows))

        for i in range(0, nr_of_windows):
            raw_batch = []
            for j in range(0, conf['seq_length']):
                raw_batch.append(data[i*conf['seq_length']+j][1])

            mean = np.mean(raw_batch)
            total_std = total_std + np.array(raw_batch).std()
            total_power = total_power + np.array(raw_batch).sum()
            batch = raw_batch - mean
            batch = batch / conf['input_std']
            np_batch = np.array(batch).reshape(1, conf['seq_length'], 1)
            scaled_threshold = float(conf['activation_threshold']) / float(conf['target_std'])

            start_time = time.time()

            pred = model.predict(np_batch)

            end_time = time.time()
            total_prediction_time = total_prediction_time + (end_time - start_time)

            if pred[0][2] > scaled_threshold and pred[0][0] < pred[0][1]:
                positive_samples = positive_samples + 1
                target = pred[0]
                start_batch = start_date_obj + timedelta(seconds=((1 / sampling_rate) * i * conf['seq_length']))
                target_rect_on = start_batch + timedelta(seconds=((1 / sampling_rate) * target[0] * conf['seq_length']))
                target_rect_off = start_batch + timedelta(seconds=((1 / sampling_rate) * target[1] * conf['seq_length']))
                target_rect_width = target_rect_off - target_rect_on
                target_rect_height = target[2] * conf['target_std']
                target_rect = plt.Rectangle((target_rect_on, 0), target_rect_width, target_rect_height, color='#D01431', fill=None, linestyle='--')

                axes.add_patch(target_rect)
                print('Positive sample: i=' + str(i) + ' start_date=' + data[i*conf['seq_length']][0] + ' start=' + str(target[0]) + ' end=' + str(target[1]) + ' avg_power=' + str(target[2] * conf['target_std']))

        print('')
        print('avg_std=' + str(total_std / nr_of_windows))
        print('avg_power=' + str(total_power / nr_of_windows / conf['seq_length']))
        print('avg_prediction_time=' + str(total_prediction_time / nr_of_windows))
        print('positive_samples_percentage=' + str(float(positive_samples) / float(nr_of_windows)))

        plt.gca().set_xlim([start_date_obj, end_date_obj])
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        #  avoid max ticks error
        if end_date_obj - start_date_obj < timedelta(hours=(2 * 1000)):
            plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
        else:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        figure.savefig('out/csv_predictor.pdf', format='pdf', dpi=400)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--device_name', help='The name of the device that should be tested, see "nnilm/device_config".', type=str)
    parser.add_argument('-i', '--input_data_path', help='Path to the CSV file containing the data.', type=str)
    parser.add_argument('-s', '--start_date', help='Start date in the format d-m-Y.', type=str)
    parser.add_argument('-e', '--end_date', help='End date in the format d-m-Y.', type=str)

    args = parser.parse_args()

    predict(args.device_name, args.input_data_path, args.start_date, args.end_date)
