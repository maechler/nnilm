import argparse
import os
import yaml
import numpy as np
from keras.models import load_model
from neuralnilm.metrics import Metrics
from nnilm.metrics import rectangular_metrics
from nnilm.utils import create_data_pipeline
from nnilm.plotter import plot_prediction, plot_weights, plot_activations as plotter_plot_activations


def predict(device_name, sample_period=10, num_seq_per_batch=512, save_path=None, verbose=True, plot_predictions=[], plot_layer_weights=[], plot_activations=False):
    file_dir = os.path.dirname(__file__)
    conf_filepath = os.path.join(file_dir, 'device_config/{}.json'.format(device_name))
    plots_save_path = os.path.join(save_path, 'plots')

    if not os.path.isfile(conf_filepath):
        print('No config found for device "{}". Please see "nnilm/device_config/".'.format(device_name))
        exit(1)

    with open(conf_filepath) as config_file:
        conf = yaml.safe_load(config_file)

    pipeline, real_input_std, real_target_std, real_avg_power = create_data_pipeline(
        conf=conf,
        sample_period=sample_period,
        num_seq_per_batch=num_seq_per_batch,
        source_probabilities=(0, 1),  # only use real data
        windows_key='predict_windows'
    )
    test_batch = pipeline.get_batch(fold='unseen_appliances')

    model = load_model(save_path + 'checkpoints/{}-BEST.hdf5'.format(device_name))
    pred = model.predict(test_batch.input)

    activation_threshold = conf['activation_threshold'] / float(conf['target_std'])
    my_metrics = rectangular_metrics(pred, test_batch.target, print_results=False, activation_threshold=activation_threshold)
    metrics = Metrics(state_boundaries=[activation_threshold])
    results = metrics.compute_metrics(np.reshape(pred, [num_seq_per_batch, 3, 1]), test_batch.target)

    if verbose:
        classification = results['classification_2_state']
        regression = results['regression']

        print('real_input_std: ' + str(real_input_std))
        print('real_target_std: ' + str(real_target_std))
        print('real_avg_power: ' + str(real_avg_power))
        print('')
        print('TP: ' + str(my_metrics['tp']))
        print('FP: ' + str(my_metrics['fp']))
        print('TN: ' + str(my_metrics['tn']))
        print('FN: ' + str(my_metrics['fn']))
        print('')
        print('Accuracy: ' + str(classification['accuracy_score']))
        print('F1: ' + str(classification['f1_score']))
        print('Precision: ' + str(classification['precision_score']))
        print('Recall: ' + str(classification['recall_score']))
        print('')
        print('Mean absolute error: ' + str(regression['mean_absolute_error'] * conf['target_std']))
        print('Relative error in total energy: ' + str(regression['relative_error_in_total_energy']))

    for sample_index in plot_predictions:
        plot_prediction(
            data=test_batch.before_processing.target[sample_index],
            series_label='Appliance power',
            target=test_batch.target[sample_index],
            prediction=pred[sample_index],
            target_std=conf['target_std'],
            sequence_length=conf['seq_length'],
            save_path=plots_save_path,
        )

        plot_prediction(
            data=test_batch.before_processing.input[sample_index],
            series_label='Aggregate power',
            save_path=plots_save_path,
        )

    for layer_index in plot_layer_weights:
        plot_weights(
            model.get_layer(index=layer_index).get_weights()[0],
            save_path=plots_save_path,
        )

    if plot_activations:
        index = 0
        positive_sample_index = 0
        negative_sample_index = 0

        for test_sample in test_batch.target:
            if test_sample[2] == 0:
                negative_sample_index = index
            elif test_sample[2] > 0:
                positive_sample_index = index

            index = index + 1

            if positive_sample_index > 0 and negative_sample_index > 0:
                break

        positive_sample = test_batch.input[positive_sample_index].reshape(1, test_batch.input.shape[1], 1)
        negative_sample = test_batch.input[negative_sample_index].reshape(1, test_batch.input.shape[1], 1)

        plotter_plot_activations(model, positive_sample, plots_save_path)
        plotter_plot_activations(model, negative_sample, plots_save_path)

        plot_prediction(
            data=test_batch.before_processing.target[positive_sample_index],
            series_label='Appliance power',
            target=test_batch.target[positive_sample_index],
            prediction=pred[positive_sample_index],
            target_std=conf['target_std'],
            sequence_length=conf['seq_length'],
            save_path=plots_save_path,
        )

        plot_prediction(
            data=test_batch.before_processing.input[positive_sample_index],
            series_label='Aggregate power',
            save_path=plots_save_path,
        )

        plot_prediction(
            data=test_batch.before_processing.target[negative_sample_index],
            series_label='Appliance power',
            target=test_batch.target[negative_sample_index],
            prediction=pred[negative_sample_index],
            target_std=conf['target_std'],
            sequence_length=conf['seq_length'],
            save_path=plots_save_path,
        )

        plot_prediction(
            data=test_batch.before_processing.input[negative_sample_index],
            series_label='Aggregate power',
            save_path=plots_save_path,
        )

    if verbose:
        print('')
        print('Showing first five predictions and ground truth:')
        print('Prediction:')
        print(pred[:5])
        print('Ground truth:')
        print(np.reshape(test_batch.target[:5], [5, 3]))

    return pred, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('device_name', help='The name of the device that should be trained, see "nnilm/device_config".', type=str)

    args = parser.parse_args()

    predict(args.device_name)
