import matplotlib.pyplot as plt
import os
from keras import models
import numpy as np


colors = {
    'red': '#D01431',
    'yellow': '#F2C430',
    'green': '#5DB64D',
    'grey': '#A6A5A1',
    'blue': '#2A638C',
}


def plot_prediction(data, series_label, target=None, prediction=None, target_std=None, sequence_length=None, save_path=None, ylim=None):
    figure = plt.figure()
    axes = figure.add_subplot(1, 1, 1)

    if target is not None:
        target_rect_on = min(target[0] * sequence_length, sequence_length)
        target_rect_off = min(target[1] * sequence_length, sequence_length)
        target_rect_width = target_rect_off - target_rect_on
        target_rect_height = target[2] * target_std
        target_rect = plt.Rectangle((target_rect_on, 0), target_rect_width, target_rect_height, color=colors['green'], alpha=.5, label='Target')

        axes.add_patch(target_rect)

    if prediction is not None:
        prediction_rect_on = min(prediction[0] * sequence_length, sequence_length)
        prediction_rect_off = min(prediction[1] * sequence_length, sequence_length)
        prediction_rect_width = prediction_rect_off - prediction_rect_on
        prediction_rect_height = prediction[2] * target_std
        prediction_rect = plt.Rectangle((prediction_rect_on, 0), prediction_rect_width, prediction_rect_height, color=colors['red'], alpha=1, label='Prediction', fill=None, linestyle='--')

        axes.add_patch(prediction_rect)

    if ylim is not None:
        axes.set_ylim(ylim)

    axes.set_xlabel('time [10s]', labelpad=10)
    axes.set_ylabel('power [W]', labelpad=10)

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.plot(data, label=series_label, color=colors['blue'])
    plt.legend(loc='best')
    plt.show()

    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        figure.savefig(get_valid_save_path(save_path, series_label, 'pdf'), format='pdf', dpi=400)


# Get weights like this: weights = model.get_layer(index=0).get_weights()[0]
def plot_weights(weights, save_path):
    figure = plt.figure()

    for i in range(0, weights.shape[2], 1):
        axes = figure.add_subplot(4, 4, i+1)
        filter = []

        for j in range(0, weights.shape[0], 1):
            filter.append(weights[j][0][i])

        axes.set_ylim([-0.5, 0.5])

        if i < 12:
            axes.set_xticklabels([])

        if i % 4 != 0:
            axes.set_yticklabels([])

        plt.xticks(np.arange(0, 4, 1))
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.plot(filter, color=colors['blue'])

    plt.show()

    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        figure.savefig(get_valid_save_path(save_path, 'Weights', 'pdf'), format='pdf', dpi=400)


def plot_activations(model, sample, save_path):
    figure = plt.figure()
    first_layer_output = model.layers[0].output
    activation_model = models.Model(inputs=model.input, outputs=first_layer_output)

    first_layer_activations = activation_model.predict(sample)

    for i in range(0, first_layer_activations.shape[2], 1):
        axes = figure.add_subplot(4, 4, i+1)
        activation = []

        for j in range(0, first_layer_activations.shape[1], 1):
            activation.append(first_layer_activations[0][j][i])

        axes.set_ylim([-1, 1])

        if i < 12:
            axes.set_xticklabels([])

        if i % 4 != 0:
            axes.set_yticklabels([])

        plt.xticks(np.arange(0, first_layer_activations.shape[1], first_layer_activations.shape[1] - 1))
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.plot(activation, color=colors['blue'])

    plt.show()

    if save_path is not None:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        figure.savefig(get_valid_save_path(save_path, 'Activations', 'pdf'), format='pdf', dpi=400)


def get_valid_save_path(save_path, filename, file_format):
    file_save_path = os.path.join(save_path, filename.replace(' ', '_') + '.' + file_format)
    file_index = 1

    while os.path.isfile(file_save_path):
        file_save_path = os.path.join(save_path, filename.replace(' ', '_') + '_' + str(file_index) + '.' + file_format)
        file_index = file_index + 1

    return file_save_path
