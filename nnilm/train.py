import os
import yaml
import argparse
import json
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import plot_model
from keras.models import load_model
from nnilm.model import create_rectangular_model
from nnilm.utils import set_keras_backend, create_data_pipeline


def train(
        device_config,
        start_epoch=0,
        end_epoch=100,
        sample_period=10,
        samples_per_epoch=10000,
        num_seq_per_batch=64,
        verbose=True,
        keras_backend=None,
        save_path='./experiments/'):
    file_dir = os.path.dirname(__file__)
    conf_filepath = os.path.join(file_dir, 'device_config/{}.json'.format(device_config))
    checkpoint_path = os.path.join(save_path, 'checkpoints')

    if not os.path.isfile(conf_filepath):
        print('No config found for device "{}". Please see "nnilm/device_config/".'.format(device_config))
        exit(1)

    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    with open(conf_filepath) as config_file:
        conf = yaml.safe_load(config_file)

    if keras_backend is not None:
        set_keras_backend(keras_backend)

    if start_epoch > 0:
        model = load_model(os.path.join(checkpoint_path, '{}-{}epochs.hdf5'.format(device_config, start_epoch)))
    else:
        model = create_rectangular_model(conf['seq_length'])

    pipeline, real_input_std, real_target_std, real_avg_power = create_data_pipeline(conf, sample_period, num_seq_per_batch)
    validation_batch = pipeline.data_generator(fold='unseen_activations_of_seen_appliances').next()
    stats_file_name = 'stats-e{}-to-e{}.json'.format(start_epoch, end_epoch)
    stats = {
        'conf': conf,
        'real_input_std': float(real_input_std),
        'real_target_std': float(real_target_std),
        'real_avg_power': float(real_avg_power),
        'log': {},
    }

    with open(os.path.join(save_path, stats_file_name), 'w') as stats_file:
        json.dump(stats, stats_file, ensure_ascii=False, indent=4, sort_keys=True)

    def scheduler(epoch):
        learning_rate = K.eval(model.optimizer.lr)
        default_learning_rate = 0.001

        # decrease learning rate every 10 epochs by factor 10
        if epoch > 0 and epoch % 10 == 0:
            learning_rate = default_learning_rate * pow(10, -epoch / 10)

        return float(learning_rate)

    if verbose:
        print(model.summary())
        plot_model(model, to_file=os.path.join(save_path, 'model.png'), show_shapes=True, show_layer_names=False)

    if end_epoch > start_epoch:
        checkpoint_epoch = ModelCheckpoint(os.path.join(checkpoint_path, device_config + '-{epoch:01d}epochs.hdf5'), verbose=1, save_best_only=False)
        checkpoint_best = ModelCheckpoint(os.path.join(checkpoint_path, device_config + '-BEST.hdf5'), verbose=1, save_best_only=True)
        checkpoint_learning_rate = LearningRateScheduler(scheduler)

        log = model.fit_generator(
            pipeline.data_generator(),
            samples_per_epoch=samples_per_epoch,
            epochs=end_epoch,
            initial_epoch=start_epoch,
            callbacks=[checkpoint_epoch, checkpoint_best, checkpoint_learning_rate],
            validation_data=validation_batch,
            verbose=verbose
        )

        if verbose:
            plt.plot(log.history['loss'], label='Training loss')
            plt.plot(log.history['val_loss'], label='Validation loss')
            plt.legend()
            plt.grid()

        stats['log']['loss'] = log.history['loss']
        stats['log']['val_loss'] = log.history['val_loss']

        with open(os.path.join(save_path, stats_file_name), 'w') as stats_file:
            json.dump(stats, stats_file, ensure_ascii=False, indent=4, sort_keys=True)

        return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('device_config', help='The name of the device that should be trained, see "nnilm/device_config".', type=str)
    parser.add_argument('--start_epoch', '-s', help='The epoch from which training should be started.', type=int, default=0)
    parser.add_argument('--end_epoch', '-e', help='The epoch until the training should run.', type=int, default=100)

    args = parser.parse_args()

    train(args.device_config, args.start_epoch, args.end_epoch)
