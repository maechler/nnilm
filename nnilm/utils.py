import os
import keras.backend as K
from neuralnilm.data.loadactivations import load_nilmtk_activations
from neuralnilm.data.syntheticaggregatesource import SyntheticAggregateSource
from neuralnilm.data.realaggregatesource import RealAggregateSource
from neuralnilm.data.processing import DivideBy, IndependentlyCenter
from nnilm.rectangledatapipeline import RectangleDataPipeline
from nnilm.rectangulariser import start_and_end_and_mean


def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend


def create_data_pipeline(conf, sample_period, num_seq_per_batch, source_probabilities=(.5, .5), windows_key='windows'):
    appliances = conf['distracting_appliances']
    appliances.append(conf['target_appliance'])
    data_file_path = conf['data_file'] if os.path.isabs(conf['data_file']) else os.path.join(os.path.dirname(__file__) + '/../', conf['data_file'])
    windows = {}

    for window_name, window in conf[windows_key].iteritems():
        windows[window_name] = {}

        for house, window_selection in window.iteritems():
            windows[window_name][int(house)] = window_selection

    appliance_activations = load_nilmtk_activations(
        appliances=appliances,
        filename=data_file_path,
        sample_period=sample_period,
        windows=windows
    )

    synthetic_agg_source = SyntheticAggregateSource(
        activations=appliance_activations,
        target_appliance=conf['target_appliance'],
        seq_length=conf['seq_length'],
        sample_period=sample_period
    )

    real_agg_source = RealAggregateSource(
        activations=appliance_activations,
        target_appliance=conf['target_appliance'],
        seq_length=conf['seq_length'],
        filename=data_file_path,
        windows=windows,
        sample_period=sample_period
    )

    sample = real_agg_source.get_batch(num_seq_per_batch=1024).next()
    sample = sample.before_processing
    real_input_std = sample.input.flatten().std()
    real_target_std = sample.target.flatten().std()
    real_avg_power = sample.target.flatten().sum() / 1024 / conf['seq_length']

    pipeline = RectangleDataPipeline(
        [synthetic_agg_source, real_agg_source],
        num_seq_per_batch=num_seq_per_batch,
        source_probabilities=source_probabilities,
        input_processing=[DivideBy(conf['input_std']), IndependentlyCenter()],
        target_processing=[DivideBy(conf['target_std']), start_and_end_and_mean]
    )

    return pipeline, real_input_std, real_target_std, real_avg_power