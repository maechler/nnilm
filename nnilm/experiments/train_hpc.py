import os
import argparse
from nnilm.train import train
from nilmtk import Appliance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('device_config', help='The name of the device that should be trained, see "nnilm/device_config".', type=str)
    parser.add_argument('--gpu', '-g', help='ID of GPU that should be used.', type=str, default="")

    args = parser.parse_args()

    epxeriment_save_path = os.path.join(os.path.dirname(__file__), '../../experiments/' + args.device_config + '/')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print('Using GPU: ' + args.gpu)

    # Avoid errors because fridge and freezer are considered same appliance
    Appliance.allow_synonyms = False

    stats = train(
        args.device_config,
        start_epoch=0,
        end_epoch=30,
        sample_period=10,
        samples_per_epoch=10000,
        num_seq_per_batch=64,
        verbose=True,
        keras_backend=None,
        save_path=epxeriment_save_path
    )

    print(stats)
