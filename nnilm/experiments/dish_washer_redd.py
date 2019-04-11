import os
from nnilm.train import train

device_config = 'dish_washer_redd'
epxeriment_save_path = os.path.join(os.path.dirname(__file__), '../../experiments/' + device_config + '/')

stats = train(
    device_config,
    start_epoch=0,
    end_epoch=30,
    sample_period=10,
    samples_per_epoch=10000,
    num_seq_per_batch=64,
    verbose=True,
    keras_backend='plaidml.keras.backend',
    save_path=epxeriment_save_path
)

print(stats)

