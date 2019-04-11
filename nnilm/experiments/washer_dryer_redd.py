import os
from nnilm.train import train

epxeriment_save_path = os.path.join(os.path.dirname(__file__), '../../experiments/washer_dryer_redd/')

stats = train(
    'washer_dryer_redd',
    start_epoch=15,
    end_epoch=20,
    sample_period=10,
    samples_per_epoch=10000,
    num_seq_per_batch=64,
    verbose=True,
    keras_backend='plaidml.keras.backend',
    save_path=epxeriment_save_path
)

print(stats)

