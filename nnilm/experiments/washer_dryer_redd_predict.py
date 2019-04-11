import os
from nnilm.predict import predict

epxeriment_save_path = os.path.join(os.path.dirname(__file__), '../../experiments/washer_dryer_redd/')


stats = predict(
    'washer_dryer_redd',
    save_path=epxeriment_save_path,
    plot_activations=True,
    plot_predictions=[0, 1, 2, 3, 4]
)

print(stats)

