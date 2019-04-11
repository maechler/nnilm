import os
from nnilm.predict import predict

epxeriment_save_path = os.path.join(os.path.dirname(__file__), '../../experiments/dish_washer_redd/')


stats = predict(
    'dish_washer_redd',
    save_path=epxeriment_save_path,
    plot_activations=False,
    plot_layer_weights=[0,1],
    plot_predictions=[]
)

print(stats)

