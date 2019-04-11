import os
from nnilm.predict import predict
from nilmtk import Appliance

appliance = 'dish_washer_eco'
epxeriment_save_path = os.path.join(os.path.dirname(__file__), '../../experiments/' + appliance + '/')

# Avoid errors because fridge and freezer are considered same appliance
Appliance.allow_synonyms = False

stats = predict(
    appliance,
    save_path=epxeriment_save_path,
    plot_activations=True,
    plot_predictions=[0, 1, 2, 3, 4]
)

print(stats)

