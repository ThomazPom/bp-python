# load model
import json
from keras.models import load_model
import numpy as np

from tools import cut_image, plot_figures

model = load_model("model.h5")

import sys

args = sys.argv[1:]
if len(args)==0:
    args = ["./numbergrid_v.png"]

imgs,labels = cut_image(args[0])

predictions = model.predict(np.array(imgs))
config = json.load(open("./config.json", "r"))

classes_names = list(config.get("class_orders").keys())

print(
    json.dumps([classes_names[np.argmax(prediction, axis=-1)] for prediction in predictions])
)
