# load model
import json
import os

from keras.models import load_model
import numpy as np

from tools import load_image, cut_image, plot_figures

model = load_model("model.h5")

ledir = "./src_imgs"

imgs, labels = cut_image("./numbergrid.gif")
imgs1, labels1 = cut_image("./numbergrid_v.gif")
# imgs1, labels1 = [],[]
predictions = model.predict(np.array(imgs + imgs1))
config = json.load(open("./config.json", "r"))

to_plot = []
for prediction, img in zip(predictions, imgs + imgs1):
    y_predict = np.argmax(prediction, axis=-1)
    to_plot.append({
        list(config.get("class_orders").keys())[y_predict]: img
    })

plot_figures(to_plot)
