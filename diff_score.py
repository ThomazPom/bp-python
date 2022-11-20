import json

from PIL import ImageChops
from PIL import Image

from tools import cut_image, plot_figures
import numpy as np

from numpy import asarray

imgs, labels = cut_image("./numbergrid.png")

# imgs1, labels1 = [],[]
config = json.load(open("./config.json", "r"))

to_plot = []


def count_black(img):
    arraydiff = asarray(img)
    black_pixels = np.logical_and(0 == arraydiff[:, :, 0],
                                  np.logical_and(0 == arraydiff[:, :, 1], 0 == arraydiff[:, :, 2]))

    return np.sum(black_pixels)


def score_match(img1, img2, label="0"):
    diff = ImageChops.difference(Image.fromarray(img2),Image.fromarray(img1))

    num_black = count_black(diff)
    #diff.save(label+"-difference.png")
    return num_black


detection = []

import sys

args = sys.argv[1:]
if len(args) == 0:
    args = ["./numbergrid_v.png"]

imgs, labels = cut_image("./numbergrid.png")
imgs1, labels1 = cut_image(args[0])
for index_cible, img_cible in enumerate(imgs1):
    score = 0
    detected = ""
    for index,img_source in enumerate(imgs):
        label = next(iter([key for key,item in config.get("class_orders").items() if item==index]),"void" )
        newscore = score_match(img_source, img_cible, label + str(index_cible))
        if newscore > score:
            score = newscore
            detected = label
    detection.append(detected)
    to_plot.append({
        detected: img_cible
    })
print(json.dumps(detection))
#plot_figures(to_plot)
