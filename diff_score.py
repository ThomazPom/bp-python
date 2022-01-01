import json

from PIL import ImageChops
from PIL import Image

from tools import cut_image, plot_figures
import numpy as np

from numpy import asarray

imgs, labels = cut_image("./numbergrid.gif")

# imgs1, labels1 = [],[]
config = json.load(open("./config.json", "r"))

to_plot = []


def count_black(img):
    arraydiff = asarray(img)
    black_pixels = np.logical_and(0 == arraydiff[:, :, 0],
                                  np.logical_and(0 == arraydiff[:, :, 1], 0 == arraydiff[:, :, 2]))

    return np.sum(black_pixels)


def score_match(img1, img2, label="0"):
    diff = ImageChops.difference(Image.fromarray(img1), Image.fromarray(img2))

    num_black = count_black(diff)
    # diff.save(label+"-difference.png")
    return num_black


detection = []

import sys

args = sys.argv[1:]
if len(args) == 0:
    args = ["./numbergrid_v.gif"]

imgs, labels = cut_image("./numbergrid.gif")
imgs1, labels1 = cut_image(args[0])
for index_cible, img_cible in enumerate(imgs1):
    score = 0
    detected = ""
    for img_source, label in zip(imgs, config.get("class_orders").keys()):
        newscore = score_match(img_source, img_cible, label + str(index_cible))
        if newscore > score:
            score = newscore
            detected = label
    if detected == "2":
        score_void = score_match(img_cible, imgs[[*config.get("class_orders").keys()].index("void")])
        detected = "2" if score_void < img_cible.shape[0] * img_cible.shape[0] * 0.96 else "void"

    detection.append(detected)
    to_plot.append({
        detected: img_cible
    })
print(json.dumps(detection))
plot_figures(to_plot)
