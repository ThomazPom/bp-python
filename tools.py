from keras_preprocessing.image import ImageDataGenerator

# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.preprocessing.image_dataset import load_image

def load_image(img_path):
    img = image.load_img(img_path)
    #img = img.resize(newsize)
    #img = img.convert("L")
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    #img_tensor.resize( 28, 28, 1)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


if __name__ == "__main__":
    # load model
    model = load_model("model.h5")

    # image path
    img_path = './src_imgs/0.png'  # dog
    # img_path = '/media/data/dogscats/test1/19.jpg'      # cat

    # load a single image
    new_image = load_image(img_path)
    print(new_image.shape)
    # check prediction
    pred = model.predict(new_image)
    print(pred)
def plot_figures(images_dicts, ncols=5):
    import matplotlib.pyplot as plt
    import numpy as np
    import PIL

    from PIL import Image
    # settings
    import math
    nrows = math.ceil(len(images_dicts) / ncols)
    figsize = [nrows * 6, ncols * 1.5]  # figure size, inches
    # prep (x,y) for extra plotting on selected sub-plots
    xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))  # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    if len(images_dicts) >= ncols:
        ax = ax.flatten()
    # plot simple raster image on each sub-plot
    for axi, img_dict in zip(ax, images_dicts):
        title, img = list(img_dict.items())[0]

        # print("Plotting", title)
        if type(img) is str:
            img = Image.open(img)
        if type(PIL.PngImagePlugin.PngImageFile) is PIL.PngImagePlugin.PngImageFile:
            img = np.array(img)
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        axi.imshow(img)
        axi.set_title(title)

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    # ax[0][2].plot(xs, 3 * ys, color='red', linewidth=3)
    # ax[4][3].plot(ys ** 2, xs, color='green', linewidth=3)

    plt.tight_layout()
    plt.show()


def plot_generator(gen, label="", nbplot=5, nbcols=5):
    plot_figures(
        [
            {
                label: gen.next()[0]
            }
            for i in range(0, nbplot)
        ], nbcols)


from PIL import Image
import numpy as np


def cut_image(img_path='numbergrid.gif', labels=None):
    # Load image
    if labels is None:
        labels = [
            "6", "4", "8", "1", "0", "5", "7", "3", "9", "void", "void", "void", "2", "void", "void", "void"
        ]
    nbcols = 4
    nbrows = 4
    bordersize = 3
    image = Image.open(img_path)

    image_arr = np.array(image)
    # print(image.size)
    # Crop image
    cropw = int((image.size[0]-(nbcols-1)*bordersize)/nbcols)
    croph = int((image.size[1]-(nbrows-1)*bordersize)/nbrows)

    imgs = []


    for icol in range(0, nbcols):
        for irow in range(0, nbrows):


            # print(icol,irow)
            # print(icol, irow, cropw, croph)
            # crop = image_arr[icol * cropw + bordersize:icol * cropw + cropw - bordersize,
            #        irow * croph + bordersize:irow * croph + croph - bordersize, :]

            crop = image_arr[

                   icol * (bordersize + cropw):
                   icol * (bordersize + cropw) + cropw,
                   irow * (bordersize + croph):
                   irow * (bordersize + croph) + croph


            , :]
            # print([icol * (bordersize + cropw),
            #        icol * (bordersize + cropw) + cropw,
            #        irow * (bordersize + croph),
            #        irow * (bordersize + croph) + croph
            #        ])
            imgs.append(crop)



    return imgs, labels


def create_dataset(number_each=100,greyscale=False,num_classes=11):
    datagen = ImageDataGenerator(
       # rescale=1./255,
        zoom_range=1. / 255,
        width_shift_range=1./255,

        rotation_range=1. / 255
    )
    img_data_array = []
    class_names = []
    class_ids = []
    class_map = {}
    imgs, labels = cut_image()

    ##########""
    lastimg_void = imgs[labels.index("void")]
    imgs = [img for index, img in enumerate(imgs) if labels[index] != "void"]
    imgs.append(lastimg_void)
    labels = [label for label in labels if label != "void"]
    labels.append("void")
    imgs=imgs[0:num_classes]
    labels=labels[0:num_classes]
    labelsidx = [i for i, v in enumerate(labels)]
    #############"""



    for image, label, idxClass in zip(imgs, labels, labelsidx):
        datagen.fit([image])

        iter = datagen.flow(np.array([image]))
        # plot_generator(iter)

        for i in range(0, number_each):
            image = iter.next()
            # image = image.astype('float32')
            # image /= 255
            imgappend = image[0]
            if greyscale:

                imgappend = np.dot(imgappend[..., :3], [0.2989, 0.5870, 0.1140])
            img_data_array.append(imgappend)
            class_ids.append(idxClass)
            class_names.append(label)
        class_map[label] = idxClass
    return np.array(img_data_array), np.array(class_ids), np.array(
        class_names), class_map  # extract the image array and class name
