from PIL import Image
import numpy as np


def get_images():
    img = Image.open('./data/training/Acer_Mono/Acer_Mono_01.ab.jpg')

    np_img = np.array(img)

    print(np_img.shape)

get_images()