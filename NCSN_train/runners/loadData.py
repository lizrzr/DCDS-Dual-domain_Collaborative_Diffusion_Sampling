import os
import numpy as np


def loadData(file_path, img_width, img_height, img_N, datatype, file_head):
    with open(file_path, 'rb') as f:
        img = np.fromfile(f, count=img_height * img_width * img_N, dtype=datatype)
        img = img.reshape(img_height, img_width)
    return img


if __name__ == "__main__":
    loadData('./data/20221101-140232_' + str(10).zfill(2) + '_geo_re/DetY.seq', 29, 1, 1, '<f4', 0)
