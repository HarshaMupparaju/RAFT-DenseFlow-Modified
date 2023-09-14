
import argparse
import numpy as np
import skimage
from pathlib import Path
import torch 


def read_image(filepath):
    image = skimage.io.imread(filepath)
    return image

def resize_image(image,scaling_factor):
    image = skimage.transform.resize(image, (int(image.shape[0]*scaling_factor), int(image.shape[1]*scaling_factor)), anti_aliasing=True)
    return image


def get_flow(image1, image2):
    flow = np.load('/mnt/2tb-hdd/harshaM/DenseFlow/RAFT/DyNeRF/Test/01/140_150/flow_up_12.npy')
    return flow


def main():
    camera_id = 1
    image_id_1 = 140
    image_id_2 = 150
    images_dirpath = Path(f'DyNeRF/{camera_id:02}/images')
    image1_filepath = images_dirpath / f'{image_id_1:04}.jpg'
    image2_filepath = images_dirpath / f'{image_id_2:04}.jpg'
    image1 = read_image(image1_filepath)
    image2 = read_image(image2_filepath)

    image1 = resize_image(image1, 0.5)
    image2 = resize_image(image2, 0.5)

    flow_12 = get_flow(image1, image2)
    print(flow_12)
    print(1/0)
    






if __name__ == '__main__':
    main()



