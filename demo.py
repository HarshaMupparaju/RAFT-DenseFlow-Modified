import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import Warper


DEVICE = 'cuda'

def load_image(imfile):
    i = Image.open(imfile)
    imfile = i.resize((i.size[0]//2, i.size[1]//2))

    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()

    return img[None].to(DEVICE)


def viz(img, flo,path):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    import matplotlib.pyplot as plt
    plt.imshow(img_flo / 255.0)
    plt.show()
    cv2.imwrite(path+'/flow.jpg', img_flo[:, :, [2,1,0]])

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        


        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            i1 = cv2.imread(imfile1)
            i2 = cv2.imread(imfile2)

            i1 = cv2.resize(i1,(i1.shape[1]//2, i1.shape[0]//2))
            i2 = cv2.resize(i2,(i2.shape[1]//2, i2.shape[0]//2))

            cv2.imwrite(args.path+'/i1.jpg', i1)
            cv2.imwrite(args.path+'/i2.jpg', i2)

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low_12, flow_up_12 = model(image1, image2, iters=20, test_mode=True)
            flow_low_21, flow_up_21 = model(image2, image1, iters=20, test_mode=True)
            print(flow_up_12.shape)





            flow_up_12 = flow_up_12[0].permute(1,2,0).cpu().numpy()
            flow_up_21 = flow_up_21[0].permute(1,2,0).cpu().numpy()
            np.save(args.path+'/flow_up_12.npy', flow_up_12)

            warped_flow_21, mask_warped_flow_21 = Warper.demo1(flow_up_21, None, flow_up_12, None, False)
            mask = (np.abs(warped_flow_21+flow_up_12)[:,:,0] < 2).astype(np.uint8)


            mask = mask.reshape((mask.shape[0],mask.shape[1],1))
            cv2.imwrite(args.path+'/mask.jpg', mask*255)

            # viz(image1, flow_up_12,args.path)
            # flow_up = flow_up[0].permute(1,2,0).cpu().numpy()
            warped_frame1, mask1 = Warper.demo1(image2[0].permute(1,2,0).cpu().numpy(), None, flow_up_12, None, True)

            warper_frame1_final = warped_frame1 * mask + np.zeros(warped_frame1.shape) * (1 - mask)
            # print(warper_frame1_final)
            # cv2.imshow('image', cv2.cvtColor(warper_frame1_final.astype('uint8'), cv2.COLOR_RGB2BGR))
            # cv2.waitKey()
            # print(1/0)
            cv2.imwrite(args.path+'/warped_frame1.jpg', cv2.cvtColor(warped_frame1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(args.path+'/warper_frame1_final.jpg', cv2.cvtColor(warper_frame1_final.astype('uint8'), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
