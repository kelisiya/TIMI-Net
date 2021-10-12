import os
import cv2
import torch
import networks
from utils import *



if __name__ == '__main__':
    model = networks.get_generator(encoder='resnet_timi_encoder', decoder='resnet_timi_decoder')
    model.cuda()
    # load checkpoint
    checkpoint = torch.load('TIMI-Net.pth')
    model.load_state_dict(checkpoint, strict=True)
    # inference
    model = model.eval()
    # define the inputs and outputs folder
    image_root = './inputs/Image'
    trimap_rot = './inputs/Trimap'
    save_root = './outputs'
    img_list = os.listdir(image_root)
    for index, image_name in enumerate(img_list):
        # please make sure that the image and trimap share the same file name
        image_path = os.path.join(image_root, image_name)
        trimap_path = os.path.join(trimap_rot, os.path.splitext(image_name)[0]+".png")
        image = cv2.imread(image_path)
        trimap = cv2.imread(trimap_path,0)
        h,w,c = image.shape
        image_re, trimap_re = resize(image,trimap)
        image_dict = generator_tensor_dict(image_re, trimap_re)
        alpha_pre = single_inference(model,image_dict)
        alpha_final = cv2.resize(alpha_pre,(w,h))
        cv2.imwrite(os.path.join(save_root,image_name),alpha_final)
        print('Progress: ', index+1, '/', len(img_list))
        """
        one can find the evaluation code from GCA-matting if you are intending to test the metric for each pair of GT and result
        """






