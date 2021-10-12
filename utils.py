import cv2
import math
import torch
import numpy as np



def resize(image,trimap):
    try:
        w,h,c = trimap.shape
        if (c == 3):
            trimap = cv2.cvtColor(trimap, cv2.COLOR_BGR2GRAY)
    except ValueError:
        trimap = trimap

    max_size = 1600 * 1280    # in case that some images will cause GPU overload

    h, w, _ = image.shape
    if h * w > max_size:
        s = w / h
        new_h = int(math.sqrt(max_size / s))
        new_w = int(s * new_h)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return image,trimap

def generator_tensor_dict(image, trimap):
    # read images
    sample = {'image': image, 'trimap': trimap, 'alpha_shape': trimap.shape}

    # reshape
    h, w = sample["alpha_shape"]
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
        sample['image'] = padded_image
        sample['trimap'] = padded_trimap

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # convert GBR images to RGB
    image, trimap = sample['image'][:,:,::-1], sample['trimap']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)

    trimap[trimap < 85] = 0
    trimap[trimap >= 170] = 2
    trimap[trimap >= 85] = 1
    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
    sample['image'] = sample['image'].sub_(mean).div_(std)
    sample['trimap'] = sample['trimap'][None, ...].float()

    # add first channel
    sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

    return sample


def single_inference(model, image_dict):

    with torch.no_grad():
        image, trimap = image_dict['image'], image_dict['trimap']
        alpha_shape = image_dict['alpha_shape']
        image = image.cuda()
        trimap = trimap.cuda()
        alpha_pred  = model(image, trimap)

        torch.cuda.empty_cache()

        alpha_pred[trimap == 2] = 1
        alpha_pred[trimap == 0] = 0

        h, w = alpha_shape
        test_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
        test_pred = test_pred.astype(np.uint8)
        test_pred = test_pred[32:h+32, 32:w+32]

        return test_pred
