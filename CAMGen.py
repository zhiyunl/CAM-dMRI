'''
@Author: your name
@Date: 2020-04-06 21:39:48
@LastEditTime: 2020-04-15 08:09:55
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /undefined/home/zhiyunl/MIA-Proj/pytorch-CAM-master/CAMGen.py
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw
from torch.nn import functional as F
from torchvision import transforms

import utils


# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx, img_size):
    # generate the class activation maps upsample to 256x256
    size_upsample = (img_size, img_size)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def get_cam(net, features_blobs, root, BBOX=True):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.485],
        std=[0.225]
    )
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((utils.IMG_SIZE, utils.IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    root_img = Image.open(root)
    img_tensor = preprocess(root_img)
    img_variable = img_tensor.unsqueeze(0).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], utils.classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()], img_size=utils.IMG_SIZE)
    width, height = root_img.size
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    print(CAM.shape)
    # CAM_pil = Image.fromarray(cv2.cvtColor(CAM, cv2.COLOR_BGR2RGB))
    CAM_pil = Image.fromarray(CAM)
    heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    heatmap_pil_box = heatmap_pil.copy()
    # bbox
    heatmap_pil_draw = Draw(heatmap_pil)
    if BBOX:
        threshold = CAM_pil.getextrema()[1] * 0.8
        # print(threshold)
        CAM_bin = CAM_pil.point(lambda p: p > threshold and 255)
        sli = CAM_bin.getbbox()
        heatmap_pil_draw = Draw(heatmap_pil_box)
        heatmap_pil_draw.rectangle(xy=sli, outline=(0, 0, 0), width=2)
        # heatmap_pil_box = Image.blend(heatmap_pil, box, 1)

    result = Image.blend(heatmap_pil, root_img.convert("RGB"), 0.5)
    result_box = Image.blend(heatmap_pil_box, root_img.convert("RGB"), 0.5)
    print(root_img.size, heatmap.size)

    fig, axs = plt.subplots(3, 2, constrained_layout=True)
    axs[0][0].imshow(root_img.convert("RGB"))
    axs[0][0].set_title('Source Image')
    axs[0][1].imshow(CAM_pil)
    axs[0][1].set_title('CAM mask')
    axs[1][0].imshow(heatmap_pil)
    axs[1][0].set_title('CAM heatmap')
    axs[1][1].imshow(result)
    axs[1][1].set_title('result')
    axs[2][0].imshow(CAM_bin)
    axs[2][0].set_title('CAM bin(thresh=0.2)')
    axs[2][1].imshow(result_box)
    axs[2][1].set_title('result bbox')
    # render the CAM and output
    print('output {} for the top1 prediction: {}'.format("cam_" + root, utils.classes[idx[0].item()]))

    if BBOX:
        utils.writer.add_figure(root + '_cam_bbox', fig, global_step=utils.EPOCH)
    else:
        utils.writer.add_figure(root + '_cam', fig, global_step=utils.EPOCH)
