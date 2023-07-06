# %% set up environment
import os

import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from utils.SurfaceDice import compute_dice_coefficient
from vst_main.Testing import VST_test_once
import cv2
import matplotlib


def show(save_root, class_name, img_id, ann_id):
    img_save_root = osp.join(save_root, class_name, "{:0>12d}".format(img_id))

    result_list = []
    for name in os.listdir(img_save_root):
        if "ann{}".format(ann_id) in name:
            if ".jpg" in name:
                result_list.append(name)

    result_list.sort()

    fig = plt.figure()
    fig.suptitle(img_save_root)
    im = cv2.imread(osp.join(img_save_root, result_list[0]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(241)
    ax1.set_title(result_list[0])
    ax1.imshow(im)

    im = cv2.imread(osp.join(img_save_root, result_list[1]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(242)
    ax1.set_title(result_list[1])
    ax1.imshow(im)

    im = cv2.imread(osp.join(img_save_root, result_list[2]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(243)
    ax1.set_title(result_list[2])
    ax1.imshow(im)

    im = cv2.imread(osp.join(img_save_root, result_list[3]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(245)
    ax1.set_title(result_list[3])
    ax1.imshow(im)

    im = cv2.imread(osp.join(img_save_root, result_list[4]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(246)
    ax1.set_title(result_list[4])
    ax1.imshow(im)

    im = cv2.imread(osp.join(img_save_root, result_list[5]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(247)
    ax1.set_title(result_list[5])
    ax1.imshow(im)

    im = cv2.imread(osp.join(img_save_root, result_list[6]))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax1 = fig.add_subplot(248)
    ax1.set_title(result_list[6])
    ax1.imshow(im)

    plt.show()

    # cv2.waitKey(0)
    # pass




if __name__ == '__main__':
    # save_root = r"/data3/machong/project_results/SAM_AUG/coco2017/train2017/2023-06-24_00-40-39"

    """3 points root path"""
    # save_root = r"/data3/machong/project_results/SAM_AUG/coco2017/train2017/2023-06-23_17-14-34"

    """5 points root path"""
    save_root = r"/data3/machong/project_results/SAM_AUG/coco2017/train2017/2023-06-23_09-34-03"

    class_name = "toothbrush"
    img_id = 404943
    ann_id = 0
    show(save_root, class_name, img_id, ann_id)
