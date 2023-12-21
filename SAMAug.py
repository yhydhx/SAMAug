# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from tqdm import tqdm
import logging
from segment_anything import sam_model_registry, SamPredictor
import argparse
import datetime
from utils.SurfaceDice import compute_dice_coefficient
from PIL import Image
from vst_main.Testing import VST_test_once
import cv2
import matplotlib
import json
matplotlib.use('Agg')
os.environ['CUDA_VISIBLE_DEVICES'] = '9'


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=150):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=0.75)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=0.75)


"""Random Sample Point"""
def get_random_point(mask):
  indices = np.argwhere(mask==True)

  random_point = indices[np.random.choice(list(range(len(indices))))]
  random_point = [random_point[1], random_point[0]]
  return random_point


"""Max Entropy Point"""
def image_entropy(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist /= hist.sum()
    # Calculate the entropy
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    return entropy

def calculate_image_entroph(img1, img2):
    # Calculate the entropy for each image
    entropy1 = image_entropy(img1)
    # print(img2)
    try:
        entropy2 = image_entropy(img2)
    except:
        entropy2 = 0
    # Compute the entropy between the two images
    entropy_diff = abs(entropy1 - entropy2)
    # print("Entropy Difference:", entropy_diff)
    return entropy_diff

def select_grid(image, center_point, grid_size):
    (img_h, img_w, _) = image.shape

    # Extract the coordinates of the center point
    x, y = center_point
    x = int(np.floor(x))
    y = int(np.floor(y))
    # Calculate the top-left corner coordinates of the grid
    top_left_x = x - (grid_size // 2) if x - (grid_size // 2) > 0 else 0
    top_left_y = y - (grid_size // 2) if y - (grid_size // 2) > 0 else 0
    bottom_right_x = top_left_x + grid_size if top_left_x + grid_size < img_w else img_w
    bottom_right_y = top_left_y + grid_size if top_left_y + grid_size < img_h else img_h

    # Extract the grid from the image
    grid = image[top_left_y: bottom_right_y, top_left_x: bottom_right_x]

    return grid

def get_entropy_points(input_point,mask,image):
    max_entropy_point = [0,0]
    max_entropy = 0
    grid_size = 9
    center_grid = select_grid(image, input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        grid = select_grid(image, [x,y], grid_size)
        entropy_diff = calculate_image_entroph(center_grid, grid)
        if entropy_diff > max_entropy:
            max_entropy_point = [x,y]
            max_entropy = entropy_diff
    return [max_entropy_point[1], max_entropy_point[0]]


"""Max Distance Point"""
def get_distance_points(input_point, mask):
    max_distance_point = [0,0]
    max_distance = 0
    # grid_size = 9
    # center_grid = select_grid(image,input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        distance = np.sqrt((x- input_point[0])**2 + (y- input_point[1]) ** 2)
        if max_distance < distance:
            max_distance_point = [x,y]
            max_distance = distance
    return [max_distance_point[1],max_distance_point[0]]


"""Saliency Point"""
def get_saliency_point(img, mask, img_name, save_img_path):
    (img_h, img_w, _) = img.shape

    coor = np.argwhere(mask > 0)
    ymin = min(coor[:, 0])
    ymax = max(coor[:, 0])
    xmin = min(coor[:, 1])
    xmax = max(coor[:, 1])

    xmin2 = xmin - 10 if xmin - 10 > 0 else 0
    xmax2 = img_w if xmax + 10 > img_w else xmax + 10
    ymin2 = ymin - 10 if ymin - 10 > 0 else 0
    ymax2 = img_h if ymax + 10 > img_h else ymax + 10

    vst_input_img = img[ymin2:ymax2, xmin2:xmax2, :]

    # VST mask
    vst_mask = VST_test_once(img_path=vst_input_img)

    # judge point in the vst mask
    vst_indices = np.argwhere(vst_mask > 0)
    random_index = np.random.choice(len(vst_indices), 1)[0]
    # vst_random_point = [vst_indices[random_index][1], vst_indices[random_index][0]]
    vst_roi_random_point = [vst_indices[random_index][1], vst_indices[random_index][0]]

    plt.imshow(vst_input_img)
    plt.axis('off')
    show_mask(np.array(vst_mask > 0).astype(int), plt.gca())
    show_points(np.array([vst_roi_random_point]), np.array([1]), plt.gca())
    plt.savefig(osp.join(save_img_path,
                         "{}_5_vst_mask_point.jpg".format(img_name.split('.')[0])), bbox_inches='tight', dpi=100,
                pad_inches=0)
    plt.clf()

    vst_random_point = [vst_roi_random_point[0] + xmin - 10, vst_roi_random_point[1] + ymin - 10]

    return vst_random_point


# SAM inference and compute dice
def gen_SAM_mask_and_dice(sam_predictor, input_point, input_label, gt_mask):
    masks, _, _ = sam_predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    mask = masks[0].astype(int)
    dice = round(compute_dice_coefficient(gt_mask > 0, mask > 0), 4)
    return mask, dice



def main_first_sam_saliency_all_positive(args):
    # define logging
    os.makedirs(args.save_root_path, exist_ok=True)

    # Clear former logger
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)

    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("{}/log.txt".format(args.save_root_path), mode='a', encoding='UTF-8'),
            logging.StreamHandler()
        ]
    )

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    src_root_path = osp.join(args.img_root_path, "src")
    mask_root_path = osp.join(args.img_root_path, "mask")

    run_img_names = os.listdir(src_root_path)
    logging.info(f"Medical Dataset {args.dataset_name}")
    logging.info(f"Set {len(run_img_names)} Images:")

    cate_dice = []
    test_count = 0
    max_diff_case_dice, max_diff_case_ann_index = [], []

    img_max_diff_save_dict = {}
    for img_name in tqdm(run_img_names):

        # read and set image
        im = np.asarray(Image.open(osp.join(src_root_path, img_name)).convert('RGB'))
        predictor.set_image(im)
        (img_h, img_w, _) = im.shape

        query_mask = np.load(osp.join(mask_root_path, img_name.replace('.png', '.npy')))
        query_mask[query_mask > 0] = 1
        save_init_point = []

        save_img_path = args.save_root_path

        """1 -> Set first point"""
        # first_point random selected from gt mask
        indices = np.argwhere(query_mask > 0)
        random_index = np.random.choice(len(indices), 1)[0]
        first_point = [indices[random_index][1], indices[random_index][0]]

        input_point = np.array([first_point])
        input_label = np.array([1])

        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        # SAM first result
        mask = masks[0].astype(int)
        first_dice = round(compute_dice_coefficient(query_mask > 0, mask > 0), 4)

        # save gt mask with init point
        plt.imshow(im)
        plt.axis('off')
        show_mask(query_mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_0_GT_1point.jpg".format(img_name.split('.')[0])), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()

        save_init_point.append({"pos": first_point, "label": 1})

        # first dice result
        dice_list = []
        dice_list.append(first_dice)
        # save first sam result
        plt.imshow(im)
        plt.axis('off')
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_1_SAM_1point_d{}.jpg".format(img_name.split('.')[0], dice_list[0])), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        np.save(osp.join(save_img_path, "{}_1_init_SAM_mask.npy".format(img_name.split('.')[0])), mask.astype(np.uint8))

        """2 -> Generate random point"""
        random_point = get_random_point(mask)
        input_point = np.array([first_point, random_point])
        input_label = np.array([1, 1])
        mask2, dice2 = gen_SAM_mask_and_dice(predictor, input_point, input_label, gt_mask=query_mask)

        plt.imshow(im)
        plt.axis('off')
        show_mask(mask2, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_2_random_2point_d{}.jpg".format(img_name.split('.')[0], dice2)), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        save_init_point.append({"pos": random_point, "label": 1})
        np.save(osp.join(save_img_path, "{}_2_random_point_mask.npy".format(img_name.split('.')[0])), mask2.astype(np.uint8))
        dice_list.append(dice2)

        """3 -> Generate Max entropy point"""
        entropy_point = get_entropy_points(first_point, mask, im)
        input_point = np.array([first_point, entropy_point])
        input_label = np.array([1, 1])
        mask3, dice3 = gen_SAM_mask_and_dice(predictor, input_point, input_label, gt_mask=query_mask)

        plt.imshow(im)
        plt.axis('off')
        show_mask(mask3, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_3_entorpy_2point_d{}.jpg".format(img_name.split('.')[0], dice3)), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        save_init_point.append({"pos": entropy_point, "label": 1})
        np.save(osp.join(save_img_path, "{}_3_entorpy_point_mask.npy".format(img_name.split('.')[0])), mask3.astype(np.uint8))
        dice_list.append(dice3)

        """4 -> Generate Max dis point"""
        max_dis_point = get_distance_points(first_point, mask)
        input_point = np.array([first_point, max_dis_point])
        input_label = np.array([1, 1])
        mask4, dice4 = gen_SAM_mask_and_dice(predictor, input_point, input_label, gt_mask=query_mask)

        plt.imshow(im)
        plt.axis('off')
        show_mask(mask4, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_4_maxdis_2point_d{}.jpg".format(img_name.split('.')[0], dice4)), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        save_init_point.append({"pos": max_dis_point, "label": 1})
        np.save(osp.join(save_img_path, "{}_4_maxdis_point_mask.npy".format(img_name.split('.')[0])), mask4.astype(np.uint8))
        dice_list.append(dice4)

        """5 -> Generate Saliency point"""
        vst_random_point = get_saliency_point(im, mask, img_name, save_img_path)

        input_point = np.array([first_point, vst_random_point])
        input_label = np.array([1, 1])
        mask5, dice5 = gen_SAM_mask_and_dice(predictor, input_point, input_label, gt_mask=query_mask)

        plt.imshow(im)
        plt.axis('off')
        show_mask(mask5, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path,
                             "{}_5_saliency_2point_d{}.jpg".format(img_name.split('.')[0], dice5)), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        save_init_point.append({"pos": vst_random_point, "label": 1})
        np.save(osp.join(save_img_path, "{}_5_saliency_point_mask.npy".format(img_name.split('.')[0])), mask5.astype(np.uint8))
        dice_list.append(dice5)

        for item in save_init_point:
            item['pos'][0] =int(item['pos'][0])
            item['pos'][1] =int(item['pos'][1])

        save_dict = {"points": save_init_point.copy(), "dice_socres": dice_list.copy()}
        json.dump(save_dict, open(osp.join(save_img_path, "{}_info.json".format(img_name.split('.')[0])), "w"))

        cate_dice.append(dice_list)

        mean_diff = (dice_list[4]-dice_list[0] + dice_list[3]-dice_list[0] + dice_list[2]-dice_list[0] + dice_list[1]-dice_list[0]) / 4
        max_diff_case_dice.append(mean_diff)
        max_diff_case_ann_index.append(img_name)

    all_dice_sort = np.argsort(-np.array(max_diff_case_dice))
    for i in all_dice_sort:
        img_max_diff_save_dict.update({str(max_diff_case_ann_index[i]): max_diff_case_dice[i]})

    json.dump(img_max_diff_save_dict, open(osp.join(args.save_root_path, "cat_diff.json"), 'w'))

    logging.info(f"{img_max_diff_save_dict}\n")

    for opt_key in vars(args).keys():
        logging.info("{}: {}".format(opt_key, vars(args)[opt_key]))

    logging.info(f"Medical Image Data: {args.dataset_name}")
    logging.info(f"init sam dice: {np.mean(np.array(cate_dice)[:, 0])}")
    logging.info(f"random sam dice: {np.mean(np.array(cate_dice)[:, 1])}")
    logging.info(f"max entropy sam dice: {np.mean(np.array(cate_dice)[:, 2])}")
    logging.info(f"max dis dice: {np.mean(np.array(cate_dice)[:, 3])}")
    logging.info(f"saliency dice: {np.mean(np.array(cate_dice)[:, 4])}")


def main_best_first_sam_saliency_all_positive(args):
    # define logging
    os.makedirs(args.save_root_path, exist_ok=True)

    # Clear former logger
    root_logger = logging.getLogger()
    for h in root_logger.handlers:
        root_logger.removeHandler(h)

    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("{}/log.txt".format(args.save_root_path), mode='a', encoding='UTF-8'),
            logging.StreamHandler()
        ]
    )

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    src_root_path = osp.join(args.img_root_path, "src")
    mask_root_path = osp.join(args.img_root_path, "mask")

    run_img_names = os.listdir(src_root_path)
    logging.info(f"Medical Dataset {args.dataset_name}")
    logging.info(f"Set {len(run_img_names)} Images:")

    cate_dice = []
    test_count = 0
    max_diff_case_dice, max_diff_case_ann_index = [], []

    img_max_diff_save_dict = {}
    for img_name in tqdm(run_img_names):

        # read and set image
        im = np.asarray(Image.open(osp.join(src_root_path, img_name)).convert('RGB'))
        predictor.set_image(im)
        (img_h, img_w, _) = im.shape

        query_mask = np.load(osp.join(mask_root_path, img_name.replace('.png', '.npy')))
        query_mask[query_mask > 0] = 1
        save_init_point = []

        save_img_path = args.save_root_path

        """1 -> Set first point"""
        temp_mask_list, temp_point_list, temp_plabel_list, temp_dice_list = [], [], [], []
        for i in range(3):
            # first_point random selected from gt mask
            indices = np.argwhere(query_mask > 0)
            random_index = np.random.choice(len(indices), 1)[0]
            first_point = [indices[random_index][1], indices[random_index][0]]

            input_point = np.array([first_point])
            input_label = np.array([1])

            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            # SAM first result
            mask = masks[np.argmax(scores)].astype(int)
            first_dice = round(compute_dice_coefficient(query_mask > 0, mask > 0), 4)

            temp_mask_list.append(mask.copy())
            temp_point_list.append(first_point.copy())
            temp_dice_list.append(first_dice)

            # Select a random point that demonstrates a higher dice score among the three loops.
            if first_dice > 0.3:
                break

        # save max first
        first_max_index = np.argmax(np.array(temp_dice_list))
        first_point = temp_point_list[first_max_index]
        input_point = np.array([first_point])
        input_label = np.array([1])
        first_dice = temp_dice_list[first_max_index]
        mask = temp_mask_list[first_max_index]

        if first_dice < 0.1:
            print(img_name, "Dice Score is too low", first_dice)
            continue
        # save gt mask with init point
        plt.imshow(im)
        plt.axis('off')
        show_mask(query_mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_0_GT_1point.jpg".format(img_name.split('.')[0])), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()

        save_init_point.append({"pos": first_point, "label": 1})

        # first dice result
        dice_list = []
        dice_list.append(first_dice)
        # save first sam result
        plt.imshow(im)
        plt.axis('off')
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_1_SAM_1point_d{}.jpg".format(img_name.split('.')[0], dice_list[0])), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        np.save(osp.join(save_img_path, "{}_1_init_SAM_mask.npy".format(img_name.split('.')[0])), mask.astype(np.uint8))

        """2 -> Generate random point"""
        random_point = get_random_point(mask)
        input_point = np.array([first_point, random_point])
        input_label = np.array([1, 1])
        mask2, dice2 = gen_SAM_mask_and_dice(predictor, input_point, input_label, gt_mask=query_mask)

        plt.imshow(im)
        plt.axis('off')
        show_mask(mask2, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_2_random_2point_d{}.jpg".format(img_name.split('.')[0], dice2)), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        save_init_point.append({"pos": random_point, "label": 1})
        np.save(osp.join(save_img_path, "{}_2_random_point_mask.npy".format(img_name.split('.')[0])), mask2.astype(np.uint8))
        dice_list.append(dice2)

        """3 -> Generate Max entropy point"""
        entropy_point = get_entropy_points(first_point, mask, im)
        input_point = np.array([first_point, entropy_point])
        input_label = np.array([1, 1])
        mask3, dice3 = gen_SAM_mask_and_dice(predictor, input_point, input_label, gt_mask=query_mask)

        plt.imshow(im)
        plt.axis('off')
        show_mask(mask3, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_3_entorpy_2point_d{}.jpg".format(img_name.split('.')[0], dice3)), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        save_init_point.append({"pos": entropy_point, "label": 1})
        np.save(osp.join(save_img_path, "{}_3_entorpy_point_mask.npy".format(img_name.split('.')[0])), mask3.astype(np.uint8))
        dice_list.append(dice3)

        """4 -> Generate Max dis point"""
        max_dis_point = get_distance_points(first_point, mask)
        input_point = np.array([first_point, max_dis_point])
        input_label = np.array([1, 1])
        mask4, dice4 = gen_SAM_mask_and_dice(predictor, input_point, input_label, gt_mask=query_mask)

        plt.imshow(im)
        plt.axis('off')
        show_mask(mask4, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path, "{}_4_maxdis_2point_d{}.jpg".format(img_name.split('.')[0], dice4)), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        save_init_point.append({"pos": max_dis_point, "label": 1})
        np.save(osp.join(save_img_path, "{}_4_maxdis_point_mask.npy".format(img_name.split('.')[0])), mask4.astype(np.uint8))
        dice_list.append(dice4)

        """5 -> Generate Saliency point"""
        vst_random_point = get_saliency_point(im, mask, img_name, save_img_path)

        input_point = np.array([first_point, vst_random_point])
        input_label = np.array([1, 1])
        mask5, dice5 = gen_SAM_mask_and_dice(predictor, input_point, input_label, gt_mask=query_mask)

        plt.imshow(im)
        plt.axis('off')
        show_mask(mask5, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.savefig(osp.join(save_img_path,
                             "{}_5_saliency_2point_d{}.jpg".format(img_name.split('.')[0], dice5)), bbox_inches='tight', dpi=100, pad_inches=0)
        plt.clf()
        save_init_point.append({"pos": vst_random_point, "label": 1})
        np.save(osp.join(save_img_path, "{}_5_saliency_point_mask.npy".format(img_name.split('.')[0])), mask5.astype(np.uint8))
        dice_list.append(dice5)

        for item in save_init_point:
            item['pos'][0] =int(item['pos'][0])
            item['pos'][1] =int(item['pos'][1])

        save_dict = {"points": save_init_point.copy(), "dice_socres": dice_list.copy()}
        json.dump(save_dict, open(osp.join(save_img_path, "{}_info.json".format(img_name.split('.')[0])), "w"))

        cate_dice.append(dice_list)

        mean_diff = (dice_list[4]-dice_list[0] + dice_list[3]-dice_list[0] + dice_list[2]-dice_list[0] + dice_list[1]-dice_list[0]) / 4
        max_diff_case_dice.append(mean_diff)
        max_diff_case_ann_index.append(img_name)

    all_dice_sort = np.argsort(-np.array(max_diff_case_dice))
    for i in all_dice_sort:
        img_max_diff_save_dict.update({str(max_diff_case_ann_index[i]): max_diff_case_dice[i]})

    json.dump(img_max_diff_save_dict, open(osp.join(args.save_root_path, "cat_diff.json"), 'w'))

    logging.info(f"{img_max_diff_save_dict}\n")

    for opt_key in vars(args).keys():
        logging.info("{}: {}".format(opt_key, vars(args)[opt_key]))

    logging.info(f"Medical Image Data: {args.dataset_name}")
    logging.info(f"init sam dice: {np.mean(np.array(cate_dice)[:, 0])}")
    logging.info(f"random sam dice: {np.mean(np.array(cate_dice)[:, 1])}")
    logging.info(f"max entropy sam dice: {np.mean(np.array(cate_dice)[:, 2])}")
    logging.info(f"max dis dice: {np.mean(np.array(cate_dice)[:, 3])}")
    logging.info(f"saliency dice: {np.mean(np.array(cate_dice)[:, 4])}")


# Test area
def area():
    npy_path = r"/data3/machong/project_results/SAM_AUG/coco2017/train2017/2023-06-10_10-21-24/person/000000353056/ann10_c1_0_gt_mask.npy"
    gt_mask = np.load(npy_path)
    area = gt_mask.sum()


data_name = "01-MR-T1_Brain_Ventricle"
# data_name = "17-CT_Lungs"
# data_name = "20-CT_Stomach"

# SAM setup
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='vit_h')
parser.add_argument('--checkpoint', type=str, default='/data3/machong/LM_tools/MedSAM-main/work_dir/SAM/sam_vit_h_4b8939.pth')
# sam_vit_h_4b8939.pth sam_vit_l_0b3195.pth sam_vit_b_01ec64.pth
parser.add_argument('--device', type=str, default='cuda:0')

# data path
parser.add_argument('--dataset_name', type=str, default=data_name)
parser.add_argument('--img_root_path', type=str, default="/data3/machong/datasets/SAM_Medical_Test_3D_Imgs/{}/".format(data_name))
parser.add_argument('--save_root_path', type=str, default="/data3/machong/project_results/SAM_AUG/{}/".format(data_name))

# para
# parser.add_argument('--img_num_each_cat', type=int, default=20)
# parser.add_argument('--img_set_num_each_cat', type=int, default=200)
# parser.add_argument('--ann_area_thre', type=int, default=4000)

args = parser.parse_args()


if __name__ == '__main__':
    args.save_root_path = osp.join(args.save_root_path, "{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    main_best_first_sam_saliency_all_positive(args)
