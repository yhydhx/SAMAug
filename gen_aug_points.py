# %% set up environment
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from utils.SurfaceDice import compute_dice_coefficient
from vst_main.Testing import VST_test_once
import cv2
import matplotlib
matplotlib.use('Agg')


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


def swap_xy(points):
    new_points = np.zeros((len(points),2))
    new_points[:,0] = points[:,1]
    new_points[:,1] = points[:,0]
    return new_points

def swap_xy2(points):
    new_points = np.zeros((len(points),2))
    new_points[0, 0] = points[0, 0]
    new_points[0, 1] = points[0, 1]
    new_points[1:,0] = points[1:,1]
    new_points[1:,1] = points[1:,0]
    return new_points


"""Random Sample Point"""
def get_random_point(mask):
  indices = np.argwhere(mask==True)

  random_point = indices[np.random.choice(list(range(len(indices))))]
  random_point = [random_point[1], random_point[0]]
  return random_point
def get_multi_random_point(mask,points_nubmer):
    indices = np.argwhere(mask==True)

    random_point = indices[np.random.choice(list(range(len(indices))),points_nubmer,replace=False)]
    new_points = swap_xy(random_point)
    return new_points


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

def get_multi_entropy_points(input_point, mask, image, points_nubmer):
    new_points = np.zeros((points_nubmer + 1, 2))
    new_points[0] = [input_point[0], input_point[1]]
    for i in range(points_nubmer):
        new_points[i + 1] = get_next_entropy_point(new_points[:i + 1, :], mask, image)

    # new_points = swap_xy2(new_points)
    return new_points

def get_next_entropy_point(input_points, mask, image):
    max_entropy_point = [0, 0]
    max_entropy = 0
    grid_size = 9

    center_grids = [select_grid(image, input_point, grid_size) for input_point in input_points]

    indices = np.argwhere(mask == True)
    # for x, y in indices:
    for y, x in indices:
        grid = select_grid(image, [x, y], grid_size)
        entropy_diff = 0
        for center_grid in center_grids:
            entropy_diff += calculate_image_entroph(center_grid, grid)
        if entropy_diff > max_entropy:
            max_entropy_point = [x, y]
            max_entropy = entropy_diff
    return max_entropy_point


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

def get_multi_distance_points(input_point, mask, points_nubmer):
    new_points = np.zeros((points_nubmer + 1, 2))
    new_points[0] = [input_point[1], input_point[0]]
    for i in range(points_nubmer):
        new_points[i + 1] = get_next_distance_point(new_points[:i + 1, :], mask)

    new_points = swap_xy(new_points)
    return new_points

def get_next_distance_point(input_points, mask):
    max_distance_point = [0, 0]
    max_distance = 0
    input_points = np.array(input_points)

    indices = np.argwhere(mask == True)
    for x, y in indices:
        # print(x,y,input_points)
        distance = np.sum(np.sqrt((x - input_points[:, 0]) ** 2 + (y - input_points[:, 1]) ** 2))
        if max_distance < distance:
            max_distance_point = [x, y]
            max_distance = distance
    return max_distance_point


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

def get_multi_saliency_point(img, mask, input_points, points_nubmer, img_name, save_img_path):
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

    new_points = np.zeros((points_nubmer + 1, 2))
    new_points[0] = [input_points[1], input_points[0]]

    # judge point in the vst mask
    vst_indices = np.argwhere(vst_mask > 0)
    random_index = np.random.choice(len(vst_indices), points_nubmer)

    vst_roi_random_point = []
    for i, item in enumerate(random_index):
        new_points[i + 1] = [vst_indices[item][1], vst_indices[item][0]]
        vst_roi_random_point.append([vst_indices[item][1], vst_indices[item][0]])

    # vst_random_point = [vst_indices[random_index][1], vst_indices[random_index][0]]
    # vst_roi_random_point = [vst_indices[random_index][1], vst_indices[random_index][0]]

    plt.imshow(vst_input_img)
    plt.axis('off')
    show_mask(np.array(vst_mask > 0).astype(int), plt.gca())
    show_points(np.array(vst_roi_random_point), np.array([1 for i in range(points_nubmer)]), plt.gca())
    plt.savefig(osp.join(save_img_path,
                         "{}_5_vst_mask_point.jpg".format(img_name.split('.')[0])), bbox_inches='tight', dpi=100,
                pad_inches=0)
    plt.clf()

    # vst_random_point = [vst_roi_random_point[0] + xmin - 10, vst_roi_random_point[1] + ymin - 10]
    vst_random_point = [[pts[0] + xmin - 10, pts[1] + ymin - 10] for pts in vst_roi_random_point]

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


