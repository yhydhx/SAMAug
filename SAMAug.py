import torch
import torchvision

import numpy as np
import matplotlib
import cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO
import sys  

from sklearn.cluster import KMeans

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 144/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
import os
def get_random_point(mask):
  indices = np.argwhere(mask ==True)

  random_point = indices[np.random.choice(list(range(len(indices))))]
  random_point = [random_point[1], random_point[0]]
  return random_point


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

def calculate_image_entroph(img1,img2):
    # Calculate the entropy for each image
    entropy1 = image_entropy(img1)
    #print(img2)
    try:
        entropy2 = image_entropy(img2)
    except:
        entropy2 = 0
    # Compute the entropy between the two images
    entropy_diff = abs(entropy1 - entropy2)
    #print("Entropy Difference:", entropy_diff)
    return entropy_diff

def select_grid(image, center_point, grid_size):
    # Extract the coordinates of the center point
    x, y = center_point
    x = int(np.floor(x))
    y = int(np.floor(y))
    # Calculate the top-left corner coordinates of the grid
    top_left_x = x - (grid_size // 2)
    top_left_y = y - (grid_size // 2)
    
    # Extract the grid from the image
    grid = image[top_left_y : top_left_y + grid_size, top_left_x : top_left_x + grid_size]

    return grid

def get_entropy_points(input_point,mask,image):
    max_entropy_point = [0,0]
    max_entropy = 0
    grid_size = 9
    center_grid = select_grid(image,input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        grid = select_grid(image, [x,y], grid_size)
        entropy_diff = calculate_image_entroph(center_grid, grid)
        if entropy_diff > max_entropy:
            max_entropy_point = [x,y]
            max_entropy = entropy_diff
    return [max_entropy_point[1], max_entropy_point[0]]


def get_distance_points(input_point,mask,image):
    max_distance_point = [0,0]
    max_distance = 0
    grid_size = 9
    center_grid = select_grid(image,input_point, grid_size)

    indices = np.argwhere(mask ==True)
    for x,y in indices:
        distance = np.sqrt((x- input_point[0])**2 + (y- input_point[1]) ** 2)
        if max_distance < distance:
            max_distance_point = [x,y]
            max_distance = distance
    return [max_distance_point[1],max_distance_point[0]]

def extract_features(image):
    # Extract relevant features from the image
    # For example, you can use color-based features or texture features
    # Here, we'll use a simple color-based feature by converting the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image.flatten()

def select_core_points(features, num_core_points):
    # Apply a coreset selection algorithm to select core points
    # Here, we'll use K-means clustering to select core points
    kmeans = KMeans(n_clusters=num_core_points)
    kmeans.fit(features.reshape(-1, 1))  # Reshape features to a 2D array if necessary
    core_points = kmeans.cluster_centers_.squeeze()
    return core_points

def get_kmeans_points(input_point,mask,image):
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    print(masked_image)
    exit()
    #masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    
    features = extract_features(masked_image)
    print("features:",list(features))
    num_core_points = 5
    core_points = select_core_points(features, num_core_points)
    

    width, height, _ = image.shape
    pixel_coordinates = np.array(core_points) * np.array([width, height])
    for coordinate in pixel_coordinates:
        x, y = coordinate.astype(int)
        print(x,y)
    exit()
    return [core_points[1],core_points[0]]

def gen_pic_with_points(points,predictor,image ,des,path):
    input_point = np.array(points)
    #print(input_point)
    input_label = np.array([1, 1])


    masks, _, _ = predictor.predict(
      point_coords=input_point,
      point_labels=input_label,
      multimask_output=False,
    )

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig(f"{path}/{des}.jpg", bbox_inches="tight", pad_inches=0)

def get_aug_pics(path, predictor,image,bbox, mask):
  if not os.path.exists(path):
    os.makedirs(path)
  center = [bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2]
  predictor.set_image(image)
  input_point = np.array([center])
  input_label = np.array([1])
  
  plt.figure(figsize=(10,10))
  plt.imshow(image)
  show_mask(mask, plt.gca())
  show_points(input_point, input_label, plt.gca())
  plt.axis('on')
  plt.savefig(f"{path}/ori.jpg", bbox_inches="tight", pad_inches=0)

  masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
  )

  best_mask = []
  best_score = 0
  for i, (mask, score) in enumerate(zip(masks, scores)):
      if score > best_score:
        best_score = score
        best_mask = mask
      plt.figure(figsize=(10,10))
      plt.imshow(image)
      show_mask(mask, plt.gca())
      show_points(input_point, input_label, plt.gca())
      plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
      plt.axis('off')
      plt.savefig(f"{path}/sam_mask{i}.jpg", bbox_inches="tight", pad_inches=0)
    
  kmeans_points = get_kmeans_points(center, best_mask,image)
  random_point = get_random_point(best_mask)
  max_entropy_point = get_entropy_points(center, best_mask, image)
  max_dist_point = get_distance_points(center, best_mask,image)
  indices = np.argwhere(best_mask ==True)
  print(kmeans_points)
  print(center, random_point, max_entropy_point, max_dist_point)    


  input_point = np.array([center, random_point])
  gen_pic_with_points(input_point,predictor,image ,"random_points",path)
  input_point = np.array([center, max_entropy_point])
  gen_pic_with_points(input_point,predictor,image ,"max_entropy",path)
  input_point = np.array([center, max_dist_point])
  gen_pic_with_points(input_point,predictor,image ,"max_distance",path)


def test():
    img_local_path = "C:/Users/harold/Desktop/Results/000000133149/000000133149.jpg"
    npy_path = "C:/Users/harold/Desktop/Results/000000133149/ann0_c16_5_saliency_point_mask.npy"
    image = np.asarray(Image.open(img_local_path).convert('RGB'))
    mask =  np.load(npy_path)
    filtered_image = cv2.blur(image, (12, 12))
    plt.figure(figsize=(10,10))
    plt.imshow(filtered_image)
    show_mask(mask, plt.gca())
    plt.show()
    plt.axis('off')
    plt.savefig(f"./saliency_mask.jpg", bbox_inches="tight", pad_inches=0)
test()

exit()
coco_annotation_file_path = r"H:/My Drive/Download/COCO2017/annotations/instances_train2017.json"

coco_annotation = COCO(annotation_file=coco_annotation_file_path)

# Category IDs.
cat_ids = coco_annotation.getCatIds()
print(f"Number of Unique Categories: {len(cat_ids)}")
print("Category IDs:")
print(cat_ids)  # The IDs are not necessarily consecutive.

# All categories.
cats = coco_annotation.loadCats(cat_ids)
cat_names = [cat["name"] for cat in cats]
print("Categories Names:")
print(cat_names)

for i in range(20):
  # Category Name -> Category ID.
  query_name = np.random.choice(cat_names)
  #random choose an ID
  query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
  print("Category Name -> ID:")
  print(f"Category Name: {query_name}, Category ID: {query_id}")
  # Get the ID of all the images containing the object of the category.
  img_ids = coco_annotation.getImgIds(catIds=[query_id])
  print(f"Number of Images Containing {query_name}: {len(img_ids)}")
  # Pick one image.
  img_id = np.random.choice(img_ids)
  img_info = coco_annotation.loadImgs([img_id])[0]
  img_file_name = img_info["file_name"]
  img_url = img_info["coco_url"]
  print(
      f"Image ID: {img_id}, File Name: {img_file_name}, Image URL: {img_url}"
  )
  # Get all the annotations for the specified image.
  ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
  anns = coco_annotation.loadAnns(ann_ids)
  path = f"H:/My Drive/AGI/SAMAug/results/{img_file_name[:-4]}"
  if not os.path.exists(path):
      os.makedirs(path)

  print(f"Annotations for Image ID {img_id}:")
  plt.clf()
  image = cv2.imread(f'H:/My Drive/Download/COCO2017/train2017/{img_file_name}')
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.axis("off")
  plt.imshow(np.asarray(image))
  plt.savefig(f"{path}/{img_id}.jpg", bbox_inches="tight", pad_inches=0)
  # Plot segmentation and bounding box.
  coco_annotation.showAnns(anns, draw_bbox=True)
  plt.savefig(f"{path}/annotated.jpg", bbox_inches="tight", pad_inches=0)
  for  i, annotate in enumerate(anns):
    mask = coco_annotation.annToMask(annotate)
    bbox = annotate['bbox']
    get_aug_pics(f"{path}/mask{i}/",predictor, image, bbox, mask)



