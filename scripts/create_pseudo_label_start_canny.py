import numpy as np
import cv2
import os
from tqdm import tqdm
import sys
import time
sys.path.append("/home/yyf/Workspace/edge_detection/codes/deep_learning_methods/STEdge")
from utils.functions import get_imgs_list, fit_img_postfix
from utils.pseudo_label import create_pseudo_label_with_canny
from skimage import measure


def filter_canny_connectivity(canny_edge, min_thresh):
    labels = measure.label(canny_edge, connectivity=2)  # connectivity=2是8连通区域标记，等于1则是4连通
    props = measure.regionprops(labels)
    # print("total area:", len(props))
    for each_area in props:
        if each_area.area <= min_thresh:
            for each_point in each_area.coords:
                canny_edge[each_point[0]][each_point[1]] = 0
    return canny_edge


def use_more_canny(imgs, t_min, t_max):
    # (h, w) = imgs[0].shape[:2]
    # canny_final = np.zeros((h, w), dtype=np.float64)
    # for each_img in imgs:
    #     canny_temp = cv2.Canny(each_img, t_min, t_max)
    #     canny_final = canny_final + canny_temp
    for index, each_img in enumerate(imgs):
        if index == 0:
            canny_final = cv2.Canny(each_img, t_min, t_max)
        else:
            canny_tmp = cv2.Canny(each_img, t_min, t_max)
            canny_final = cv2.add(canny_final, canny_tmp)
    return canny_final


imgs_path = "/home/yyf/Workspace/COCO_datasets/COCO_COCO_2017_Val_images/val2017/image"
# imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BSDS/BSDS_train/image"
# imgs_path = "/home/speed_kai/yeyunfan_phd/edge_detection/datasets/BIPED/BIPED_few_shot/BIPED_50%/remains_for_self_train/image"

t_min = 300
t_max = 400
output_path = imgs_path + "_start_label_"+str(t_min)+"_"+str(t_max)
if not os.path.exists(output_path):
    os.mkdir(output_path)

imgs = get_imgs_list(imgs_path)
for i in tqdm(range(len(imgs))):
    if i == 5000:
        break
    img_name = os.path.basename(imgs[i])
    img_data = cv2.imread(fit_img_postfix(os.path.join(imgs_path, img_name)))
    img_data = cv2.GaussianBlur(img_data, (5, 5), 0)
    img_data = cv2.bilateralFilter(img_data, 15, 50, 50)   # defualt: one GaussianBlur, one bilateralFilter

    # img_data = cv2.bilateralFilter(img_data, 15, 50, 50)

    canny_edge = cv2.Canny(img_data, t_min, t_max)
    canny_result = filter_canny_connectivity(canny_edge, min_thresh=50)  # defualt 50
    cv2.imwrite(os.path.join(output_path, img_name), canny_result)
