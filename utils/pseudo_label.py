import numpy as np
import cv2
from .canny import filter_canny_connectivity


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


def create_pseudo_label_with_blur_canny_union(img_data, pred_data):
    _, pred_data_binary = cv2.threshold(pred_data, 0, 255, cv2.THRESH_OTSU)
    pred_data_binary = cv2.bitwise_not(pred_data_binary)

    # -----------------------combine with the adaptive one------------------------
    pred_data_binary2 = cv2.adaptiveThreshold(pred_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 20)
    pred_data_binary2 = cv2.bitwise_not(pred_data_binary2)
    pred_data_binary = cv2.bitwise_or(pred_data_binary, pred_data_binary2)
    # ----------------------------------------------------------------------------
    # pred_data_binary = filter_canny_connectivity(pred_data_binary, min_thresh=50)
    pred_data_binary[pred_data_binary == 255] = 1
    (h, w) = pred_data.shape[:2]
    merge_result = np.zeros((h, w), dtype=np.float64)

    img_blur_g = cv2.GaussianBlur(img_data, (5, 5), 0)
    img_blur_g_b1 = cv2.bilateralFilter(img_blur_g, 15, 50, 50)
    img_blur_g_b2 = cv2.bilateralFilter(img_blur_g_b1, 15, 50, 50)
    imgs_data = [img_blur_g_b1, img_blur_g_b2]
    # imgs_data = [img_blur_g]
    canny_ranges = [(20, 40)]
    for j, canny_range in enumerate(canny_ranges):
        edge_canny = use_more_canny(imgs_data, canny_range[0], canny_range[1])
        merge_tmp = pred_data_binary * edge_canny
        merge_result = cv2.add(merge_result, merge_tmp / len(canny_ranges))
    merge_result = filter_canny_connectivity(merge_result, min_thresh=30)
    return merge_result

