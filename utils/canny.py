import numpy as np
import cv2
from skimage import measure

def merge_dl_canny(pred_data, edge_canny, threshold):
    (h, w) = pred_data.shape
    binary_map = np.zeros((h, w), dtype=np.uint8)
    binary_map[pred_data <= threshold] = 0
    binary_map[pred_data > threshold] = 1
    merged_map = binary_map * edge_canny
    return merged_map


def filter_canny_connectivity(canny_edge, min_thresh):
    labels = measure.label(canny_edge, connectivity=2)
    props = measure.regionprops(labels)
    # print("total area:", len(props))

    for each_area in props:
        if each_area.area <= min_thresh:
            for each_point in each_area.coords:
                canny_edge[each_point[0]][each_point[1]] = 0
    return canny_edge


def filter_canny_connectivity_old(canny_edge, min_thresh):
    labels = measure.label(canny_edge, connectivity=2)
    props = measure.regionprops(labels)
    # print("total area:", len(props))

    for each_area in props:
        if each_area.area <= 2 * min_thresh:
            if each_area.area <= min_thresh:
                for each_point in each_area.coords:
                    canny_edge[each_point[0]][each_point[1]] = 0
            else:
                sum = 0
                for each_point in each_area.coords:
                    sum = sum + canny_edge[each_point[0]][each_point[1]] / 255
                avg = sum / each_area.area
                # print(avg)
                if avg < 0.7:
                    for each_point in each_area.coords:
                        canny_edge[each_point[0]][each_point[1]] = 0
    return canny_edge


def merge_canny4pred(pred_data, threshold, img_data, filter_thresh=15):
    edge_canny = cv2.Canny(img_data, 10, 10, apertureSize=3, L2gradient=False)
    (h, w) = pred_data.shape
    binary_map = np.zeros((h, w), dtype=np.uint8)
    binary_map[pred_data <= threshold] = 0
    binary_map[pred_data > threshold] = 1
    merge_result = binary_map * edge_canny
    if filter_thresh:
        merge_result = filter_canny_connectivity(merge_result, min_thresh=15)
    return merge_result
