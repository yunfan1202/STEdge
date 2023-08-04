import os
import cv2
from tqdm import tqdm
import numpy as np
import shutil


def get_imgs_list(imgs_dir):
    imgs_list = os.listdir(imgs_dir)
    imgs_list.sort()
    return [os.path.join(imgs_dir, f) for f in imgs_list if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]

def get_edge_pixel_num(data):
    data[data < 122] == 0
    data[data >= 122] == 255
    count = np.sum(data == 255)
    return count


def select_img_samples(work_dir, select_ratio):
    COCO_dir = "/home/yyf/Workspace/COCO_datasets/COCO_COCO_2017_Val_images/val2017/image"
    imgs_path = get_imgs_list(os.path.join(work_dir, "start", "edges"))

    # select_ratio = 1.1
    target_dir = os.path.join(work_dir, "select_ratio" + str(select_ratio), "image", "real")
    os.makedirs(target_dir, exist_ok=True)

    dir_names = ["after_round1", "after_round2", "after_round3", "after_round4", "after_round5", "after_round6",
                 "after_round7", "after_round8"]
    for i in tqdm(range(len(imgs_path))):
        if i == 50000:
            break
        name = os.path.basename(imgs_path[i])
        edge_data = cv2.imread(imgs_path[i], 0)

        if select_ratio == "all":
            shutil.copy(os.path.join(COCO_dir, name[:-4] + ".jpg"), os.path.join(target_dir, name[:-4] + ".jpg"))
        else:
            count_start = get_edge_pixel_num(edge_data)
            counts = [count_start]
            ratios = [1]
            for dir_name in dir_names:
                edge_path_r = os.path.join(work_dir, dir_name, "edges", name)
                edge_data_r = cv2.imread(edge_path_r, 0)
                if edge_data_r is None:
                    continue
                count_r = get_edge_pixel_num(edge_data_r)
                counts.append(count_r)
                ratios.append(round(count_r / count_start, 4))
            if max(ratios) <= select_ratio:
                # print(max(ratios))
                shutil.copy(os.path.join(COCO_dir, name[:-4] + ".jpg"), os.path.join(target_dir, name[:-4] + ".jpg"))


def uncertainty_pseudo_label(work_dir, select_ratio):
    select_imgs_dir = os.path.join(work_dir, "select_ratio" + str(select_ratio), "image", "real")
    target_edge_dir = os.path.join(work_dir, "select_ratio" + str(select_ratio), "edge", "real")

    os.makedirs(target_edge_dir, exist_ok=True)
    selected_names = []
    for each in os.listdir(select_imgs_dir):
        selected_names.append(each[:-4])

    dir_names = ["start", "after_round1", "after_round2", "after_round3", "after_round4", "after_round5", "after_round6",
                 "after_round7", "after_round8"]

    imgs_path = get_imgs_list(select_imgs_dir)
    for i in tqdm(range(len(imgs_path))):
        if i == 50000:
            break
        name = os.path.basename(imgs_path[i])[:-4] + ".png"
        if name[:-4] not in selected_names:
            continue
        # ---------------begin the processing--------------
        img_data = cv2.imread(imgs_path[i])
        H, W = img_data.shape[:2]

        fuse = np.zeros((H, W), np.float32)
        try:
            for dir in dir_names:
                edge_path = os.path.join(work_dir, dir, "edges", name)
                edge_data = cv2.imread(edge_path, 0)
                fuse = np.add(fuse, edge_data)
            fuse = fuse / len(dir_names)
            fuse = fuse.astype(np.uint8)
            fuse[np.logical_and(fuse > 0, fuse < 0.9 * 240)] = 0.25 * 255
            cv2.imwrite(os.path.join(target_edge_dir, name), fuse)
        except:
            print("Error: ", name)


work_dir = "/home/yyf/Workspace/edge_detection/codes/deep_learning_methods/STEdge/datasets/COCO/COCO_val2017_consistency_mse1_interval2_fuse_union"
select_ratio = "all"
select_img_samples(work_dir, select_ratio)
uncertainty_pseudo_label(work_dir, select_ratio)