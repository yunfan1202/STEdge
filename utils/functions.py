import numpy as np
import cv2
import os
import torch
from .image import image_normalization


device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

def get_imgs_list(imgs_dir):
    imgs_list = os.listdir(imgs_dir)
    imgs_list.sort()
    return [os.path.join(imgs_dir, f) for f in imgs_list if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.pgm')]


def get_confidence_kl(preds, preds_blur):
    preds = torch.sigmoid(preds)
    preds_blur = torch.sigmoid(preds_blur)
    # variance = kl_distance(torch.log(preds), preds_blur)
    # variance = torch.exp(-variance)
    pred1 = torch.stack([preds, 1 - preds], dim=-1)
    pred2 = torch.stack([preds_blur, 1 - preds_blur], dim=-1)
    # print(pred1.shape, pred2.shape) # torch.Size([B, 1, H, W, 2])
    kl_distance = torch.nn.KLDivLoss(reduction="none")
    variance = torch.sum(kl_distance(torch.log(pred1), pred2), dim=-1)
    confidence = torch.exp(-variance)   # torch.Size([B, 1, H, W])
    return confidence


def get_uncertainty_from_all_block(preds_list):
    tensor = torch.stack(preds_list, dim=0) # after stack: torch.Size([7, B, 1, 512, 512])
    tensor = tensor.transpose(0, 1)     # after transpose: torch.Size([B, 7, 1, 512, 512])
    confs = []
    for each_pred in tensor:
        # print(each_pred.shape)  # torch.Size([7, 1, 512, 512])
        blocks_preds = []
        for i in range(len(each_pred)):
            edge_map = torch.sigmoid(each_pred[i]).cpu().detach().numpy()
            edge_map = np.squeeze(edge_map)  # like (512, 512)
            edge_map = np.uint8(image_normalization(edge_map))
            edge_map = cv2.bitwise_not(edge_map)
            blocks_preds.append(edge_map)

        std = get_uncertainty(blocks_preds)
        # hm = cv2.applyColorMap(std.astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imwrite("./all_block_uncertainty.png", hm)
        # print(std.shape, np.max(std), np.mean(std))
        confidence = np.exp(-std)
        # print(confidence.shape, np.max(confidence), np.mean(confidence))
        confs.append(torch.from_numpy(confidence))

    conf_tensors = torch.stack(confs, dim=0).to(device)
    conf_tensors = torch.unsqueeze(conf_tensors, dim=1)
    # print(conf_tensors.shape)       # torch.Size([B, 1, 512, 512])
    return conf_tensors


def adapt_img_name(img_path):
    if (not os.path.exists(img_path)) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if (not os.path.exists(img_path)) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path

def concatenate_images(final_result, row, column, interval, strategy="left2right", background="white"):
    (h, w, _) = final_result[0].shape


    if background == "black":

        for _ in range(row * column - len(final_result)):
            final_result.append(np.zeros((h, w, 3), dtype=np.uint8))


        concat_result = np.zeros((row * h + interval * (row - 1), column * w + interval * (column - 1), 3),
                                 dtype=np.uint8)

    else:
        for _ in range(row * column - len(final_result)):
            temp = np.zeros((h, w, 3), dtype=np.uint8)
            temp[temp == 0] = 255
            final_result.append(temp)

        concat_result = np.zeros((row * h + interval * (row - 1), column * w + interval * (column - 1), 3),
                                 dtype=np.uint8)
        concat_result[concat_result == 0] = 255

    for r in range(row):
        for c in range(column):
            if strategy == "left2right":
                index = r * column + c
            if strategy == "top2down":
                index = c * row + r
            # print(index)
            if len(final_result[index].shape) != 3:
                final_result[index] = cv2.cvtColor(final_result[index], cv2.COLOR_GRAY2BGR)
            range_h1 = r * h + r * interval
            range_h2 = (r + 1) * h + r * interval
            range_w1 = c * w + c * interval
            range_w2 = (c + 1) * w + c * interval
            concat_result[range_h1: range_h2, range_w1: range_w2] = final_result[index]
    return concat_result


def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path
