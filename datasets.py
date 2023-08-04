import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import measure


def fit_img_postfix(img_path):
    if not os.path.exists(img_path) and img_path.endswith(".jpg"):
        img_path = img_path[:-4] + ".png"
    if not os.path.exists(img_path) and img_path.endswith(".png"):
        img_path = img_path[:-4] + ".jpg"
    return img_path


class Dataset_with_l0smooth_and_name(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            if not(file_name_ext.endswith(".jpg") or file_name_ext.endswith(".png")):
                continue
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)

        img_name = os.path.basename(image_path)
        # imgs_blur_path = fit_img_postfix(os.path.join(self.images_path, "l0smooth", img_name))
        imgs_blur_path = fit_img_postfix(os.path.join(self.images_path + "_gaussian_l0smooth", img_name))
        # print(imgs_blur_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image_blur = cv2.imread(imgs_blur_path, cv2.IMREAD_COLOR)
        # print(image.shape, label.shape, image_blur.shape)

        image, label, image_blur = self.transform(img=image, gt=label, img_blur=image_blur)
        return dict(images=image, labels=label, images_blur=image_blur, img_names=img_name)

    def transform(self, img, gt, img_blur):
        crop_size = self.img_height if self.img_height == self.img_width else 400
        # -----------------------------------------------------------
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255.  # for DexiNed input and BDCN
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()
        # ------------------------blur-----------------------------------
        # img_blur = cv2.bilateralFilter(img, 15, 50, 50)
        # img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)
        # img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)
        # img_blur = cv2.ximgproc.l0Smooth(img, kappa=2)

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))

        img_blur = np.array(img_blur, dtype=np.float32)
        img_blur -= self.mean_bgr
        # print(crop_size)
        img_blur = cv2.resize(img_blur, dsize=(crop_size, crop_size))

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_blur = img_blur.transpose((2, 0, 1))
        img_blur = torch.from_numpy(img_blur.copy()).float()
        return img, gt, img_blur


class CocoDataset(Dataset):
    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False
                 #  arg=None
                 ):
        self.data_root = data_root
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img

        self.data_index = self._build_index()

    def _build_index(self):
        data_root = os.path.abspath(self.data_root)
        images_path = os.path.join(data_root, 'image', "aug")
        labels_path = os.path.join(data_root, 'edge', "aug")

        sample_indices = []
        for directory_name in os.listdir(images_path):
            image_directories = os.path.join(images_path, directory_name)
            for file_name_ext in os.listdir(image_directories):
                file_name = os.path.splitext(file_name_ext)[0]
                sample_indices.append(
                    (os.path.join(images_path, directory_name, file_name + '.png'),
                     os.path.join(labels_path, directory_name, file_name + '.png'))
                )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr

        crop_size = self.img_height if self.img_height == self.img_width else 400
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        # for DexiNed input
        # -----------------------------------
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)

        # For RCF input
        # -----------------------------------
        # gt[gt == 0] = 0
        # gt[np.logical_and(gt > 0., gt < 0.5)] = 2
        # gt[gt >= 0.5] = 1
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt


class CocoDataset_no_aug(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(image_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255. # for DexiNed input and BDCN

        img = np.array(img, dtype=np.float32)

        img -= self.mean_bgr

        crop_size = self.img_height if self.img_height == self.img_width else 400
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        # # For RCF input
        # # -----------------------------------
        # gt[gt==0]=0.
        # gt[np.logical_and(gt>0.,gt<0.5)] = 2.
        # gt[gt>=0.5]=1.
        #
        # gt = gt.astype('float32')
        # ----------------------------------

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt


class Dataset_multiscale(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image, label, image_s1, image_s2 = self.transform(img=image, gt=label)
        return dict(images=image, labels=label, images_s1=image_s1, images_s2=image_s2)

    def transform(self, img, gt):
        crop_size = self.img_height if self.img_height == self.img_width else 400
        # -----------------------------------------------------------
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255. # for DexiNed input and BDCN
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()
        # -----------------------------------------------------------
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))
        img_s1 = cv2.resize(img, None, fx=0.5, fy=0.5)
        img_s2 = cv2.resize(img, None, fx=1.5, fy=1.5)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_s1 = img_s1.transpose((2, 0, 1))
        img_s1 = torch.from_numpy(img_s1.copy()).float()

        img_s2 = img_s2.transpose((2, 0, 1))
        img_s2 = torch.from_numpy(img_s2.copy()).float()

        return img, gt, img_s1, img_s2


class Dataset_with_l0smooth(Dataset):
    def __init__(self,
                 images_path,
                 labels_path,
                 img_height,
                 img_width,
                 mean_bgr,
                 crop_img=False
                 #  arg=None
                 ):
        self.images_path = images_path
        self.labels_path = labels_path
        # self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        for file_name_ext in os.listdir(self.images_path):
            if not(file_name_ext.endswith(".jpg") or file_name_ext.endswith(".png")):
                continue
            file_name = os.path.splitext(file_name_ext)[0]
            sample_indices.append(
                (os.path.join(self.images_path, file_name + '.png'),
                 os.path.join(self.labels_path, file_name + '.png'),)
            )
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]
        image_path = fit_img_postfix(image_path)
        label_path = fit_img_postfix(label_path)

        img_name = os.path.basename(image_path)
        # imgs_blur_path = fit_img_postfix(os.path.join(self.images_path, "l0smooth", img_name))
        imgs_blur_path = fit_img_postfix(os.path.join(self.images_path + "_gaussian_l0smooth", img_name))
        # print(imgs_blur_path)
        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image_blur = cv2.imread(imgs_blur_path, cv2.IMREAD_COLOR)
        # print(image.shape, label.shape, image_blur.shape)

        image, label, image_blur = self.transform(img=image, gt=label, img_blur=image_blur)
        return dict(images=image, labels=label, images_blur=image_blur)

    def transform(self, img, gt, img_blur):
        crop_size = self.img_height if self.img_height == self.img_width else 400
        # -----------------------------------------------------------
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255.  # for DexiNed input and BDCN
        gt = cv2.resize(gt, dsize=(crop_size, crop_size))
        gt[gt > 0.2] += 0.5
        gt = np.clip(gt, 0., 1.)
        gt = torch.from_numpy(np.array([gt])).float()
        # ------------------------blur-----------------------------------
        # img_blur = cv2.bilateralFilter(img, 15, 50, 50)
        # img_blur = cv2.bilateralFilter(img_blur, 15, 50, 50)
        # img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)
        # img_blur = cv2.ximgproc.l0Smooth(img, kappa=2)

        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        # print(crop_size)
        img = cv2.resize(img, dsize=(crop_size, crop_size))

        img_blur = np.array(img_blur, dtype=np.float32)
        img_blur -= self.mean_bgr
        # print(crop_size)
        img_blur = cv2.resize(img_blur, dsize=(crop_size, crop_size))

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        img_blur = img_blur.transpose((2, 0, 1))
        img_blur = torch.from_numpy(img_blur.copy()).float()
        return img, gt, img_blur