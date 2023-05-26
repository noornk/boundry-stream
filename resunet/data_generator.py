# +


"""" Modified version of https://github.com/jeffwen/road_building_extraction/blob/master/src/utils/data_utils.py """
from __future__ import print_function, division
from torch.utils.data import Dataset
from skimage import io
import glob
import os
import torch
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
"""
Data Generator
"""
import os
import numpy as np
import cv2
# from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

# +
def parse_image(img_path, image_size):
#     print(img_path, "aaaa")
    image_rgb = np.load(img_path)
#     print(image)
#     image_rgb = image_rgb.astype('float32')
#     print(image_rgb.shape)
#     image_rgb = cv2.imread(img_path, 1)
    h, w = image_rgb.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        image_rgb = cv2.resize(image_rgb, (image_size, image_size))
    image_rgb = np.expand_dims(image_rgb, -1)
#     image_rgb = np.expand_dims(image_rgb, -1)
#     print(image_rgb.shape, "img")
#     image_rgb = image_rgb/255.0
#     image_rgb = (image_rgb - np.min(image_rgb)) * (255.0 / (np.max(image_rgb) - np.min(image_rgb)))
    
#     plt.imshow(image_rgb)
#     plt.show()
    return image_rgb


# -

def parse_mask(mask_path, image_size):
    mask = np.load(mask_path)
#     mask = image.astype('float32')
#     mask = cv2.imread(mask_path, -1)
    h, w = mask.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        mask = cv2.resize(mask, (image_size, image_size))
#     print(mask.shape, "1")
    mask = np.expand_dims(mask, -1)
#     mask = np.expand_dims(mask, -1)
#     print(mask.shape, "2")
#     mask = mask/255.0
#     mask = (mask - np.min(mask)) * (255.0 / (np.max(mask) - np.min(mask)))
#     plt.imshow(mask)
#     plt.show()
    return mask

# +
# class DataGen(Sequence):
#     def __init__(self, image_size, images_path, masks_path, batch_size=8):
#         self.image_size = image_size
#         self.images_path = images_path
#         self.masks_path = masks_path
#         self.batch_size = batch_size
#         self.on_epoch_end()

#     def __getitem__(self, index):
#         if(index+1)*self.batch_size > len(self.images_path):
#             self.batch_size = len(self.images_path) - index*self.batch_size

#         images_path = self.images_path[index*self.batch_size : (index+1)*self.batch_size]
#         masks_path = self.masks_path[index*self.batch_size : (index+1)*self.batch_size]

#         images_batch = []
#         masks_batch = []

#         for i in range(len(images_path)):
#             ## Read image and mask
#             image = parse_image(images_path[i], self.image_size)
# #             print(image.shape)
#             mask = parse_mask(masks_path[i], self.image_size)

#             images_batch.append(image)
# #             print(images_batch, len(images_batch))
#             masks_batch.append(mask)

#         return np.array(images_batch), np.array(masks_batch)
# #         return images_batch, masks_batch



#     def on_epoch_end(self):
#         pass

#     def __len__(self):
#         return int(np.ceil(len(self.images_path)/float(self.batch_size)))

# +
# class DataGen(Sequence):
#     def __init__(self, image_size, images_path, masks_path, batch_size=8):
#         self.image_size = image_size
#         self.images_path = images_path
#         self.masks_path = masks_path
#         self.batch_size = batch_size
#         self.on_epoch_end()

#     def __getitem__(self, index):
#         if(index+1)*self.batch_size > len(self.images_path):
#             self.batch_size = len(self.images_path) - index*self.batch_size

#         images_path = self.images_path[index*self.batch_size : (index+1)*self.batch_size]
#         masks_path = self.masks_path[index*self.batch_size : (index+1)*self.batch_size]

#         images_batch = []
#         masks_batch = []

#         for i in range(len(images_path)):
#             ## Read image and mask
#             image = parse_image(images_path[i], self.image_size)
# #             print(image.shape)
#             mask = parse_mask(masks_path[i], self.image_size)

#             images_batch.append(image)
# #             print(images_batch, len(images_batch))
#             masks_batch.append(mask)

#         return np.array(images_batch), np.array(masks_batch)
# #         return images_batch, masks_batch



#     def on_epoch_end(self):
#         pass

#     def __len__(self):
#         return int(np.ceil(len(self.images_path)/float(self.batch_size)))

# +

# import gensim

class Dataset_seq(Dataset):
    def __init__(self, image_size, images_path, masks_path, batch_size=1):
        self.image_size = image_size
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
#         print(batch_size)
        self.on_epoch_end()

    def __getitem__(self, idx):
#         print(idx)
        index = idx
# 		# return the seq and label 
# 		seq = self.preprocess(self.data[index])
# 		label = self.label[index]
# 		return seq, label
        if(index+1)*self.batch_size > len(self.images_path):
            self.batch_size = len(self.images_path) - index*self.batch_size

        images_path = self.images_path[index*self.batch_size : (index+1)*self.batch_size]
        masks_path = self.masks_path[index*self.batch_size : (index+1)*self.batch_size]

        images_batch = []
        masks_batch = []

#         for i in range(len(images_path)):
#             ## Read image and mask
#             image = parse_image(images_path[i], self.image_size)
#             image = torch.from_numpy(image)
# #             print(image.shape)
#             mask = parse_mask(masks_path[i], self.image_size)
#             mask = torch.from_numpy(mask)

#             images_batch.append(image)
# #             print(images_batch, len(images_batch))
#             masks_batch.append(mask)
    
# 
#         idx = index
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = parse_image(images_path[idx], self.image_size)
        image = torch.from_numpy(image)
#         print(image.shape, "1")

        image = image.permute(2, 0, 1)
        image = image.float()
#         print(image.shape, "2")

        mask = parse_mask(masks_path[idx], self.image_size)
        
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        mask=mask.float()
        
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([mask])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'mask': mask}

#         if self.transform:
#             sample = self.transform(sample)

        return sample
# 

#         return np.array(images_batch), np.array(masks_batch)

    def __len__(self):
        return int(np.ceil(len(self.images_path)/float(self.batch_size)))
#         return(len(self.data))

# 	def preprocess(self, text):
# 		# used to convert line into tokens and then into their corresponding numericals values using word2id
# 		line = gensim.utils.simple_preprocess(text)
# 		seq = []
# 		for word in line:
# 			if word in self.word2id:
# 				seq.append(self.word2id[word])
# 			else:
# 				seq.append(self.word2id['<unk>'])
# 		#convert list into tensor
# 		seq = torch.from_numpy(np.array(seq))
# 		return seq
    
    def on_epoch_end(self):
        pass
# +
class ImageDataset(Dataset):
    """Massachusetts Road and Building dataset"""

    def __init__(self, mask_list, image_list, train=True, transform=None, image_size=512):
        """
        Args:
            csv_file (string): Path to the csv file with image paths
            train_valid_test (string): 'train', 'valid', or 'test'
            root_dir (string): 'mass_roads', 'mass_roads_crop', or 'mass_buildings'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.train = train
#         self.path = hp.train if train else hp.valid
        self.mask_list = mask_list
        self.image_list = image_list
        self.transform = transform
        self.image_size = 512

    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, idx):
#         maskpath = self.mask_list[idx]
#         imgpath = self.image_list[idx]
        image = parse_image(self.image_list[idx], self.image_size)
        image = torch.from_numpy(image)
#         print(image.shape, "1")

        image = image.permute(2, 0, 1)
        image = image.float()
        
        mask = parse_image(self.mask_list[idx], self.image_size)
        mask = torch.from_numpy(mask)
#         print(image.shape, "1")

        mask = mask.permute(2, 0, 1)
        mask = mask.float()
#         image = io.imread(maskpath.replace("mask_crop", "input_crop"))
#         mask = io.imread(maskpath)

        sample = {"image": image, "mask": mask}

#         if self.transform:
#             sample = self.transform(sample)

        return sample


class ToTensorTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sat_img, map_img = sample["image"], sample["mask"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {
            "sat_img": transforms.functional.to_tensor(sat_img),
            "map_img": torch.from_numpy(map_img).unsqueeze(0).float().div(255),
        }  # unsqueeze for the channel dimension


class NormalizeTarget(transforms.Normalize):
    """Normalize a tensor and also return the target"""

    def __call__(self, sample):
        return {
            "image": transforms.functional.normalize(
                sample["image"], self.mean, self.std
            ),
            "mask": sample["mask"],
        }


# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
# -



