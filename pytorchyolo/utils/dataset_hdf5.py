from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import glob
import random
import os
import warnings
import numpy as np
from PIL import Image
from PIL import ImageFile
import h5py
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

HDF5_PATH = "/public_shared/online/PVDN_thesis_data/bright_spot_data/bright_spot_dataset.hdf5"

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class HDF5Dataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.indices = [int(line.rstrip()) for line in file.readlines()]
        self.hf = h5py.File(HDF5_PATH, "r")
        self.group = self.hf["VASP"]
        self.image_dataset = self.group["images"]
        self.label_dataset = self.group["labels"]
        # self.image_dataset_size = self.image_dataset.shape[0]
        # self.label_dataset_size = self.label_dataset.shape[0]

        # assert (self.image_dataset_size == self.label_dataset_size)
        # print(self.image_dataset_size)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def read_image(self, idx):
        image_bytes = self.image_dataset[idx]
        # print(image_bytes)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_label(self, idx):
        data_str = self.label_dataset[idx].decode('utf-8').split('\n')
        label = np.array([list(map(float, s.split())) for s in data_str if s]).reshape(-1, 5)
        # print(label)
        return label

    def __getitem__(self, index):
        idx = self.indices[index % len(self.indices)]
        # print(idx)

        # ---------
        #  Image
        # ---------
        try:
            img = self.read_image(idx)
        except Exception:
            print(f"Could not read image '{idx}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = self.read_label(idx)
            
        except Exception:
            print(f"Could not read label '{idx}'.")
            return

        # print(img, boxes)
        # print(self.image_labels[index])
        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
            # Ignore warning if file is empty
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return imgs, bb_targets

    def __len__(self):
        return len(self.indices)