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
import lmdb, pickle, mmcv, h5py
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True
# DB_PATH = "/private_shared/Projects/PyTorch-YOLOv3/data/custom/images_hdf5_test/images_test.hdf5"
# PKL_PATH = "/private_shared/Projects/PyTorch-YOLOv3/data/custom/labels_test.pkl"
PKL_PATH = "/private_shared/Projects/PyTorch-YOLOv3/data/custom/labels.pkl"
# DB_PATH = "/public_shared/online/images_hdf5/images.hdf5"
# PKL_PATH = "/public_shared/online/labels_pkl/labels.pkl"


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        print(img, boxes)
        # print(self.image_labels[index])
        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

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

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)

class ListDataset_PKL(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.pkl_path = PKL_PATH
        self._init_pkl()

    def _init_pkl(self):
        with open(self.pkl_path, 'rb') as handle:
            self.key_labels = pickle.load(handle)

    def read_pkl(self, key):
        # print(key)
        data_str = self.key_labels[key].decode('utf-8').split('\n')
        label = np.array([list(map(float, s.split())) for s in data_str if s]).reshape(-1, 5)
        # print(label)
        return label

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        img_label = img_path.split("/")[-1].split(".")[0]
        # label_path = self.label_files[index % len(self.img_files)].rstrip()
        try:
            # Ignore warning if file is empty
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     boxes_1 = np.loadtxt(label_path).reshape(-1, 5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = self.read_pkl(str(img_label))
            # print(boxes_1, boxes_2)
            # assert(boxes_1 == boxes_2)
            
        except Exception:
            print(f"Could not read label '{img_label}'.")
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

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

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

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)


class my_dataset_LMDB_PKL(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        self.db_path = DB_PATH
        self.pkl_path = PKL_PATH
        self.key_labels = {}

        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.image_labels = [path.split("/")[-1].split(".")[0] for path in self.img_files]

        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self._init_db()
        self._init_pkl()

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        

    def _init_db(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        self.txn = self.env.begin()
        
    
    def _init_pkl(self):
        with open(self.pkl_path, 'rb') as handle:
            self.key_labels = pickle.load(handle)

    def read_lmdb(self, key):
        lmdb_data = self.txn.get(key.encode("ascii"))
        lmdb_data = np.frombuffer(lmdb_data, dtype=np.uint8)
        lmdb_data = cv2.imdecode(lmdb_data, cv2.IMREAD_COLOR)
        pil_image = np.array(Image.fromarray(lmdb_data).convert('RGB'), dtype=np.uint8)
        return pil_image

    def read_pkl(self, key):
        data_str = self.key_labels[key].decode('utf-8').split('\n')
        label = np.array([list(map(float, s.split())) for s in data_str if s])
        return label

    def __getitem__(self, index):
        img = self.read_lmdb(self.image_labels[index])
        boxes = self.read_pkl(self.image_labels[index])
        # print(img, boxes)
        print(self.image_labels[index])
        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return
        return None, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        __, imgs, bb_targets = list(zip(*batch))

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

        return __, imgs, bb_targets
    
    def __len__(self):
        return len(self.image_labels)

class my_dataset_HDF5_PKL(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        self.db_path = DB_PATH
        self.pkl_path = PKL_PATH
        self.key_labels = {}

        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.image_labels = [path.split("/")[-1].split(".")[0] for path in self.img_files]

        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self._init_db()
        self._init_pkl()

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        

    def _init_db(self):
        self.hf = h5py.File(self.db_path, "r")  # open a hdf5 file
        
    def _init_pkl(self):
        with open(self.pkl_path, 'rb') as handle:
            self.key_labels = pickle.load(handle)

    def read_hdf5(self, key):
        img_bytes = np.array(self.hf[str(key)])  # read image in bytes
        img = mmcv.imfrombytes(
            img_bytes, flag="color", channel_order="rgb"
        )  # convert to array
        return img

    def read_pkl(self, key):
        data_str = self.key_labels[key].decode('utf-8').split('\n')
        label = np.array([list(map(float, s.split())) for s in data_str if s]).reshape(-1, 5)
        return label

    def __getitem__(self, index):
        try:

            img = self.read_hdf5(self.image_labels[index])
        except Exception:
            print(f"Could not read image '{self.image_labels[index]}'.")
            return
        
        try:
            boxes = self.read_pkl(self.image_labels[index])
        except Exception:
            print(f"Could not read label '{self.image_labels[index]}'.")
            return
        
        # print(img, boxes)
        # print(self.image_labels[index])
        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return
        return None, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        __, imgs, bb_targets = list(zip(*batch))

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

        return __, imgs, bb_targets
    
    def __len__(self):
        return len(self.image_labels)