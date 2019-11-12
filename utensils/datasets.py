#! /usr/bin/env python3

import numpy as np

from torch.utils.data import Dataset

from utensils.mnist import mnist_image_to_numpy
from utensils.mnist import mnist_label_to_numpy


class MnistDataset(Dataset):

    def __init__(
        self, images_file, labels_file, dtype=np.float32,
        max_images_to_load=None, onehot_encode=True, shape=None
    ):
        self.images = mnist_image_to_numpy(
            images_file, dtype=dtype,
            max_images_to_load=max_images_to_load
        )

        self.labels = mnist_label_to_numpy(
            labels_file, max_images_to_load=max_images_to_load,
            onehot_encode=onehot_encode, dtype=dtype
        )

        if shape:
            self.images = self.images.reshape(
                self.images.shape[0], *shape
            )

        self.images /= 255

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class NflRushingDataset(Dataset):
    def __init__(
        self, images_file, labels_file, dtype=np.float32, shape=None
    ):
        self.images = np.load(images_file).astype(dtype)
        self.labels = np.load(labels_file).astype(dtype)

        if shape:
            self.images = self.images.reshape(
                self.images.shape[0], *shape
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CreditCardDataset(Dataset):
    def __init__(
        self, data_file, labels_file=None,
        amounts_file=None, dtype=np.float32
    ):
        self.data = np.load(data_file).astype(dtype)

        if not(labels_file):
            self.labels = self.data[:, -1]
            self.amounts = self.data[:, -2]
            self.data = self.data[:, :-2]

        else:
            self.labels = np.load(labels_file).astype(dtype)
            self.amounts = np.load(amounts_file).astype(dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.amounts[idx], self.labels[idx]
