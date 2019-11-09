#! /usr/bin/env python3

import numpy as np

from torch.utils.data import Dataset

from mnist import mnist_image_to_numpy, mnist_label_to_numpy


class MnistDataset(Dataset):

    def __init__(
        self, images_file, labels_file, datatype=np.float32,
        max_images_to_load=None, onehot_encode=True, shape=None
    ):
        self.images = mnist_image_to_numpy(
            images_file, datatype=datatype,
            max_images_to_load=max_images_to_load
        )

        self.labels = mnist_label_to_numpy(
            labels_file, max_images_to_load=max_images_to_load,
            onehot_encode=onehot_encode, datatype=datatype
        )

        if shape:
            self.images = self.images.reshape(*shape)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
