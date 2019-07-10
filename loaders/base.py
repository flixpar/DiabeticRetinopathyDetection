import os
import random
import numpy as np
import pandas as pd

import torch
import torchvision

import cv2
import albumentations as tfms

class BaseDataset(torch.utils.data.Dataset):

	def __init__(self, split, args, transforms=None, test_transforms=None, debug=False, n_samples=None):
		self.split = split
		self.args = args
		self.transforms = transforms
		self.test_transforms = test_transforms if test_transforms else None
		self.full_size = args.img_size
		self.debug = debug
		self.n_classes = 5
		self.resize = tfms.Resize(args.img_size, args.img_size) if args.img_size is not None else None
		self.n_samples = n_samples
		if self.debug: self.n_samples = 128 if n_samples is None else min([128, n_samples])

	def setup(self):

		# set the image normalization
		self.normalization = tfms.Normalize(mean=self.img_mean, std=self.img_std)

		if self.split == "test": return

		# subsampling
		if self.n_samples is not None and self.n_samples < len(self.data):
			ind = np.random.choice(self.data.shape[0], self.n_samples, replace=False)
			self.data = self.data[ind]

		# class and example weighting
		labels = torch.tensor(self.data[:,1].astype(np.int))
		self.class_weights = torch.bincount(labels) / len(labels)
		self.class_weights = 1 - self.class_weights
		self.example_weights = torch.zeros(len(self.data), dtype=torch.float32)
		for i in range(self.n_classes):
			self.example_weights[labels == i] = self.class_weights[i]

	def __getitem__(self, index):

		if self.split != "test": img_id, lbl = self.data[index]
		else:                    img_id = self.data[index]

		fn = os.path.join(self.base_path, self.img_folder, f"{img_id}.{self.extension}")
		img = cv2.imread(fn)

		if self.resize is not None:
			img = self.resize(image=img)["image"]
		if self.transforms is not None:
			img = self.transforms(image=img)["image"]

		if self.test_transforms is not None:
			imgs = [self.normalization(image=t(image=img)["image"])["image"] for t in self.test_transforms]
			imgs = np.asarray(imgs)
			if len(imgs.shape) == 3: imgs = imgs[:,:,:,np.newaxis]
			imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2)))

		img = self.normalization(image=img)["image"]
		img = img.transpose(2, 0, 1)
		img = torch.from_numpy(img)

		if self.test_transforms is not None:
			img = torch.cat((img.unsqueeze(0), imgs), dim=0)

		if self.split != "test": return img, lbl
		else:                    return img, img_id

	def __len__(self):
		return len(self.data)
