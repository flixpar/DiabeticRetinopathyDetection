import os
import csv
import glob
import random
import numpy as np
import pandas as pd

import torch
import torchvision

import cv2
from PIL import Image
import albumentations as tfms

import sklearn

class RetinaImageDataset(torch.utils.data.Dataset):

	def __init__(self, split, args, transforms=None, test_transforms=None, debug=False, n_samples=None):
		self.split = split
		self.transforms = transforms
		self.test_transforms = test_transforms if test_transforms else None
		self.full_size = args.full_size
		self.debug = debug
		self.n_classes = 2
		self.resize = tfms.Resize(args.img_size, args.img_size) if args.img_size is not None else None
		self.base_path = args.datapath
		self.n_samples = n_samples
		if self.debug: self.n_samples = 128

		self.ann_data = pd.read_csv(os.path.join(self.base_path, "ann.csv"))
		labels = [a[4] for a in self.ann_data]

		# subsampling
		if self.n_samples is not None and self.n_samples < len(self.data):
			self.data = random.sample(self.data, self.n_samples)

		# class and example weighting

		self.class_weights = np.sum(labels, axis=0).astype(np.float32)
		self.class_weights[self.class_weights == 0] = np.inf
		self.class_weights = self.class_weights[self.class_weights != np.inf].max() / self.class_weights
		self.class_weights = self.class_weights / self.n_classes

		self.example_weights = np.asarray(labels) * self.class_weights[np.newaxis, :]
		self.example_weights = np.sum(self.example_weights, axis=1)

		self.class_weights   = torch.tensor(self.class_weights, dtype=torch.float32)
		self.example_weights = torch.tensor(self.example_weights, dtype=torch.float32)

		# set the image normalization
		img_mean = [0.06898253, 0.17419075, 0.16167488]
		img_std  = [0.06259116, 0.09672542, 0.10255357]
		self.normalization = tfms.Normalize(mean=img_mean, std=img_std)

	def __getitem__(self, index):

		person, eye, quality, file_num, seafan, laser = self.data[index]

		fn = os.path.join(self.base_path, "batch6", "{}.tif".format(file_num))
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

		return img, seafan

	def __len__(self):
		return len(self.data)
