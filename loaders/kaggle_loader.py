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

class KaggleRetinaDataset(torch.utils.data.Dataset):

	def __init__(self, split, args, transforms=None, test_transforms=None, debug=False, n_samples=None):
		self.split = split
		self.transforms = transforms
		self.test_transforms = test_transforms if test_transforms else None
		self.full_size = args.img_size
		self.debug = debug
		self.n_classes = 2
		self.resize = tfms.Resize(args.img_size, args.img_size) if args.img_size is not None else None
		self.base_path = os.path.join(args.kaggle_datapath, "train")
		self.n_samples = n_samples
		if self.debug: self.n_samples = 128 if n_samples is None else min([128, n_samples])

		if self.split == "train":
			self.data = pd.read_csv(os.path.join(args.kaggle_datapath, "train_ann.csv")).values
		elif self.split == "val":
			self.data = pd.read_csv(os.path.join(args.kaggle_datapath, "val_ann.csv")).values
		elif self.split == "trainval":
			d1 = pd.read_csv(os.path.join(args.kaggle_datapath, "train_ann.csv"))
			d2 = pd.read_csv(os.path.join(args.kaggle_datapath, "val_ann.csv"))
			self.data = pd.concat([d1, d2]).values
		else:
			raise ValueError("Invalid dataset split.")

		# subsampling
		if self.n_samples is not None and self.n_samples < len(self.data):
			ind = np.random.choice(self.data.shape[0], self.n_samples, replace=False)
			self.data = self.data[ind]

		# class and example weighting
		labels = torch.tensor(self.data[:,1].astype(np.float))
		self.class_weights = [1-labels.mean(), labels.mean()]
		self.class_weights = torch.tensor(self.class_weights)
		self.example_weights = torch.zeros(len(self.data), dtype=torch.float32)
		self.example_weights[labels == 0] = self.class_weights[0]
		self.example_weights[labels == 1] = self.class_weights[1]

		# set the image normalization
		img_mean = [0.06898253, 0.17419075, 0.16167488]
		img_std  = [0.06259116, 0.09672542, 0.10255357]
		self.normalization = tfms.Normalize(mean=img_mean, std=img_std)

	def __getitem__(self, index):

		img_id, lbl = self.data[index]

		fn = os.path.join(self.base_path, f"{img_id}.jpeg")
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

		return img, lbl

	def __len__(self):
		return len(self.data)
