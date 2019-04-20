import os
import csv
import glob
import random
import numpy as np

import torch
import torchvision

import cv2
from PIL import Image
import albumentations as tfms

import sklearn
import iterstrat
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

class ProteinImageDataset(torch.utils.data.Dataset):

	def __init__(self, split, args, transforms=None, test_transforms=None, channels="g", debug=False, n_samples=None):
		self.split = split
		self.transforms = transforms
		self.test_transforms = test_transforms if test_transforms else None
		self.image_channels = channels
		self.full_size = args.full_size
		self.debug = debug
		self.n_classes = 28
		self.resize = tfms.Resize(args.img_size, args.img_size) if args.img_size is not None else None
		self.base_path = args.primary_datapath if not args.full_size else args.fullsize_datapath
		self.n_samples = n_samples
		if self.debug: self.n_samples = 128

		# check for valid image mode
		if not (set(self.image_channels) <= set("rgby")):
			raise ValueError("Invalid image channels selection.")

		# split the training set into training and validation
		if split in ["train", "val", "trainval"]:
			with open(os.path.join(self.base_path, 'train.csv'), 'r') as f:
				csvreader = csv.reader(f)
				data = list(csvreader)[1:]
			label_lookup = {k:np.array(v.split(' ')) for k,v in data}

			ids  = sorted(list(label_lookup.keys()))
			lbls = [self.encode_label(label_lookup[k]) for k in ids]

			ids  = np.asarray(ids).reshape(-1, 1)
			lbls = np.asarray(lbls)

			msss = MultilabelStratifiedShuffleSplit(n_splits=1, train_size=args.trainval_ratio, test_size=None, random_state=0)
			train_inds, val_inds = list(msss.split(ids, lbls))[0]

			train_ids = ids[train_inds].flatten().tolist()
			val_ids   = ids[val_inds].flatten().tolist()
			ids       = ids.flatten().tolist()

			# if using external data, add it
			self.source_lookup = {i: "trainval" for i in ids}
			if args.use_external:
				with open(os.path.join(args.primary_datapath, 'external.csv'), 'r') as f:
					csvreader = csv.reader(f)
					external_data = list(csvreader)[1:]
				external_label_lookup = {k:np.array(v.split(' ')) for k,v in external_data}
				external_ids  = sorted(list(external_label_lookup.keys()))
				self.source_lookup.update({i: "external" for i in external_ids})
				label_lookup.update(external_label_lookup)
				ids = ids + external_ids
				train_ids = train_ids + external_ids

		# select data
		if self.split == "train":
			self.data = [(i, label_lookup[i]) for i in train_ids]
		elif self.split == "val":
			self.data = [(i, label_lookup[i]) for i in val_ids]
		elif self.split == "trainval":
			self.data = [(i, label_lookup[i]) for i in ids]
		elif self.split == "test":
			with open(os.path.join(self.base_path, 'sample_submission.csv'), 'r') as f:
				lines = list(csv.reader(f))[1:]
				test_ids = [line[0] for line in lines]
			self.data = [(i, None) for i in test_ids]
			self.test_ids = test_ids
			self.source_lookup = {i: "test" for i in test_ids}
		else:
			raise Exception("Invalid dataset split.")

		# subsampling
		if self.n_samples is not None and self.n_samples < len(self.data):
			self.data = random.sample(self.data, self.n_samples)

		# class and example weighting
		if self.split == "train" or self.split == "trainval":

			labels = [self.encode_label(l[1]) for l in self.data]

			self.class_weights = np.sum(labels, axis=0).astype(np.float32)
			self.class_weights[self.class_weights == 0] = np.inf
			self.class_weights = self.class_weights[self.class_weights != np.inf].max() / self.class_weights
			self.class_weights = self.class_weights / self.n_classes

			self.example_weights = np.asarray(labels) * self.class_weights[np.newaxis, :]
			self.example_weights = np.sum(self.example_weights, axis=1)

			self.class_weights   = torch.tensor(self.class_weights, dtype=torch.float32)
			self.example_weights = torch.tensor(self.example_weights, dtype=torch.float32)

		# set the image normalization
		p_mean = [0.08033423981012082, 0.05155526791740866,  0.05359709020876417,  0.0811968791288488]
		p_std  = [0.1313705843029108,  0.08728413305330673,  0.13922084421796302,  0.12760922364487468]
		t_mean = [0.05860568283679439, 0.04606191081626742,  0.03982708801568723,  0.06027994646558575]
		t_std  = [0.10238559670323068, 0.08069846376704155,  0.10501834094962233,  0.09908335311368136]
		e_mean = [0.03775239471734739, 0.04191453443041034,  0.007705539179783242, 0.0942332991656135]
		e_std  = [0.05167756366610396, 0.061291035726105815, 0.019559849511340346, 0.13389048820718571]
		if self.image_channels == "g":
			p_mean, p_std = p_mean[2], p_std[2]
			t_mean, t_std = t_mean[2], t_std[2]
			e_mean, e_std = e_mean[2], e_std[2]
		elif self.image_channels == "rgb":
			p_mean, p_std = p_mean[:3], p_std[:3]
			t_mean, t_std = t_mean[:3], t_std[:3]
			e_mean, e_std = e_mean[:3], e_std[:3]
		elif self.image_channels == "rgby":
			pass
		else:
			raise NotImplementedError("Unsupported image channels selection.")

		self.primary_normalization  = tfms.Normalize(mean=p_mean, std=p_std)
		self.test_normalization     = tfms.Normalize(mean=t_mean, std=t_std)
		self.external_normalization = tfms.Normalize(mean=e_mean, std=e_std)

	def __getitem__(self, index):

		example_id, label = self.data[index]
		example_source = self.source_lookup[example_id]

		if example_source == "trainval": imgdir = "train"
		elif example_source == "test": imgdir = "test"
		else: imgdir = "external"
		img_folder = os.path.join(self.base_path, imgdir)

		ext = ".tif" if self.full_size else ".png"
		if self.image_channels == "g":
			fn = os.path.join(img_folder, example_id + "_green" + ext)
			img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

		elif set(self.image_channels) == set("rgb"):
			r = cv2.imread(os.path.join(img_folder, example_id + "_red"   + ext), cv2.IMREAD_GRAYSCALE)
			g = cv2.imread(os.path.join(img_folder, example_id + "_green" + ext), cv2.IMREAD_GRAYSCALE)
			b = cv2.imread(os.path.join(img_folder, example_id + "_blue"  + ext), cv2.IMREAD_GRAYSCALE)
			img = np.stack([r, g, b], axis=-1)

		elif set(self.image_channels) == set("rgby"):
			r = cv2.imread(os.path.join(img_folder, example_id + "_red"    + ext), cv2.IMREAD_GRAYSCALE)
			g = cv2.imread(os.path.join(img_folder, example_id + "_green"  + ext), cv2.IMREAD_GRAYSCALE)
			b = cv2.imread(os.path.join(img_folder, example_id + "_blue"   + ext), cv2.IMREAD_GRAYSCALE)
			y = cv2.imread(os.path.join(img_folder, example_id + "_yellow" + ext), cv2.IMREAD_GRAYSCALE)
			img = np.stack([r, g, b, y], axis=-1)

		else:
			raise NotImplementedError("Image channel mode not yet supported.")

		if img is None:
			img = np.zeros((512, 512), dtype=np.uint8)
			img = np.stack([img]*len(self.image_channels), axis=-1).squeeze()
		if len(img.shape) == 1:
			img = np.zeros((512, 512), dtype=np.uint8)
			img = np.stack([img]*len(self.image_channels), axis=-1).squeeze()

		if self.resize is not None:
			img = self.resize(image=img)["image"]
		if self.transforms is not None:
			img = self.transforms(image=img)["image"]

		if example_source == "trainval": norm = self.primary_normalization
		elif example_source == "test": norm = self.test_normalization
		else: norm = self.external_normalization

		if self.test_transforms is not None:
			imgs = [norm(image=t(image=img)["image"])["image"] for t in self.test_transforms]
			imgs = np.asarray(imgs)
			if len(imgs.shape) == 3: imgs = imgs[:,:,:,np.newaxis]
			imgs = torch.from_numpy(imgs.transpose((0, 3, 1, 2)))

		img = norm(image=img)["image"]
		if len(img.shape) == 2: img = img[:,:,np.newaxis]
		img = torch.from_numpy(img.transpose((2, 0, 1)))

		if self.test_transforms is not None:
			img = torch.cat((img.unsqueeze(0), imgs), dim=0)

		if self.split in ["train", "val", "trainval"]:
			label = self.encode_label(label)
			return img, label
		else:
			return img, example_id

	def __len__(self):
		return len(self.data)

	def encode_label(self, lbl):
		out = np.zeros(self.n_classes)
		for i in lbl:
			out[int(i)] = 1
		return out

	def decode_label(self, lbl):
		return np.where(lbl.flatten() == 1)[0].tolist()

