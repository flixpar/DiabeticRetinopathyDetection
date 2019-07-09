import os
import glob
import random
import numpy as np
import pandas as pd
import torch

import loaders.base

class BlindnessDataset(loaders.base.BaseDataset):

	def __init__(self, *args, **kwargs):
		loaders.base.BaseDataset.__init__(self, *args, **kwargs)
		self.base_path = self.args.blindness_datapath
		self.extension = "png"

		if self.split == "train":
			self.data = pd.read_csv(os.path.join(self.args.blindness_datapath, "ann/train.csv")).values
			self.img_folder = "train"

		elif self.split == "val":
			self.data = pd.read_csv(os.path.join(self.args.blindness_datapath, "ann/val.csv")).values
			self.img_folder = "train"

		elif self.split == "trainval":
			d1 = pd.read_csv(os.path.join(self.args.blindness_datapath, "ann/train.csv"))
			d2 = pd.read_csv(os.path.join(self.args.blindness_datapath, "ann/val.csv"))
			self.data = pd.concat([d1, d2]).values
			self.img_folder = "train"

		elif self.split == "test":
			imgs = glob.glob(os.path.join(self.args.blindness_datapath, "test", "*.png"))
			ids = [os.path.basename(fn).replace(".png", "") for fn in imgs]
			self.data = sorted(ids)
			self.img_folder = "test"

		else:
			raise ValueError("Invalid dataset split.")

		self.setup()
