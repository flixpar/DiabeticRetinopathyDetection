import os
import random
import numpy as np
import pandas as pd
import torch

import loaders.base

class DRDataset(loaders.base.BaseDataset):

	def __init__(self, *args, **kwargs):
		loaders.base.BaseDataset.__init__(self, *args, **kwargs)
		self.base_path = self.args.dr_datapath
		self.extension = "jpeg"

		self.img_mean = [0.16147814, 0.22585096, 0.32250461]
		self.img_std  = [0.05810949, 0.05864657, 0.07206357]

		if self.split == "train":
			self.data = pd.read_csv(os.path.join(self.args.dr_datapath, "ann/train.csv")).values
			self.img_folder = "train"

		elif self.split == "val":
			self.data = pd.read_csv(os.path.join(self.args.dr_datapath, "ann/val.csv")).values
			self.img_folder = "train"

		elif self.split == "trainval":
			d1 = pd.read_csv(os.path.join(self.args.dr_datapath, "ann/train.csv"))
			d2 = pd.read_csv(os.path.join(self.args.dr_datapath, "ann/val.csv"))
			self.data = pd.concat([d1, d2]).values
			self.img_folder = "train"

		elif self.split == "test":
			raise ValueError("Test set not implemented yet.")
			self.img_folder = "test"

		else:
			raise ValueError("Invalid dataset split.")

		self.setup()
