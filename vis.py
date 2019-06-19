import os
import tqdm
import random
import numpy as np
import torch
from util.misc import get_dataset_class

import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore")

from args import Args
args = Args()

def main():

	Dataset = get_dataset_class(args)
	dataset = Dataset(split=args.train_split, args=args, transforms=args.train_augmentation, debug=args.debug)

	transform = True
	num = 3

	for _ in range(num):
		img = dataset[random.randint(0, len(dataset)-1)][0]
		img = img.numpy()
		img = img.transpose(1, 2, 0)
		if transform:
			img = img * dataset.img_std
			img = img + dataset.img_mean
			img = img * 255.0
			img = img.astype(np.uint8)
		else:
			img = img - img.min()
			img = img / img.max()
		plt.imshow(img)
		plt.show()

if __name__ == "__main__":
	main()
