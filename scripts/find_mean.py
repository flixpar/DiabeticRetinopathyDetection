import os
import sys
import glob
import random

import cv2
import numpy as np

def main():

	folder = sys.argv[1]

	files = glob.glob(os.path.join(folder, "*.png"))
	ids = ['_'.join(fn.split('_')[:-1]) for fn in files]
	ids = list(set(ids))

	ids = random.sample(ids, 1000)

	for color in ["red", "green", "blue", "yellow"]:
		imgs = np.stack([cv2.imread(i + "_{}.png".format(color), 0) for i in ids], axis=0)
		mean = imgs.mean(axis=(1,2)).mean() / 255
		std  = imgs.std(axis=(1,2)).mean() / 255
		print("{}: {}, {}".format(color, mean, std))

if __name__ == "__main__":
	main()
