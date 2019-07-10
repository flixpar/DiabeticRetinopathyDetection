import os
import glob
import tqdm
import random
import argparse

import cv2
import numpy as np

def main(args):
	
	fn_list  = glob.glob(os.path.join(args.path, "*.png"))
	fn_list += glob.glob(os.path.join(args.path, "*.jpg"))
	fn_list += glob.glob(os.path.join(args.path, "*.jpeg"))

	N = len(fn_list)

	if args.n > 0 and args.n < N:
		fn_list = random.sample(fn_list, args.n)

	pixel_means = []
	pixel_stds  = []

	for fn in tqdm.tqdm(fn_list):
		img = cv2.imread(fn)
		pixel_means.append(img.mean(axis=(0,1)))
		pixel_stds.append(img.std(axis=(0,1)))

	pixel_mean = np.mean(pixel_means, axis=0) / 255.0
	pixel_std  = np.std(pixel_stds, axis=0) / 255.0

	print("N =", N)
	if args.n > 0 and args.n < N: print("n =", args.n)

	print("pixel mean:", pixel_mean)
	print("pixel std: ", pixel_std)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Compute dataset statistics.")
	parser.add_argument("path", type=str, help="path to images")
	parser.add_argument("--n", type=int, default=-1, help="number of images to sample")
	args = parser.parse_args()
	main(args)
