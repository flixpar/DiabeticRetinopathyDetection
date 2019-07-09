import numpy as np
import pandas as pd
import random
import tqdm
import os
import sys
import argparse

def main(args):

	df = pd.read_csv(args.file)
	data = df.values
	labels = data[:,1].astype(np.int)
	N, _ = data.shape
	print(f"Creating splits for dataset with {N} images.")

	if args.train_ratio > 1: args.train_ratio /= 100
	train_ratio, val_ratio = args.train_ratio, 1-args.train_ratio
	n_train = int(train_ratio * N)

	true_dist = np.bincount(labels) / N

	best_score = np.inf
	best_train_ind = None

	for _ in tqdm.tqdm(range(args.iter)):

		ind = random.sample(range(N), n_train)
		dist = np.bincount(labels[ind]) / n_train

		score = np.linalg.norm(dist - true_dist, ord=2)

		if score < best_score:
			best_score = score
			best_train_ind = ind
			tqdm.tqdm.write(f"best score: {best_score:.7f}")

	best_train_ind = sorted(best_train_ind)
	best_val_ind = sorted(list(set(range(N)) - set(best_train_ind)))

	train_df = df.iloc[best_train_ind]
	val_df   = df.iloc[best_val_ind]

	train_df.to_csv("train.csv", index=False)
	val_df.to_csv("val.csv", index=False)

	print("Full: ", true_dist)
	print("Train:", np.bincount(labels[best_train_ind]) / n_train)

	print("Done!")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Make train/val splits.")
	parser.add_argument("file", type=str, help="path to input file")
	parser.add_argument("--iter", type=int, default=10000, help="number of random iterations")
	parser.add_argument("--train_ratio", type=float, default=0.85, help="percent of dataset in train split")
	args = parser.parse_args()
	main(args)
