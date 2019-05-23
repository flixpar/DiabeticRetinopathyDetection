import numpy as np
import pandas as pd
import random
import tqdm

def main():

	df = pd.read_csv("../data/ann.csv")
	data = df.values
	N, _ = data.shape

	train_size, test_size, val_size = 0.7, 0.15, 0.15

	person_groups = [np.argwhere(data[:,0] == p).flatten().tolist() for p in range(data[:,0].max()+3)]
	person_groups = [g for g in person_groups if g]

	min_dist = np.inf
	z0 = np.mean(data[:, 4])

	best_train_ind = None
	best_test_ind  = None
	best_val_ind   = None

	for _ in tqdm.tqdm(range(100000)):

		k1 = random.randint(1, len(person_groups)-3)
		k2 = random.randint(1, len(person_groups)-1-k1-1)
		train_group_ind = random.sample(range(len(person_groups)), k1)
		remaining = list(set(range(len(person_groups))) - set(train_group_ind))
		test_group_ind = random.sample(remaining, k2)
		val_group_ind = list((set(range(len(person_groups))) - set(train_group_ind)) - set(test_group_ind))

		train_group_ind = sorted(train_group_ind)
		test_group_ind  = sorted(test_group_ind)
		val_group_ind   = sorted(val_group_ind)

		train_split_ind = [i for g in train_group_ind for i in person_groups[g]]
		test_split_ind  = [i for g in test_group_ind  for i in person_groups[g]]
		val_split_ind   = [i for g in val_group_ind   for i in person_groups[g]]

		r = len(train_split_ind) / N
		d1 = abs(r - train_size)

		r = len(test_split_ind) / N
		d2 = abs(r - test_size)

		r = len(val_split_ind) / N
		d3 = abs(r - val_size)

		z = np.mean(data[train_split_ind, 4])
		d4 = abs(z - z0)

		z = np.mean(data[test_split_ind, 4])
		d5 = abs(z - z0)

		z = np.mean(data[val_split_ind, 4])
		d6 = abs(z - z0)

		r1 = len(test_split_ind) / N
		r2 = len(val_split_ind) / N
		d7 = abs(r1 - r2)

		d = 2*d1 + d2 + d3 + d4 + d5 + d6 + 5*d7

		if d < min_dist:
			min_dist = d
			best_train_ind = train_split_ind
			best_test_ind  = test_split_ind
			best_val_ind   = val_split_ind
			s = f"{min_dist:.4f}: {d1:.3f} {d2:.3f} {d3:.3f} | {d7:.3f} | {d4:.3f} {d5:.3f} {d6:.3f}"
			tqdm.tqdm.write(s)

	print(best_train_ind)
	print(best_test_ind)
	print(best_val_ind)
	print(min_dist)

	train_df = df.iloc[best_train_ind]
	test_df  = df.iloc[best_test_ind]
	val_df   = df.iloc[best_val_ind]

	train_df.to_csv("../data/train_ann.csv", index=False)
	test_df.to_csv("../data/test_ann.csv", index=False)
	val_df.to_csv("../data/val_ann.csv", index=False)

if __name__ == "__main__":
	main()
