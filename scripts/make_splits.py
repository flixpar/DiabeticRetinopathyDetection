import numpy as np
import pandas as pd
import random
import tqdm

def main():

	df = pd.read_csv("../data/ann.csv")
	data = df.values
	N, _ = data.shape

	person_groups = [np.argwhere(data[:,0] == p).flatten().tolist() for p in range(data[:,0].max()+3)]
	person_groups = [g for g in person_groups if g]

	min_dist = np.inf
	z0 = np.mean(data[:, 4])

	best_train_ind = None
	best_test_ind  = None

	for _ in tqdm.tqdm(range(20000)):

		k = random.randint(1, len(person_groups)-2)
		train_group_ind = random.sample(range(len(person_groups)), k)
		test_group_ind  = list(set(range(len(person_groups))) - set(train_group_ind))

		train_group_ind = sorted(train_group_ind)
		test_group_ind  = sorted(test_group_ind)

		train_split_ind = [i for g in train_group_ind for i in person_groups[g]]
		test_split_ind  = [i for g in test_group_ind  for i in person_groups[g]]

		r = len(train_split_ind) / N
		d1 = abs(r - 0.8)

		z = np.mean(data[train_split_ind, 4])
		d2 = abs(z - z0)

		d = d1 + d2

		if d < min_dist:
			min_dist = d
			best_train_ind = train_split_ind
			best_test_ind  = test_split_ind
			tqdm.tqdm.write(str(min_dist) + "\t" + str(d1) + "\t" + str(d2))

	print(best_train_ind)
	print(best_test_ind)
	print(min_dist)

	train_df = df.iloc[best_train_ind]
	test_df  = df.iloc[best_test_ind]

	train_df.to_csv("../data/train_ann.csv", index=False)
	test_df.to_csv("../data/test_ann.csv", index=False)

if __name__ == "__main__":
	main()