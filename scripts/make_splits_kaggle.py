import numpy as np
import pandas as pd
import random
import tqdm
import os
import sys

def main():

	base_path = sys.argv[1]
	df = pd.read_csv(os.path.join(base_path, "trainLabels.csv"))
	df.loc[df["level"] > 0, "level"] = 1
	data = df.values
	N, _ = data.shape

	print(f"Creating split for kaggle dataset with {N} images.")

	train_ratio, val_ratio = 0.85, 0.15

	img_ids = data[:,0].tolist()
	person_ids = list(set(i.split('_')[0] for i in img_ids))
	person_ids_to_ind = {}
	for i, img_id in enumerate(img_ids):
		p_id = img_id.split('_')[0]
		if p_id in person_ids_to_ind: person_ids_to_ind[p_id].append(i)
		else: person_ids_to_ind[p_id] = [i]

	z0 = np.mean(data[:,1])
	min_score = np.inf
	best_train_ind = None

	for _ in tqdm.tqdm(range(2000)):

		train_person_ids = random.sample(person_ids, int(train_ratio*len(person_ids)))
		train_split_ind = [j for i in train_person_ids for j in person_ids_to_ind[i]]

		s = abs(z0 - np.mean(data[train_split_ind, 1]))

		if s < min_score:
			min_score = s
			best_train_ind = train_split_ind
			s = f"min score: {min_score:.7f}"
			tqdm.tqdm.write(s)

	best_train_ind = sorted(best_train_ind)
	best_val_ind = sorted(list(set(range(N)) - set(best_train_ind)))

	train_df = df.iloc[best_train_ind]
	val_df   = df.iloc[best_val_ind]

	train_df.to_csv(os.path.join(base_path, "train_ann.csv"), index=False)
	val_df.to_csv(os.path.join(base_path, "val_ann.csv"), index=False)

	print("Done!")

if __name__ == "__main__":
	main()
