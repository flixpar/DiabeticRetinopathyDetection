import os
import sys
import tqdm
import glob
import numpy as np
import importlib.util
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import albumentations as tfms

from sklearn import metrics

from loaders.loader import RetinaImageDataset
from models.classifier import Classifier

from util.misc import get_model, sensitivity_specificity

import warnings
from sklearn.exceptions import UndefinedMetricWarning

primary_device = torch.device("cuda:0")

def main():

	if not len(sys.argv) == 3:
		raise ValueError("Not enough arguments")

	folder_name = sys.argv[1]
	folder_path = os.path.join("./saves", folder_name)
	if not os.path.exists(folder_path):
		raise ValueError("No matching save folder: {}".format(folder_path))

	save_id = sys.argv[2]
	if save_id.isdigit() and os.path.exists(os.path.join(folder_path, "save_{:03d}.pth".format(int(save_id)))):
		save_path = os.path.join(folder_path, "save_{:03d}.pth".format(int(save_id)))
	elif os.path.exists(os.path.join(folder_path, "save_{}.pth".format(save_id))):
		save_path = os.path.join(folder_path, "save_{}.pth".format(save_id))
	else:
		raise Exception("Specified save not found: {}".format(save_id))

	args_module_spec = importlib.util.spec_from_file_location("args", os.path.join(folder_path, "args.py"))
	args_module = importlib.util.module_from_spec(args_module_spec)
	args_module_spec.loader.exec_module(args_module)
	args = args_module.Args()

	test_dataset = RetinaImageDataset(split="test", args=args, test_transforms=args.test_augmentation, debug=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size*2, num_workers=args.workers, pin_memory=True)

	model = get_model(args)
	state_dict = torch.load(save_path)
	if "module." in list(state_dict.keys())[0]:
		temp_state = OrderedDict()
		for k, v in state_dict.items():
			temp_state[k.split("module.")[-1]] = v
		state_dict = temp_state
	model.load_state_dict(state_dict)
	model.cuda()

	model = nn.DataParallel(model, device_ids=args.device_ids)
	model.to(primary_device)

	print("Test")
	evaluate(model, test_loader)

def evaluate(model, loader, threshold=0.5):
	model.eval()

	preds = []
	targets = []

	with torch.no_grad():
		for i, (images, labels) in tqdm.tqdm(enumerate(loader), total=len(loader)):

			if len(images.shape) == 5:
				n_examples, n_copies, c, w, h = images.shape
				images = images.view(n_examples*n_copies, c, w, h)
			else:
				n_examples, _, _, _ = images.shape
				n_copies = 1

			images = images.to(primary_device, dtype=torch.float32, non_blocking=True)

			_, output = model(images)

			if n_copies != 1:
				output = torch.chunk(output, chunks=n_examples, dim=0)
				output = torch.stack(output, dim=0).squeeze(-1)
				output = (0.5 * output[:, 0]) + (0.5 * output[:, 1:].mean(dim=1))

			pred = output > threshold
			pred = pred.cpu().numpy().astype(np.int).squeeze().tolist()
			if not isinstance(pred, list): pred = [pred]
			preds.extend(pred)

			labels = labels.cpu().numpy().astype(np.int).squeeze().tolist()
			if not isinstance(labels, list): labels = [labels]
			targets.extend(labels)

	targets = np.array(targets).squeeze()
	preds = np.array(preds).squeeze()

	acc = metrics.accuracy_score(targets, preds)
	f1 = metrics.f1_score(targets, preds)

	sensitivity, specificity = sensitivity_specificity(targets, preds)

	print("Evaluation Results")
	print("Accuracy:", acc)
	print("F1:", f1)
	print("Sensitivity:", sensitivity)
	print("Specificity:", specificity)

if __name__ == "__main__":
	main()
