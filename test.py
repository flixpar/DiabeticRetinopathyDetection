import os
import tqdm
import argparse
import numpy as np
import importlib.util
from collections import OrderedDict
import datetime
import itertools

import torch
from torch import nn
import torch.nn.functional as F

from sklearn import metrics

from loaders.blindness_dataset import BlindnessDataset
from models.classifier import Classifier

from util.misc import get_model

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-paper") # alt: seaborn-darkgrid
import matplotlib2tikz

import warnings
warnings.simplefilter("ignore")

primary_device = torch.device("cuda:0")

def main(cfg):

	folder_path = os.path.join("./saves", cfg.folder_name)
	if not os.path.exists(folder_path):
		raise ValueError(f"No matching save folder: {folder_path}")

	if os.path.exists(os.path.join(folder_path, f"save_{cfg.save_id:03d}.pth")):
		save_path = os.path.join(folder_path, f"save_{cfg.save_id:03d}.pth")
	else:
		raise Exception(f"Specified save not found: {cfg.save_id:03d}")

	args_module_spec = importlib.util.spec_from_file_location("args", os.path.join(folder_path, "args.py"))
	args_module = importlib.util.module_from_spec(args_module_spec)
	args_module_spec.loader.exec_module(args_module)
	args = args_module.Args()

	test_dataset = BlindnessDataset(split=cfg.split, args=args, test_transforms=args.test_augmentation, debug=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=args.workers, pin_memory=True)

	model = get_model(args)
	state_dict = torch.load(save_path)
	if "module." == list(state_dict.keys())[0][:7]:
		temp_state = OrderedDict()
		for k, v in state_dict.items():
			temp_state[k.replace("module.", "", 1)] = v
		state_dict = temp_state
	model.load_state_dict(state_dict)
	model.to(primary_device)

	print("Test")
	evaluate(model, test_loader, folder_path, cfg)

def evaluate(model, loader, folder_path, cfg):
	model.eval()

	preds = []
	targets = []

	with torch.no_grad():
		for i, (images, labels) in tqdm.tqdm(enumerate(loader), total=len(loader)):

			images = images.to(primary_device, dtype=torch.float32, non_blocking=True).squeeze(0)
			labels = labels.to(primary_device, dtype=torch.long, non_blocking=True).repeat((images.shape[0]))

			outputs = model(images)

			pred = torch.softmax(outputs, dim=-1).mean(dim=0).argmax(dim=-1)
			pred = pred.cpu().numpy()

			labels = labels[0].cpu().numpy().astype(np.int).squeeze()

			preds.append(pred)
			targets.append(labels)

	targets = np.array(targets).squeeze()
	preds = np.array(preds).squeeze()

	dt = datetime.datetime.now().strftime("%m%d%H%M")
	test_folder_path = os.path.join(folder_path, f"test_{dt}")
	if not os.path.isdir(test_folder_path): os.makedirs(test_folder_path)

	acc = metrics.accuracy_score(targets, preds)
	f1 = metrics.f1_score(targets, preds, average="macro")
	precision = metrics.precision_score(targets, preds, average="micro")
	recall = metrics.recall_score(targets, preds, average="micro")
	kappa = metrics.cohen_kappa_score(targets, preds, weights="quadratic")

	print("Test Results")
	print(f"Accuracy:    {acc:.4f}")
	print(f"F1:          {f1:.4f}")
	print(f"Kappa:       {kappa:.4f}")
	print(f"Precision:   {precision:.4f}")
	print(f"Recall:      {recall:.4f}")

	with open(os.path.join(test_folder_path, "results.txt"), "w") as f:
		f.write("Test Results\n")
		f.write(f"Accuracy:    {acc:.4f}\n")
		f.write(f"F1:          {f1:.4f}\n")
		f.write(f"Kappa:       {kappa:.4f}\n")
		f.write(f"Precision:   {precision:.4f}\n")
		f.write(f"Recall:      {recall:.4f}\n")

	with open(os.path.join(test_folder_path, "cfg.txt"), "w") as f:
		f.write(f"folder:    {cfg.folder_name}\n")
		f.write(f"epoch:     {cfg.save_id}\n")
		f.write(f"dataset:   {cfg.split}\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Test for Retina Project")
	parser.add_argument("folder_name", type=str,   help="Name of save folder")
	parser.add_argument("save_id",     type=int,   help="Save epoch")
	parser.add_argument("--split",     type=str,   required=False, default="val", help="Dataset partition")
	args = parser.parse_args()
	main(args)
