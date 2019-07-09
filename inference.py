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

from loaders.blindness_dataset import BlindnessDataset
from models.classifier import Classifier
from util.misc import get_model

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

	print("Inference")
	inference(model, test_loader, folder_path, cfg)

def inference(model, loader, folder_path, cfg):
	model.eval()

	preds = []
	with torch.no_grad():
		for i, (images, img_id) in tqdm.tqdm(enumerate(loader), total=len(loader)):

			images = images.to(primary_device, dtype=torch.float32, non_blocking=True).squeeze(0)
			outputs = model(images)

			pred = torch.softmax(outputs, dim=-1).mean(dim=0).argmax(dim=-1)
			pred = pred.cpu().numpy()
			preds.append((img_id[0], pred))

	dt = datetime.datetime.now().strftime("%m%d%H%M")
	inference_folder_path = os.path.join(folder_path, f"inference_{dt}")
	if not os.path.isdir(inference_folder_path): os.makedirs(inference_folder_path)

	with open(os.path.join(inference_folder_path, "pred.csv"), "w") as f:
		f.write("id_code,diagnosis\n")
		for img_id, pred in preds:
			f.write(f"{img_id},{pred:d}\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Inference for Retina Project")
	parser.add_argument("folder_name", type=str,   help="Name of save folder")
	parser.add_argument("save_id",     type=int,   help="Save epoch")
	parser.add_argument("--split",     type=str,   required=False, default="test", help="Dataset partition")
	args = parser.parse_args()
	main(args)
