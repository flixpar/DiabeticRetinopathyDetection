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
from statsmodels.stats.proportion import proportion_confint

from loaders.loader import RetinaImageDataset
from models.classifier import Classifier

from util.misc import get_model, sensitivity_specificity, confusion_matrix

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

import warnings
warnings.simplefilter("ignore")

primary_device = torch.device("cuda:0")

def main(cfg):

	folder_path = os.path.join("./saves", cfg.folder_name)
	if not os.path.exists(folder_path):
		raise ValueError(f"No matching save folder: {folder_path}")

	if cfg.save_id.isdigit() and os.path.exists(os.path.join(folder_path, f"save_{int(cfg.save_id):03d}.pth")):
		save_path = os.path.join(folder_path, f"save_{int(cfg.save_id):03d}.pth")
	elif os.path.exists(os.path.join(folder_path, f"save_{cfg.save_id}.pth")):
		save_path = os.path.join(folder_path, f"save_{cfg.save_id}.pth")
	else:
		raise Exception(f"Specified save not found: {cfg.save_id}")

	args_module_spec = importlib.util.spec_from_file_location("args", os.path.join(folder_path, "args.py"))
	args_module = importlib.util.module_from_spec(args_module_spec)
	args_module_spec.loader.exec_module(args_module)
	args = args_module.Args()

	test_dataset = RetinaImageDataset(split=cfg.split, args=args, test_transforms=args.test_augmentation, debug=False)
	test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size*2, num_workers=args.workers, pin_memory=True)

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

	threshold = cfg.thresh

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

			pred = output.cpu().numpy().squeeze().tolist()
			if not isinstance(pred, list): pred = [pred]
			preds.extend(pred)

			labels = labels.cpu().numpy().astype(np.int).squeeze().tolist()
			if not isinstance(labels, list): labels = [labels]
			targets.extend(labels)

	targets = np.array(targets).squeeze()
	preds = np.array(preds).squeeze()

	dt = datetime.datetime.now().strftime("%m%d%H%M")
	test_folder_path = os.path.join(folder_path, f"test_{dt}")
	if not os.path.isdir(test_folder_path): os.makedirs(test_folder_path)
	create_plots(targets, preds, test_folder_path)

	auc_score = metrics.roc_auc_score(targets, preds)

	print("Sensitivity vs Specificity")
	sens_spec_list, opt_threshold = sensitivity_specificity_points(targets, preds)
	with open(os.path.join(test_folder_path, "sensitivity_specificity.txt"), "w") as f:
		for t, sens, spec in sens_spec_list:
			row_str = f"{t:.3f}: sens={sens:.4f}, spec={spec:.4f}"
			print(row_str)
			f.write(row_str + "\n")

	if threshold == -1.0:
		threshold = opt_threshold

	preds = preds > threshold
	preds = preds.astype(np.int)

	acc = metrics.accuracy_score(targets, preds)
	f1 = metrics.f1_score(targets, preds)

	sensitivity, specificity = sensitivity_specificity(targets, preds)

	sens_ci, spec_ci, acc_ci = confidence_intervals(targets, preds)

	confusion(targets, preds, test_folder_path)

	print("Test Results")
	print(f"Accuracy:    {acc:.4f} ({acc_ci[0]:.4f}, {acc_ci[1]:.4f})")
	print(f"F1:          {f1:.4f}")
	print(f"AUC:         {auc_score:.4f}")
	print(f"Sensitivity: {sensitivity:.4f} ({sens_ci[0]:.4f}, {sens_ci[1]:.4f})")
	print(f"Specificity: {specificity:.4f} ({spec_ci[0]:.4f}, {spec_ci[1]:.4f})")
	print(f"Threshold:   {threshold:.4f}")

	with open(os.path.join(test_folder_path, "results.txt"), "w") as f:
		f.write("Test Results\n")
		f.write(f"Accuracy:    {acc:.4f} ({acc_ci[0]:.4f}, {acc_ci[1]:.4f})\n")
		f.write(f"F1:          {f1:.4f}\n")
		f.write(f"AUC:         {auc_score:.4f}\n")
		f.write(f"Sensitivity: {sensitivity:.4f} ({sens_ci[0]:.4f}, {sens_ci[1]:.4f})\n")
		f.write(f"Specificity: {specificity:.4f} ({spec_ci[0]:.4f}, {spec_ci[1]:.4f})\n")
		f.write(f"Threshold:   {threshold:.4f}\n")

	with open(os.path.join(test_folder_path, "cfg.txt"), "w") as f:
		f.write(f"folder:    {cfg.folder_name}\n")
		f.write(f"epoch:     {cfg.save_id}\n")
		f.write(f"dataset:   {cfg.split}\n")
		f.write(f"threshold: {threshold}\n")

def create_plots(targets, preds, folder_path):

	sensitivity_specificity_curve(targets, preds)
	plt.savefig(os.path.join(folder_path, "sensitivity_specificity.png"))
	plt.clf()
	plt.close()

	precision_recall_curve(targets, preds)
	plt.savefig(os.path.join(folder_path, "precision_recall.png"))
	plt.clf()
	plt.close()

	roc_curve(targets, preds)
	plt.savefig(os.path.join(folder_path, "roc.png"))
	plt.clf()
	plt.close()

def sensitivity_specificity_curve(targets, preds):

	thresholds = np.linspace(0.0, 1.0, 500)
	points = []

	for t in thresholds:
		p = preds > t
		sens, spec = sensitivity_specificity(targets, p)
		points.append((t, sens, spec))

	points = np.array(points)
	plt.plot(points[:,1], points[:,2])
	plt.title("Sensitivity vs Specificity")
	plt.xlabel("sensitivity")
	plt.ylabel("specificity")

def sensitivity_specificity_points(targets, preds):

	thresholds = np.linspace(0.0, 1.0, 500)
	points = []

	for t in thresholds:
		p = preds > t
		sens, spec = sensitivity_specificity(targets, p)
		points.append((t, sens, spec))

	sens_spec_list_a = []
	sens_spec_list_b = []

	prev_sens = -1
	for t, sens, spec in reversed(points):
		if sens != prev_sens:
			prev_sens = sens
			sens_spec_list_a.append((t, sens, spec))
	prev_spec = -1
	for t, sens, spec in reversed(sens_spec_list_a):
		if spec != prev_spec:
			prev_spec = spec
			sens_spec_list_b.append((t, sens, spec))
	sens_spec_list = sens_spec_list_b

	points = np.array(sens_spec_list)
	scores = 3*points[:,1] + points[:,2]
	opt = np.argmax(scores)
	opt_threshold = points[opt, 0]

	return sens_spec_list, opt_threshold

def precision_recall_curve(targets, preds):
	prec, rec, _ = metrics.precision_recall_curve(targets, preds)

	plt.plot(prec, rec)
	plt.title("Precision vs Recall")
	plt.xlabel("precision")
	plt.ylabel("recall")

def roc_curve(targets, preds):
	fpr, tpr, _ = metrics.roc_curve(targets, preds)

	plt.plot(fpr, tpr)
	plt.title("ROC Curve")

def confusion(targets, preds, folder_path):

	cfm = metrics.confusion_matrix(targets, preds, labels=[0,1])
	tn, fp, fn, tp = cfm.ravel()

	with open(os.path.join(folder_path, "cfm.txt"), "w") as f:
		f.write(f"tp: {tp}\n")
		f.write(f"fp: {fp}\n")
		f.write(f"fn: {fn}\n")
		f.write(f"tn: {tn}\n")

	plt.figure()

	cfm_norm = cfm.astype(np.float32) / cfm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cfm_norm, interpolation='nearest', cmap=plt.cm.Blues)

	plt.title("Confusion Matrix")
	plt.colorbar()
	plt.xticks(np.arange(2), range(2), rotation=45)
	plt.yticks(np.arange(2), range(2))

	thresh = cfm_norm.max() / 2.
	for i, j in itertools.product(range(cfm.shape[0]), range(cfm.shape[1])):
	    plt.text(j, i, cfm[i,j], size="x-small", horizontalalignment="center", color=("white" if cfm_norm[i,j]>thresh else "black"))

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.grid(b=None)

	fn = os.path.join(folder_path, "cfm.png")
	plt.savefig(fn, dpi=300)

	plt.clf()
	plt.close()

def confidence_intervals(targets, preds):
	cfm = metrics.confusion_matrix(targets, preds, labels=[0,1])
	tn, fp, fn, tp = cfm.ravel()
	sens_ci = proportion_confint(tp, tp+fn, alpha=0.05, method="beta")
	spec_ci = proportion_confint(tn, fp+tn, alpha=0.05, method="beta")
	acc_ci  = proportion_confint(tp+tn, tp+tn+fp+fn, alpha=0.05, method="beta")
	return sens_ci, spec_ci, acc_ci

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Test for Retina Project")
	parser.add_argument("folder_name", type=str,   help="Name of save folder")
	parser.add_argument("save_id",     type=str,   help="Name of save epoch")
	parser.add_argument("--split",     type=str,   required=False, default="test", help="Dataset partition")
	parser.add_argument("--thresh",    type=float, required=False, default=-1.0, help="Prediction threshold")
	args = parser.parse_args()
	main(args)
