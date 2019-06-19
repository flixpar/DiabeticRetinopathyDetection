import torch
from torch import nn
import torch.optim.lr_scheduler
from torch.utils.data import WeightedRandomSampler

from loaders.loader import RetinaImageDataset
from loaders.kaggle_loader import KaggleRetinaDataset
from models.classifier import Classifier
from models.loss import FocalLoss, FBetaLoss

import numpy as np
from sklearn import metrics
from collections import OrderedDict
import os

def get_dataset_class(args):
	if args.pretraining: return KaggleRetinaDataset
	else: return RetinaImageDataset

def get_model(args):
	model = Classifier(arch=args.arch, ckpt=args.checkpoint)

	if args.pretrained:

		if args.pretrain_info is None or len(args.pretrain_info) != 2: raise ValueError("Invalid pretraining info.")
		save_folder, save_id = args.pretrain_info
		if isinstance(save_id, int):
			save_path = os.path.join("./saves", save_folder, f"save_{save_id:03d}.pth")
		else:
			save_path = os.path.join("./saves", save_folder, f"save_{save_id}.pth")

		state_dict = torch.load(save_path)
		if "module." in list(state_dict.keys())[0]:
			temp_state = OrderedDict()
			for k, v in state_dict.items():
				temp_state[k.split("module.")[-1]] = v
			state_dict = temp_state
		model.load_state_dict(state_dict)

	return model

def get_loss(args):

	if args.loss == "bce":
		loss_func = nn.BCEWithLogitsLoss()
	elif args.loss == "focal":
		if "focal_gamma" in args.loss_params:
			loss_func = FocalLoss(gamma=args.loss_params["focal_gamma"])
		else:
			loss_func = FocalLoss()
	elif args.loss == "fbeta":
		if "fbeta" in args.loss_params:
			loss_func = FBetaLoss(beta=args.loss_params["fbeta"], soft=True)
		else:
			loss_func = FBetaLoss(soft=True)
	elif args.loss == "softmargin":
		loss_func = nn.SoftMarginLoss()
	else:
		raise ValueError(f"Invalid loss function specifier: {args.loss}")

	return loss_func

def get_train_sampler(args, dataset):
	if args.example_weighting:
		return WeightedRandomSampler(weights=dataset.example_weights, num_samples=len(dataset))
	else:
		return None

def get_scheduler(args, optimizer):
	params = args.lr_schedule_params
	if args.lr_schedule == "poly":
		gamma = params["gamma"] if "gamma" in params else 0.9
		max_iter = args.epochs
		decay_iter = 1
		return PolynomialLR(optimizer, max_iter, decay_iter, gamma)
	elif args.lr_schedule == "exp":
		gamma = params["gamma"] if "gamma" in params else 0.95
		return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
	elif args.lr_schedule == "step":
		step_size = params["step_size"] if "step_size" in params else 5
		gamma = params["gamma"] if "gamma" in params else 0.5
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
	elif args.lr_schedule == "multistep":
		milestones = params["milestones"] if "milestones" in params else list(range(10, args.epochs, 10))
		gamma = params["gamma"] if "gamma" in params else 0.2
		return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
	elif args.lr_schedule == "cosine":
		T_max = params["period"] // 2 if "period" in params else 10
		max_decay = params["max_decay"] if "max_decay" in params else 50
		eta_min = args.initial_lr / max_decay
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
	else:
		return ConstantLR(optimizer)

class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, max_iter, decay_iter=1, gamma=0.9, last_epoch=-1):
		self.decay_iter = decay_iter
		self.max_iter = max_iter
		self.gamma = gamma
		super(PolynomialLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
			return [base_lr for base_lr in self.base_lrs]
		else:
			factor = (1 - (self.last_epoch / self.max_iter)) ** self.gamma
			return [base_lr * factor for base_lr in self.base_lrs]

class ConstantLR(torch.optim.lr_scheduler._LRScheduler):
	def __init__(self, optimizer, last_epoch=-1):
		super(ConstantLR, self).__init__(optimizer, last_epoch)

	def get_lr(self):
		return [base_lr for base_lr in self.base_lrs]

def sensitivity_specificity(y_true, y_pred):

	tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

	# Sensitivity, hit rate, recall, or true positive rate
	tpr = tp/(tp+fn)

	# Specificity or true negative rate
	tnr = tn/(tn+fp)

	# Precision or positive predictive value
	ppv = tp/(tp+fp)

	# Negative predictive value
	npv = tn/(tn+fn)

	# Fall out or false positive rate
	fpr = fp/(fp+tn)

	# False negative rate
	fnr = fn/(tp+fn)

	# False discovery rate
	fdr = fp/(tp+fp)

	# Overall accuracy
	acc = (tp+tn)/(tp+fp+fn+tn)

	return tpr, tnr

def confusion_matrix(y_true, y_pred):
	tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
	return tn, fp, fn, tp
