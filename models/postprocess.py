import torch
from torch import nn

import numpy as np
from sklearn import metrics

import pystruct
from pystruct.learners import NSlackSSVM
from pystruct.models import MultiLabelClf

def postprocess(args, preds, targets=None, threshold=None):

	preds = np.asarray(preds)
	targets = np.asarray(targets) if targets is not None else None

	if len(preds.shape) != 2: preds = preds.squeeze()
	if targets is not None and len(targets.shape) != 2: targets = targets.squeeze()

	if threshold is not None:
		if isinstance(threshold, float):
			threshold = np.full((1, preds.shape[1]), threshold)
		elif isinstance(threshold, np.ndarray):
			threshold = threshold.reshape(1, -1)
		elif isinstance(threshold, list):
			threshold = np.asarray(threshold).reshape(1, -1)
		else:
			threshold = np.full((1, preds.shape[1]), 0.5)
	elif targets is not None:
		threshold = compute_threshold(args, preds, targets)
	else:
		threshold = np.full((1, preds.shape[1]), 0.5)

	if "max3" in args.postprocessing:
		example_mask = (preds > threshold).astype(np.int).sum(axis=1) > 3
		class_mask = np.argsort(preds, axis=1)[:, :-3]
		preds[class_mask][example_mask] = 0

	elif "max4" in args.postprocessing:
		example_mask = (preds > threshold).astype(np.int).sum(axis=1) > 4
		class_mask = np.argsort(preds, axis=1)[:, :-4]
		preds[class_mask][example_mask] = 0

	if "9+10" in args.postprocessing:
		mask_9_10 = np.mean(preds[:, 9:11], axis=1)
		mask_9_10 = mask_9_10 > np.mean(threshold[:, 9:11])
		preds[mask_9_10, 9:11]  = 1

	if "min1" in args.postprocessing:
		mask = np.all(preds <= threshold, axis=1)
		tops = np.argmax(preds, axis=1)
		modified = preds.copy()
		modified[np.arange(len(preds)), tops] = 1
		preds[mask] = modified[mask]

	preds = (preds > threshold).astype(np.int)
	return preds

def optimize_uniform_threshold(p_pred, y_true):
	searchspace = np.linspace(0, 1, num=100)
	scores = np.asarray([metrics.f1_score(y_true, (p_pred>t), average="macro") for t in searchspace])
	best = searchspace[scores.argmax()]
	return best

def optimize_perclass_threshold(p_pred, y_true):
	searchspace = np.linspace(0, 1, num=100)
	scores = np.asarray([metrics.f1_score(y_true, (p_pred>t), average=None) for t in searchspace])
	best = searchspace[scores.argmax(axis=0)]
	return best

def compute_threshold(args, preds, targets):
	preds, targets = np.asarray(preds), np.asarray(targets)
	if len(preds.shape) != 2: preds = preds.squeeze()
	if len(targets.shape) != 2: targets = targets.squeeze()
	if "uniform_thresh" in args.postprocessing:
		threshold = optimize_uniform_threshold(preds, targets)
		threshold = np.full((1, preds.shape[1]), threshold)
	elif "perclass_thresh" in args.postprocessing:
		threshold = optimize_perclass_threshold(preds, targets)
		threshold = threshold.reshape(1, -1)
	else:
		threshold = np.full((1, preds.shape[1]), 0.5)
	return threshold

def crf_postprocess(X_train, y_train, X_test, train_examples=2000):
	clf = NSlackSSVM(MultiLabelClf(), verbose=1, n_jobs=-1, show_loss_every=1)
	clf.fit(X_train, y_train)
	pred = clf.predict(X_test)
	pred = np.array(pred)
	return pred
