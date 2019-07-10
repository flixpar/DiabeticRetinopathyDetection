import os
import tqdm
import numpy as np
from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F

from util.logger import Logger
from util.misc import (
	get_dataset_class, get_model, get_loss,
	get_train_sampler, get_scheduler,
	check_memory_use
)

import warnings
warnings.simplefilter("ignore")

from args import Args
args = Args()

primary_device = torch.device(f"cuda:{args.device_ids[0]}")

def main():

	# datasets
	Dataset = get_dataset_class(args)
	train_dataset = Dataset(split=args.train_split, args=args, transforms=args.train_augmentation, debug=args.debug)
	tval_dataset = Dataset(split=args.train_split, args=args, test_transforms=args.test_augmentation, debug=args.debug, n_samples=args.n_tval_samples)
	vval_dataset = Dataset(split=args.val_split, args=args, test_transforms=args.test_augmentation, debug=args.debug, n_samples=args.n_vval_samples)

	# sampling
	train_sampler = get_train_sampler(args, train_dataset)
	shuffle = (train_sampler is None)

	# dataloaders
	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
	tval_loader = torch.utils.data.DataLoader(tval_dataset, shuffle=False, batch_size=1, num_workers=args.workers, pin_memory=True)
	vval_loader = torch.utils.data.DataLoader(vval_dataset, shuffle=False, batch_size=1, num_workers=args.workers, pin_memory=True)

	# model
	model = get_model(args)
	if len(args.device_ids) > 1:
		model = nn.DataParallel(model.cuda(), device_ids=args.device_ids)
	model.to(primary_device)
	check_memory_use(args, model, primary_device)

	# training
	loss_func = get_loss(args).to(primary_device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
	scheduler = get_scheduler(args, optimizer)

	logger = Logger(args=args, enabled=args.logging_enabled)
	max_score = 0

	for epoch in range(1, args.epochs+1):
		logger.print(f"Epoch {epoch}")
		train(model, train_loader, loss_func, optimizer, logger)
		evaluate(model, tval_loader, loss_func, logger, "tval")
		score = evaluate(model, vval_loader, loss_func, logger, "vval")
		if args.save_freq == "best" and score > max_score:
			logger.save_model(model)
		elif args.save_freq is not None and epoch % args.save_freq == 0:
			logger.save_model(model)
		if score > max_score:
			max_score = score
		scheduler.step()
		logger.complete_epoch()
		logger.print("\n")

	logger.save()
	logger.save_model(model)


def train(model, train_loader, loss_func, optimizer, logger):
	model.train()
	logger.train()

	losses = []
	for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):

		images = images.to(primary_device, dtype=torch.float32, non_blocking=True)
		labels = labels.to(primary_device, dtype=torch.long, non_blocking=True)

		outputs = model(images).squeeze(-1)
		loss = loss_func(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		logger.log_loss(loss.item())
		if i % (len(train_loader)//args.log_freq) == 0:
			logger.print_loss()

	logger.print()

def evaluate(model, loader, loss_func, logger, splitname="val"):
	model.eval()
	logger.eval(splitname)

	losses = []
	preds = []
	targets = []

	logger.print(f"Eval {logger.epoch} - {splitname}")
	with torch.no_grad():
		for i, (images, labels) in tqdm.tqdm(enumerate(loader), total=len(loader)):

			images = images.to(primary_device, dtype=torch.float32, non_blocking=True).squeeze(0)
			labels = labels.to(primary_device, dtype=torch.long, non_blocking=True).repeat((images.shape[0]))

			outputs = model(images)
			loss = loss_func(outputs, labels).item()

			pred = torch.softmax(outputs, dim=-1).mean(dim=0).argmax(dim=-1)
			pred = pred.cpu().numpy()

			labels = labels[0].cpu().numpy().astype(np.int).squeeze()

			losses.append(loss)
			preds.append(pred)
			targets.append(labels)

	targets = np.array(targets).squeeze()
	preds = np.array(preds).squeeze()

	acc = metrics.accuracy_score(targets, preds)
	f1 = metrics.f1_score(targets, preds, average="macro")
	precision = metrics.precision_score(targets, preds, average="micro")
	recall = metrics.recall_score(targets, preds, average="micro")
	kappa = metrics.cohen_kappa_score(targets, preds, weights="quadratic")
	loss = np.mean(losses)

	logger.print(f"Loss:        {loss:.4f}")
	logger.print(f"Accuracy:    {acc:.4f}")
	logger.print(f"F1:          {f1:.4f}")
	logger.print(f"Kappa:       {kappa:.4f}")
	logger.print(f"Precision:   {precision:.4f}")
	logger.print(f"Recall:      {recall:.4f}")

	logger.log_scores({
		"loss": loss,
		"acc": acc,
		"f1": f1,
		"kappa": kappa,
		"precision": precision,
		"recall": recall,
	})
	return kappa

if __name__ == "__main__":
	main()
