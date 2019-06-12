import os
import tqdm
import numpy as np
from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F

from util.logger import Logger
from util.misc import get_dataset_class, get_model, get_loss, get_train_sampler, get_scheduler, sensitivity_specificity

import warnings
warnings.simplefilter("ignore")

from args import Args
args = Args()

primary_device = torch.device(f"cuda:{args.device_ids[0]}")

def main():

	# datasets

	Dataset = get_dataset_class(args)

	train_dataset = Dataset(split=args.train_split, args=args,
		transforms=args.train_augmentation, debug=args.debug)

	train_static_dataset = Dataset(split=args.train_split, args=args,
		test_transforms=args.test_augmentation, debug=args.debug,
		n_samples=args.n_train_eval_samples)

	val_dataset  = Dataset(split=args.val_split, args=args,
		test_transforms=args.test_augmentation, debug=args.debug,
		n_samples=args.n_val_samples)

	# sampling
	train_sampler = get_train_sampler(args, train_dataset)
	shuffle = (train_sampler is None)

	# dataloaders

	train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=shuffle, sampler=train_sampler,
		batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

	train_static_loader = torch.utils.data.DataLoader(train_static_dataset, shuffle=False,
		batch_size=1, num_workers=args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=1,
		num_workers=args.workers, pin_memory=True)

	# model
	model = get_model(args).cuda()
	model = nn.DataParallel(model, device_ids=args.device_ids)
	model.to(primary_device)

	# training
	loss_func = get_loss(args).to(primary_device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
	scheduler = get_scheduler(args, optimizer)

	logger = Logger(args=args)
	max_score = 0

	for epoch in range(1, args.epochs+1):
		logger.print(f"Epoch {epoch}")
		scheduler.step()
		train(model, train_loader, loss_func, optimizer, logger)
		evaluate(model, train_static_loader, loss_func, logger, epoch, "train")
		score = evaluate(model, val_loader, loss_func, logger, epoch, "val")
		logger.save()
		if score > max_score:
			logger.save_model(model.module, epoch)
			max_score = score

	logger.save()
	logger.save_model(model, "final")
	logger.print()
	logger.print("Test")
	logger.run_test("final")


def train(model, train_loader, loss_func, optimizer, logger):
	model.train()

	losses = []
	for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):

		images = images.to(primary_device, dtype=torch.float32, non_blocking=True)
		labels = labels.to(primary_device, dtype=torch.float32, non_blocking=True)

		outputs = model(images).squeeze(-1)
		loss = loss_func(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		logger.log_loss(loss.item())
		if i % (len(train_loader)//args.log_freq) == 0:
			mean_loss = np.mean(logger.losses[-10:])
			tqdm.tqdm.write(f"Train loss: {mean_loss:.5f}")
			logger.log(f"Train loss: {mean_loss:.7f}")

def evaluate(model, loader, loss_func, logger, it, splitname="val", threshold=0.5):
	model.eval()

	losses = []
	preds = []
	targets = []

	with torch.no_grad():
		for i, (images, labels) in tqdm.tqdm(enumerate(loader), total=len(loader)):

			images = images.to(primary_device, dtype=torch.float32, non_blocking=True).squeeze(0)
			labels = labels.to(primary_device, dtype=torch.float32, non_blocking=True)

			outputs, pred = model(images)
			loss = loss_func(outputs.mean(dim=0), labels).item()

			pred = pred.cpu().numpy()
			labels = labels.cpu().numpy().astype(np.int).squeeze()

			if pred.shape[0] != 1:
				pred = (0.5 * pred[0, :]) + (0.5 * pred[1:, :].mean(axis=0))
				pred = pred[np.newaxis, :]

			pred = pred > threshold

			losses.append(loss)
			preds.append(pred)
			targets.append(labels)

	targets = np.array(targets).squeeze()
	preds = np.array(preds).squeeze()

	acc = metrics.accuracy_score(targets, preds)
	f1 = metrics.f1_score(targets, preds)
	loss = np.mean(losses)

	sensitivity, specificity = sensitivity_specificity(targets, preds)

	logger.print()
	logger.print(f"Eval {it} - {splitname}")
	logger.print(f"Loss:        {loss:.4f}")
	logger.print(f"Accuracy:    {acc:.4f}")
	logger.print(f"F1:          {f1:.4f}")
	logger.print(f"Sensitivity: {sensitivity:.4f}")
	logger.print(f"Specificity: {specificity:.4f}")
	logger.print()

	logger.log_eval({
		"it": it,
		"loss": loss,
		"acc": acc,
		"f1": f1,
		"sensitivity": sensitivity,
		"specificity": specificity,
	}, splitname)
	return f1

if __name__ == "__main__":
	main()
