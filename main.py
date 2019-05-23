import os
import tqdm
import numpy as np
from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F

from loaders.loader import RetinaImageDataset
from util.logger import Logger
from util.misc import get_model, get_loss, get_train_sampler, get_scheduler, sensitivity_specificity

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from args import Args
args = Args()

primary_device = torch.device("cuda:{}".format(args.device_ids[0]))

def main():

	# datasets

	train_dataset = RetinaImageDataset(split=args.train_split, args=args,
		transforms=args.train_augmentation, debug=False)

	train_static_dataset = RetinaImageDataset(split=args.train_split, args=args,
		test_transforms=args.test_augmentation, debug=False,
		n_samples=args.n_train_eval_samples)

	val_dataset  = RetinaImageDataset(split=args.val_split, args=args,
		test_transforms=args.test_augmentation, debug=False,
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

	logger = Logger()
	max_score = 0

	for epoch in range(1, args.epochs+1):
		logger.print("Epoch {}".format(epoch))
		scheduler.step()
		train(model, train_loader, loss_func, optimizer, logger)
		_, threshold = evaluate(model, train_static_loader, loss_func, logger, splitname="train")
		score, _ = evaluate(model, val_loader, loss_func, logger, splitname="val", threshold=threshold)
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
			tqdm.tqdm.write("Train loss: {}".format(mean_loss))
			logger.log("Train loss: {}".format(mean_loss))

def evaluate(model, loader, loss_func, logger, splitname="val", threshold=0.5):
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
	logger.print("Eval - {}".format(splitname))
	logger.print("Loss:", loss)
	logger.print("Accuracy:", acc)
	logger.print("F1:", f1)
	logger.print("Sensitivity:", sensitivity)
	logger.print("Specificity:", specificity)
	logger.print()

	logger.log_eval({
		f"{splitname}-loss": loss,
		f"{splitname}-acc": acc,
		f"{splitname}-f1": f1,
		f"{splitname}-sensitivity": sensitivity,
		f"{splitname}-specificity": specificity,
	})
	return f1, threshold

if __name__ == "__main__":
	main()
