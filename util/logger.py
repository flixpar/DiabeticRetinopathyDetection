import os
import datetime
import pickle
import json
import csv
import shutil
import tqdm
import torch
import numpy as np
import pandas as pd
import subprocess

from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


class Logger:

	def __init__(self, path=None, args=None, enabled=True):
		self.logging = enabled
		if not self.logging:
			self.losses = []
			return

		if path is not None:
			if not os.path.isdir(path):
				raise ValueError("Invalid logger path given.")
			self.path = path
			self.dt = path.replace('/','').replace('.','').replace('saves','')
			self.main_log_fn = os.path.join(self.path, "test.txt")

		else:
			self.dt = datetime.datetime.now().strftime("%m%d_%H%M")
			self.path = f"./saves/{self.dt}"
			if args is not None and args.debug: self.path = self.path + "_debug"
			if args is not None and args.pretraining: self.path = self.path + "_pretraining"
			if not os.path.exists(self.path):
				os.makedirs(self.path)
			else:
				raise FileExistsError(f"Save folder already exists: {self.path}")
			self.epoch = 1
			self.iterations = 0
			self.current_split = "train"
			self.losses = []
			self.tval_scores = []
			self.vval_scores = []
			self.main_log_fn = os.path.join(self.path, "log.txt")
			shutil.copy2("args.py", self.path)
			self.tensorboard = SummaryWriter(log_dir=os.path.join(self.path, "tensorboard"))
			self.save_git()

	def save_git(self):

		git_commit_short = subprocess.run(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE)
		git_commit_long  = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
		git_diff = subprocess.run(["git", "diff"], stdout=subprocess.PIPE)

		git_commit_short = git_commit_short.stdout.decode('UTF-8').strip()
		git_commit_long  = git_commit_long.stdout.decode('UTF-8').strip()
		git_diff = git_diff.stdout.decode('UTF-8')

		with open(os.path.join(self.path, "git.txt"), "w") as f:
			f.write(f"commit: {git_commit_short}\n")
			f.write(f"commit (full): {git_commit_long}\n")
			f.write("\n\nDIFF:\n")
			f.write(git_diff)

	def save_model(self, model):
		if not self.logging: return
		fn = os.path.join(self.path, f"save_{self.epoch:03d}.pth")
		torch.save(model.state_dict(), fn)
		self.print(f"Saved model to: {fn}\n")

	def print(self, *x):
		print(*x)
		self.log(*x)

	def log(self, *x):
		if not self.logging: return
		with open(self.main_log_fn, "a") as f:
			print(*x, file=f, flush=True)

	def log_loss(self, l):
		self.losses.append((self.epoch, self.iterations, l))
		self.tensorboard.add_scalar("train/loss/", l, global_step=self.iterations)
		self.iterations += 1

	def log_scores(self, s):
		if not self.logging: return
		if self.current_split == "tval":
			self.tval_scores.append((self.epoch, s))
			self.tensorboard.add_scalars("train/scores/", s, global_step=self.epoch)
		elif self.current_split == "vval":
			self.vval_scores.append((self.epoch, s))
			self.tensorboard.add_scalars("val/scores/", s, global_step=self.epoch)
		else:
			raise ValueError("Invalid split name for logger.")

	def print_loss(self, mode="mean", k=10, print_func="tqdm"):
		if not self.logging: return

		if mode == "mean":
			if len(self.losses) < k: k = len(self.losses)
			loss = np.mean([l[-1] for l in self.losses[-k:]])

		elif mode == "last":
			loss = self.losses[-1][-1]

		loss_str = f"Loss: {loss:.5f}"

		if print_func == "tqdm":
			tqdm.tqdm.write(loss_str)
			self.log(loss_str)
		else:
			self.print(loss_str)

	def complete_epoch(self):
		self.save()
		self.epoch += 1

	def train(self):
		self.current_split = "train"

	def eval(self, dset="val"):
		self.current_split = dset

	def display_model(self, model, device, display=True):
		if not self.logging: return

		if display:

			self.print(model)

			total_params = 0
			for param in model.parameters():
				total_params += param.numel()
			self.print(f"Total number of parameters: {(total_params/1e6):.3f}M")
			self.print("-----------------------------------------------\n")

			fake_data = torch.randn((2, 3, 128, 128), dtype=torch.float32, device=device)
			self.tensorboard.add_graph(model, fake_data)

	def save(self):
		if not self.logging: return

		with open(os.path.join(self.path, "loss.csv"), "w") as f:
			csvwriter = csv.DictWriter(f, ["it", "loss"])
			csvwriter.writeheader()
			for _, it, loss in self.losses:
				row = {"it": it, "loss": loss}
				csvwriter.writerow(row)

		with open(os.path.join(self.path, "tval.csv"), "w") as f:
			cols = ["epoch"] + sorted(list(self.tval_scores[0][-1].keys()))
			csvwriter = csv.DictWriter(f, cols)
			csvwriter.writeheader()
			for e, s in self.tval_scores:
				row = s
				row.update({"epoch": e})
				csvwriter.writerow(row)

		with open(os.path.join(self.path, "vval.csv"), "w") as f:
			cols = ["epoch"] + sorted(list(self.vval_scores[0][-1].keys()))
			csvwriter = csv.DictWriter(f, cols)
			csvwriter.writeheader()
			for e, s in self.vval_scores:
				row = s
				row.update({"epoch": e})
				csvwriter.writerow(row)

		plt.clf()

		loss_data = pd.read_csv(os.path.join(self.path, "loss.csv"))
		loss_means = loss_data.copy()
		loss_means.loss = loss_means.loss.rolling(20, center=True, min_periods=1).mean()
		lossplot = sns.lineplot(
			x = "it",
			y = "loss",
			data = loss_data,
			color = "b"
		)
		lossplot = sns.lineplot(
			x = "it",
			y = "loss",
			data = loss_means,
			color = "orange"
		)
		lossplot.set_title("Train loss")
		lossplot.figure.savefig(os.path.join(self.path, "train_loss.png"))

		plt.clf()
		plt.close()

		tval_data = pd.read_csv(os.path.join(self.path, "tval.csv"))
		evalplot = tval_data.plot(x="epoch", y="loss", legend=False, color="b")
		secondary_axis = evalplot.twinx()
		evalplot = tval_data.plot(x="epoch", y="f1",    legend=False, color="r", ax=secondary_axis)
		evalplot = tval_data.plot(x="epoch", y="acc",   legend=False, color="g", ax=secondary_axis)
		evalplot = tval_data.plot(x="epoch", y="kappa", legend=False, color="m", ax=secondary_axis)
		evalplot.figure.legend()
		evalplot.grid(False)
		evalplot.set_title("Evaluation on Train Set")
		evalplot.figure.savefig(os.path.join(self.path, "tval.png"))
		plt.clf()
		plt.close()

		vval_data = pd.read_csv(os.path.join(self.path, "vval.csv"))
		evalplot = vval_data.plot(x="epoch", y="loss", legend=False, color="b")
		secondary_axis = evalplot.twinx()
		evalplot = vval_data.plot(x="epoch", y="f1",    legend=False, color="r", ax=secondary_axis)
		evalplot = vval_data.plot(x="epoch", y="acc",   legend=False, color="g", ax=secondary_axis)
		evalplot = vval_data.plot(x="epoch", y="kappa", legend=False, color="m", ax=secondary_axis)
		evalplot.figure.legend()
		evalplot.grid(False)
		evalplot.set_title("Evaluation on Validation Set")
		evalplot.figure.savefig(os.path.join(self.path, "vval.png"))

		plt.clf()
		plt.close()

		kappaplot = sns.lineplot(
			x = "epoch",
			y = "kappa",
			data = vval_data
		)
		kappaplot.set_title("Eval Quadratic-Weighted Kappa Score")
		kappaplot.figure.savefig(os.path.join(self.path, "eval_kappa.png"))

		plt.clf()
		plt.close()

		f1plot = sns.lineplot(
			x = "epoch",
			y = "f1",
			data = vval_data
		)
		f1plot.set_title("Eval F1 Score")
		f1plot.figure.savefig(os.path.join(self.path, "eval_f1.png"))

		plt.clf()
		plt.close()

		accplot = sns.lineplot(
			x = "epoch",
			y = "acc",
			data = vval_data
		)
		accplot.set_title("Eval Accuracy")
		accplot.figure.savefig(os.path.join(self.path, "eval_acc.png"))

		plt.clf()
		plt.close()

		evallossplot = sns.lineplot(
			x = "epoch",
			y = "loss",
			data = vval_data
		)
		evallossplot.set_title("Eval Loss")
		evallossplot.figure.savefig(os.path.join(self.path, "eval_loss.png"))

		plt.clf()
		plt.close()
