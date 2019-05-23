import os
import datetime
import pickle
import json
import csv
import shutil
import torch
import numpy as np
import pandas as pd
import subprocess

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


class Logger:

	def __init__(self, path=None, args=None):

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
			self.losses = []
			self.train_scores = []
			self.eval_scores = []
			self.main_log_fn = os.path.join(self.path, "log.txt")
			shutil.copy2("args.py", self.path)

	def save_model(self, model, epoch):
		if isinstance(epoch, int):
			fn = os.path.join(self.path, f"save_{epoch:03d}.pth")
		else:
			fn = os.path.join(self.path, f"save_{epoch}.pth")
		torch.save(model.state_dict(), fn)
		self.print(f"Saved model to: {fn}\n")

	def print(self, *x):
		print(*x)
		self.log(*x)

	def log(self, *x):
		with open(self.main_log_fn, "a") as f:
			print(*x, file=f, flush=True)

	def log_loss(self, l):
		self.losses.append(l)

	def log_eval(self, data, splitname):
		if splitname == "train":
			self.train_scores.append(data)
		elif splitname == "val":
			self.eval_scores.append(data)
		else:
			raise ValueError("Invalid splitname for logger.")

	def run_test(self, epoch):
		cmd = ["python3", "test.py", self.dt, epoch]
		self.print(" ".join(cmd))
		subprocess.run(cmd, shell=False)

	def save(self):

		with open(os.path.join(self.path, "loss.csv"), "w") as f:
			csvwriter = csv.DictWriter(f, ["it", "loss"])
			csvwriter.writeheader()
			for it, loss in enumerate(self.losses):
				row = {"it": it, "loss": loss}
				csvwriter.writerow(row)

		with open(os.path.join(self.path, "train_eval.csv"), "w") as f:
			cols = ["it"] + sorted(list(set(self.train_scores[0].keys()) - set(["it"])))
			csvwriter = csv.DictWriter(f, cols)
			csvwriter.writeheader()
			for row in self.train_scores:
				csvwriter.writerow(row)

		with open(os.path.join(self.path, "eval.csv"), "w") as f:
			cols = ["it"] + sorted(list(set(self.eval_scores[0].keys()) - set(["it"])))
			csvwriter = csv.DictWriter(f, cols)
			csvwriter.writeheader()
			for row in self.eval_scores:
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

		train_eval_data = pd.read_csv(os.path.join(self.path, "train_eval.csv"))
		evalplot = train_eval_data.plot(x="it", y="loss", legend=False, color="b")
		secondary_axis = evalplot.twinx()
		evalplot = train_eval_data.plot(x="it", y="f1",  legend=False, color="r", ax=secondary_axis)
		evalplot = train_eval_data.plot(x="it", y="acc", legend=False, color="g", ax=secondary_axis)
		evalplot = train_eval_data.plot(x="it", y="sensitivity", legend=False, color="m", ax=secondary_axis)
		evalplot = train_eval_data.plot(x="it", y="specificity", legend=False, color="c", ax=secondary_axis)
		evalplot.figure.legend()
		evalplot.grid(False)
		evalplot.set_title("Evaluation on Train Set")
		evalplot.figure.savefig(os.path.join(self.path, "train_eval.png"))
		plt.clf()
		plt.close()

		eval_data = pd.read_csv(os.path.join(self.path, "eval.csv"))

		evalplot = eval_data.plot(x="it", y="loss", legend=False, color="b")
		secondary_axis = evalplot.twinx()
		evalplot = eval_data.plot(x="it", y="f1",   legend=False, color="r", ax=secondary_axis)
		evalplot = eval_data.plot(x="it", y="acc",  legend=False, color="g", ax=secondary_axis)
		evalplot = eval_data.plot(x="it", y="sensitivity", legend=False, color="m", ax=secondary_axis)
		evalplot = eval_data.plot(x="it", y="specificity", legend=False, color="c", ax=secondary_axis)
		evalplot.figure.legend()
		evalplot.grid(False)
		evalplot.set_title("Evaluation on Validation Set")
		evalplot.figure.savefig(os.path.join(self.path, "eval.png"))

		plt.clf()
		plt.close()

		f1plot = sns.lineplot(
			x = "it",
			y = "f1",
			data = eval_data
		)
		f1plot.set_title("Eval F1 Score")
		f1plot.figure.savefig(os.path.join(self.path, "eval_f1.png"))

		plt.clf()
		plt.close()

		sensplot = sns.lineplot(
			x = "it",
			y = "sensitivity",
			data = eval_data
		)
		sensplot.set_title("Eval Sensitivity Score")
		sensplot.figure.savefig(os.path.join(self.path, "eval_sensitivity.png"))

		plt.clf()
		plt.close()

		specplot = sns.lineplot(
			x = "it",
			y = "specificity",
			data = eval_data
		)
		specplot.set_title("Eval Specificity Score")
		specplot.figure.savefig(os.path.join(self.path, "eval_specificity.png"))

		plt.clf()
		plt.close()

		accplot = sns.lineplot(
			x = "it",
			y = "acc",
			data = eval_data
		)
		accplot.set_title("Eval Accuracy")
		accplot.figure.savefig(os.path.join(self.path, "eval_acc.png"))

		plt.clf()
		plt.close()

		evallossplot = sns.lineplot(
			x = "it",
			y = "loss",
			data = eval_data
		)
		evallossplot.set_title("Eval Loss")
		evallossplot.figure.savefig(os.path.join(self.path, "eval_loss.png"))

		plt.clf()
		plt.close()
