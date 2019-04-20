import torch
from torch import nn
import torch.optim.lr_scheduler
from torch.utils.data import WeightedRandomSampler

from models.resnet import Resnet
from models.pretrained import Pretrained
from models.loss import MultiLabelFocalLoss, FBetaLoss

def get_model(args):
	n_channels = len(set(args.img_channels))
	if args.arch in ["resnet152", "resnet50"]:
		layers = int(args.arch[6:])
		model = Resnet(layers=layers, n_input_channels=n_channels)
	elif args.arch in ["inceptionv4", "setnet154"]:
		model = Pretrained(args.arch, n_input_channels=n_channels)
	else:
		raise ValueError("Invalid model architecture: {}".format(args.arch))
	return model

def get_loss(args, weights):

	if args.weight_method == "loss":
		if args.weight_mode is not None and "inverse" in args.weight_mode:
			class_weights = weights
			if "sqrt" in args.weight_mode:
				class_weights = torch.sqrt(class_weights)
		else:
			class_weights = None
	else:
		class_weights = None

	if args.loss == "softmargin":
		loss_func = nn.MultiLabelSoftMarginLoss(weight=class_weights)
	elif args.loss == "focal":
		loss_func = MultiLabelFocalLoss(weight=class_weights, gamma=args.focal_gamma)
	elif args.loss == "fbeta":
		loss_func = FBetaLoss(weight=class_weights, beta=args.fbeta, soft=True)
	else:
		raise ValueError("Invalid loss function specifier: {}".format(args.loss))

	return loss_func

def get_train_sampler(args, dataset):
	if args.weight_method == "sampling":
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
		gamma = params["gamma"] if "gamma" in params else 0.9
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
