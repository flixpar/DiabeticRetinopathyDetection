import torch
from torch import nn
import torchvision
import pretrainedmodels

class Classifier(nn.Module):

	def __init__(self, arch="resnet152", n_input_channels=3):
		super(Classifier, self).__init__()

		valid_model_sizes = {
			"resnet50": 2048,
			"resnet152": 2048,
			"inceptionv4": 1536,
			"senet154": 2048,
		}
		valid_models = list(valid_model_sizes.keys())

		if not arch in valid_models:
			raise ValueError({"Invalid network architecture selection."})

		self.net = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
		self.feat_size = valid_model_sizes[arch]

		self.pool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(self.feat_size, 1)

		nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x):
		x = self.net.features(x)
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x
