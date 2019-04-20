import torch
from torch import nn
import torchvision
import pretrainedmodels

class Pretrained(nn.Module):

	def __init__(self, arch="resnet152", n_classes=28, n_input_channels=3):
		super(Pretrained, self).__init__()

		valid_model_sizes = {
			"resnet152": 2048,
			"inceptionv4": 1536,
			"senet154": 2048,
		}
		valid_models = list(valid_model_sizes.keys())

		if not arch in valid_models:
			raise ValueError({"Invalid network architecture selection."})

		self.net = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained='imagenet')
		self.feat_size = valid_model_sizes[arch]

		self.inflate_conv(arch, n_input_channels)

		self.pool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(self.feat_size, n_classes)

		nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x):
		x = self.net.features(x)
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def inflate_conv(self, arch, n_channels):

		if arch == "resnet152":
			layer = self.net.conv1
		elif arch == "inceptionv4":
			layer = self.net.features[0].conv
		elif arch == "senet154":
			layer = self.net.layer0.conv1

		original_state_dict = layer.state_dict()
		original_weights = original_state_dict["weight"]
		s = original_weights.shape

		mean_weights = original_weights.mean(dim=1).unsqueeze(dim=1)

		if n_channels == 1:
			weights = mean_weights
		elif n_channels == 2:
			weights = mean_weights.repeat(1, 2, 1, 1)
		elif n_channels == 3:
			return layer
		elif n_channels == 4:
			weights = mean_weights.repeat(1, 4, 1, 1)
		else:
			raise ValueError("Invalid number of input channels")

		out_state_dict = original_state_dict
		out_state_dict["weight"] = weights

		out_layer = nn.Conv2d(n_channels, s[0], kernel_size=(s[2],s[3]),
			stride=layer.stride, padding=layer.padding, bias=layer.bias)
		out_layer.load_state_dict(out_state_dict)

		if arch == "resnet152":
			self.net.conv1 = out_layer
		elif arch == "inceptionv4":
			self.net.features[0].conv = out_layer
		elif arch == "senet154":
			self.net.layer0.conv1 = out_layer
