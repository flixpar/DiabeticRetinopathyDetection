import torch
from torch import nn
import torchvision
import pretrainedmodels

from torch.utils.checkpoint import checkpoint_sequential, checkpoint

class Classifier(nn.Module):

	def __init__(self, arch="inceptionv4", ckpt=False):
		super(Classifier, self).__init__()

		model_configs = {
			"resnet50":            {"fc_size": 2048, "lib": "ptm", "ckpt": False},
			"resnet152":           {"fc_size": 2048, "lib": "ptm", "ckpt": False},
			"inceptionv4":         {"fc_size": 1536, "lib": "ptm", "ckpt": False},
			"senet154":            {"fc_size": 2048, "lib": "ptm", "ckpt": True},
			"densenet169":         {"fc_size": 2048, "lib": "ptm", "ckpt": False},
			"densenet201":         {"fc_size": 2048, "lib": "ptm", "ckpt": False},
			"xception":            {"fc_size": 2048, "lib": "ptm", "ckpt": True},
			"se_resnet50":         {"fc_size": 2048, "lib": "ptm", "ckpt": False},
			"se_resnet152":        {"fc_size": 2048, "lib": "ptm", "ckpt": False},
			"se_resnext101_32x4d": {"fc_size": 2048, "lib": "ptm", "ckpt": False},
			"resnext101_62x4d":    {"fc_size": 2048, "lib": "ptm", "ckpt": False},
		}

		valid_models = list(model_configs.keys())
		if not arch in valid_models:
			raise ValueError({"Invalid network architecture selection."})

		self.net = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained="imagenet")
		self.feat_size = model_configs[arch]["fc_size"]

		if ckpt == "auto": self.ckpt = model_configs[arch]["ckpt"]
		else:              self.ckpt = ckpt

		if arch in ["senet154", "xception"]:
			self.features = extract_sequential(arch, self.net)
		else:
			self.features = self.net.features

		self.pool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(self.feat_size, 1)

	def forward(self, x):

		if not self.ckpt: x = self.features(x)
		else:             x = checkpoint_sequential(self.features, 3, x)

		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		if not self.training:
			x = (x, torch.sigmoid(x))

		return x


def extract_sequential(arch, net):
	if arch == "senet154":
		return nn.Sequential(
			net.layer0,
			net.layer1,
			net.layer2,
			net.layer3,
			net.layer4
		)
	elif arch == "xception":
		return nn.Sequential(
			net.conv1,
			net.bn1,
			nn.ReLU(inplace=True),
			net.conv2,
			net.bn2,
			nn.ReLU(inplace=True),
			net.block1,
			net.block2,
			net.block3,
			net.block4,
			net.block5,
			net.block6,
			net.block7,
			net.block8,
			net.block9,
			net.block10,
			net.block11,
			net.block12,
			net.conv3,
			net.bn3,
			nn.ReLU(inplace=True),
			net.conv4,
			net.bn4,
			nn.ReLU(inplace=True),
		)
	else:
		raise ValueError("Invalid arch for extract_sequential")
