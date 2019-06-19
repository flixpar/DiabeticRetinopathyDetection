import torch
from torch import nn
import torchvision
import pretrainedmodels
from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet

from torch.utils.checkpoint import checkpoint_sequential, checkpoint

class Classifier(nn.Module):

	def __init__(self, arch="inceptionv4", pool_type="avg", ckpt=False):
		super(Classifier, self).__init__()

		model_configs = {
			"resnet50":            {"fc_size": 2048, "lib": "tv",   "ckpt": False, "pretrained": True},
			"resnet152":           {"fc_size": 2048, "lib": "tv",   "ckpt": False, "pretrained": True},
			"inceptionv3":         {"fc_size": 2048, "lib": "tv",   "ckpt": False, "pretrained": True},
			"inceptionv4":         {"fc_size": 1536, "lib": "ptm",  "ckpt": False, "pretrained": True},
			"senet154":            {"fc_size": 2048, "lib": "ptm",  "ckpt": True , "pretrained": True},
			"densenet169":         {"fc_size": 2048, "lib": "tv",   "ckpt": False, "pretrained": True},
			"densenet201":         {"fc_size": 2048, "lib": "tv",   "ckpt": False, "pretrained": True},
			"xception":            {"fc_size": 2048, "lib": "ptm",  "ckpt": True , "pretrained": True},
			"se_resnet50":         {"fc_size": 2048, "lib": "ptm",  "ckpt": False, "pretrained": True},
			"se_resnet152":        {"fc_size": 2048, "lib": "ptm",  "ckpt": False, "pretrained": True},
			"se_resnext101_32x4d": {"fc_size": 2048, "lib": "ptm",  "ckpt": False, "pretrained": True},
			"resnext101_62x4d":    {"fc_size": 2048, "lib": "ptm",  "ckpt": False, "pretrained": True},
			"efficientnet-b3":     {"fc_size": 2048, "lib": "eff",  "ckpt": False, "pretrained": True},
			"efficientnet-b4":     {"fc_size": 2048, "lib": "eff",  "ckpt": False, "pretrained": True},
			"efficientnet-b5":     {"fc_size": 2048, "lib": "eff",  "ckpt": False, "pretrained": True},
			"efficientnet-b6":     {"fc_size": 2048, "lib": "eff",  "ckpt": False, "pretrained": False},
			"efficientnet-b7":     {"fc_size": 2048, "lib": "eff",  "ckpt": False, "pretrained": False},
			"cbam_resnet50":       {"fc_size": 2048, "lib": "ptcv", "ckpt": False, "pretrained": True},
			"cbam_resnet152":      {"fc_size": 2048, "lib": "ptcv", "ckpt": False, "pretrained": False},
			"cbam_resnet50":       {"fc_size": 2048, "lib": "ptcv", "ckpt": False, "pretrained": True},
			"cbam_resnet152":      {"fc_size": 2048, "lib": "ptcv", "ckpt": False, "pretrained": False},
			"resattnet56":         {"fc_size": 2048, "lib": "ptcv", "ckpt": False, "pretrained": False},
			"resattnet92":         {"fc_size": 2048, "lib": "ptcv", "ckpt": False, "pretrained": False},
			"resattnet164":        {"fc_size": 2048, "lib": "ptcv", "ckpt": False, "pretrained": False},
		}

		valid_models = list(model_configs.keys())
		if not arch in valid_models:
			raise ValueError({"Invalid network architecture selection."})

		if ckpt == "auto": self.ckpt = model_configs[arch]["ckpt"]
		else:              self.ckpt = ckpt

		if model_configs[arch]["lib"] == "tv":
			self.net = torchvision.models.__dict__[arch](pretrained=True)
		if model_configs[arch]["lib"] == "ptm":
			self.net = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained="imagenet")
		elif model_configs[arch]["lib"] == "ptcv":
			self.net = ptcv_get_model(arch, pretrained=model_configs[arch]["pretrained"])
		elif model_configs[arch]["lib"] == "eff":
			if model_configs[arch]["pretrained"]: self.net = EfficientNet.from_pretrained(arch)
			else: self.net = EfficientNet.from_name(arch)
		else:
			raise ValueError("Invalid classifier library.")

		self.features = self.get_feature_extractor(arch, self.net)
		self.pool = nn.AdaptiveAvgPool2d(1) if pool_type == "avg" else nn.AdaptiveMaxPool2d(1)
		self.classifier = nn.Linear(model_configs[arch]["fc_size"], 1)

	def forward(self, x):

		if not self.ckpt: x = self.features(x)
		else:             x = checkpoint_sequential(self.features, 3, x)

		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)

		if not self.training:
			x = (x, torch.sigmoid(x))

		return x

	def get_feature_extractor(self, arch, net):
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
		elif "efficientnet" in arch:
			def features(x):
				x = net.extract_features(x)
				x = net._bn1(net._conv_head(x))
				x = x * torch.sigmoid(x)
				return x
			return features
		elif "resatt" in arch or "bam" in arch:
			net.features["final_pool"] = nn.Identity()
			return net.features
		elif arch in ["resnet50", "resnet152"]:
			return nn.Sequential(
				net.conv1,
				net.bn1,
				net.relu,
				net.maxpool,
				net.layer1,
				net.layer2,
				net.layer3,
				net.layer4,
			)
		elif arch == "inceptionv3":
			return nn.Sequential(
				net.Conv2d_1a_3x3,
				net.Conv2d_2a_3x3,
				net.Conv2d_2b_3x3,
				nn.MaxPool2d(kernel_size=3, stride=2),
				net.Conv2d_3b_1x1,
				net.Conv2d_4a_3x3,
				nn.MaxPool2d(kernel_size=3, stride=2),
				net.Mixed_5b,
				net.Mixed_5c,
				net.Mixed_5d,
				net.Mixed_6a,
				net.Mixed_6b,
				net.Mixed_6c,
				net.Mixed_6d,
				net.Mixed_6e,
				net.Mixed_7a,
				net.Mixed_7b,
				net.Mixed_7c
			)
		elif "densenet" in arch:
			net.features.add_module("relu_end", nn.ReLU(inplace=True))
			return net.features
		else:
			return net.features
