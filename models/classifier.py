import torch
from torch import nn
import torchvision
import pretrainedmodels
from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet

from torch.utils.checkpoint import checkpoint_sequential, checkpoint

class Classifier(nn.Module):

	def __init__(self, arch="inceptionv4", pool_type="avg", norm_type="batchnorm", ckpt=False):
		super(Classifier, self).__init__()

		model_configs = {
			"resnet50":            {"fc_size": 2048, "lib": "tv",   "ckpt": False, "seq": True,  "pretrained": True},
			"resnet152":           {"fc_size": 2048, "lib": "tv",   "ckpt": False, "seq": True,  "pretrained": True},
			"inception_v3":        {"fc_size": 2048, "lib": "tv",   "ckpt": False, "seq": True,  "pretrained": True},
			"inceptionv4":         {"fc_size": 1536, "lib": "ptm",  "ckpt": False, "seq": True,  "pretrained": True},
			"senet154":            {"fc_size": 2048, "lib": "ptm",  "ckpt": True,  "seq": True,  "pretrained": True},
			"densenet169":         {"fc_size": 1664, "lib": "tv",   "ckpt": True,  "seq": True,  "pretrained": True},
			"densenet161":         {"fc_size": 2208, "lib": "tv",   "ckpt": True,  "seq": True,  "pretrained": True},
			"densenet201":         {"fc_size": 1920, "lib": "tv",   "ckpt": True,  "seq": True,  "pretrained": True},
			"xception":            {"fc_size": 2048, "lib": "ptm",  "ckpt": True,  "seq": True,  "pretrained": True},
			"se_resnet50":         {"fc_size": 2048, "lib": "ptm",  "ckpt": False, "seq": False, "pretrained": True},
			"se_resnet152":        {"fc_size": 2048, "lib": "ptm",  "ckpt": True,  "seq": False, "pretrained": True},
			"se_resnext101_32x4d": {"fc_size": 2048, "lib": "ptm",  "ckpt": True,  "seq": False, "pretrained": True},
			"resnext101_64x4d":    {"fc_size": 2048, "lib": "ptm",  "ckpt": True,  "seq": False, "pretrained": True},
			"efficientnet-b0":     {"fc_size": 1280, "lib": "eff",  "ckpt": False, "seq": False, "pretrained": True},
			"efficientnet-b3":     {"fc_size": 1536, "lib": "eff",  "ckpt": True,  "seq": False, "pretrained": True},
			"efficientnet-b4":     {"fc_size": 1792, "lib": "eff",  "ckpt": True,  "seq": False, "pretrained": True},
			"efficientnet-b5":     {"fc_size": 2048, "lib": "eff",  "ckpt": True,  "seq": False, "pretrained": True},
			"efficientnet-b6":     {"fc_size": 2304, "lib": "eff",  "ckpt": True,  "seq": False, "pretrained": False},
			"efficientnet-b7":     {"fc_size": 2560, "lib": "eff",  "ckpt": True,  "seq": False, "pretrained": False},
			"bam_resnet50":        {"fc_size": 2048, "lib": "ptcv", "ckpt": False, "seq": True,  "pretrained": True},
			"bam_resnet152":       {"fc_size": 2048, "lib": "ptcv", "ckpt": True,  "seq": True,  "pretrained": False},
			"cbam_resnet50":       {"fc_size": 2048, "lib": "ptcv", "ckpt": True,  "seq": True,  "pretrained": True},
			"cbam_resnet152":      {"fc_size": 2048, "lib": "ptcv", "ckpt": True,  "seq": True,  "pretrained": False},
			"resattnet56":         {"fc_size": 2048, "lib": "ptcv", "ckpt": True,  "seq": True,  "pretrained": False},
			"resattnet92":         {"fc_size": 2048, "lib": "ptcv", "ckpt": True,  "seq": True,  "pretrained": False},
			"resattnet164":        {"fc_size": 2048, "lib": "ptcv", "ckpt": True,  "seq": True,  "pretrained": False},
		}

		valid_models = list(model_configs.keys())
		if not arch in valid_models:
			raise ValueError({"Invalid network architecture selection."})

		if ckpt == "auto": self.ckpt = model_configs[arch]["ckpt"]
		else:              self.ckpt = ckpt
		self.sequential = model_configs[arch]["seq"]

		if model_configs[arch]["lib"] == "tv":
			self.net = torchvision.models.__dict__[arch](pretrained=True)
		elif model_configs[arch]["lib"] == "ptm":
			self.net = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained="imagenet")
		elif model_configs[arch]["lib"] == "ptcv":
			self.net = ptcv_get_model(arch, pretrained=model_configs[arch]["pretrained"])
		elif model_configs[arch]["lib"] == "eff":
			if model_configs[arch]["pretrained"]: self.net = EfficientNet.from_pretrained(arch)
			else: self.net = EfficientNet.from_name(arch)
		else:
			raise ValueError("Invalid classifier library.")

		if pool_type == "avg":   self.pool = nn.AdaptiveAvgPool2d(1)
		elif pool_type == "max": self.pool = nn.AdaptiveMaxPool2d(1)
		else: raise ValueError("Invalid pooling method.")

		self.features = self.get_feature_extractor(arch, self.net)
		self.classifier = nn.Linear(model_configs[arch]["fc_size"], 1)

		if norm_type != "batchnorm": self.convert_norm(norm_type)

	def forward(self, x):

		if not self.ckpt: x = self.features(x)
		else:
			if self.sequential: x = checkpoint_sequential(self.features, 5, x)
			else:               x = checkpoint(self.features, x)

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
			net.features._modules["final_pool"] = nn.Identity()
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
		elif arch == "inception_v3":
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

	def convert_norm(self, norm_type):

		def to_instancenorm(module):
			mod = nn.InstanceNorm2d(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
			mod.running_mean = module.running_mean
			mod.running_var = module.running_var
			if module.affine:
				mod.weight.data = module.weight.data.clone().detach()
				mod.bias.data = module.bias.data.clone().detach()
			return mod
		def to_groupnorm(module):
			groups = min(32, module.num_features // 2)
			mod = nn.GroupNorm(groups, module.num_features, module.eps, module.affine)
			mod.running_mean = module.running_mean
			mod.running_var = module.running_var
			if module.affine:
				mod.weight.data = module.weight.data.clone().detach()
				mod.bias.data = module.bias.data.clone().detach()
			return mod
		def to_layernorm(module):
			groups = module.num_features
			mod = nn.GroupNorm(groups, module.num_features, module.eps, module.affine)
			mod.running_mean = module.running_mean
			mod.running_var = module.running_var
			if module.affine:
				mod.weight.data = module.weight.data.clone().detach()
				mod.bias.data = module.bias.data.clone().detach()
			return mod

		if norm_type == "instancenorm": norm_convert = to_instancenorm
		elif norm_type == "groupnorm":  norm_convert = to_groupnorm
		elif norm_type == "layernorm":  norm_convert = to_layernorm
		else: raise ValueError("Invalid norm type.")

		def convert_module(module):
			mod = module
			if isinstance(module, nn.BatchNorm2d):
				mod = norm_convert(module)
			for name, child in module.named_children():
				mod.add_module(name, convert_module(child))
			return mod

		convert_module(self)
