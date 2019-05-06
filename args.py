import torch
import albumentations as tfms

class Args:

	##############################
	###### Hyperparameters #######
	##############################

	epochs = 50                     # DEFAULT 50 (int 1-99)
	batch_size = 4                  # DEFAULT 16 (int)
	weight_decay = 1e-5             # DEFAULT 0 (float)

	arch = "resnet50"               # DEFAULT resnet50 (resnet50 | resnet152 | senet154 | inceptionv4)

	initial_lr = 3e-5               # DEFAULT 1e-5 (float)
	lr_schedule = None              # DEFAULT None (None | poly | exp | step | multistep | cosine)
	lr_schedule_params = {          # DEFAULT {} (dict)
	}

	loss = "bce"                    # DEFAULT softmargin (bce | focal | fbeta | softmargin)
	loss_params = {
		"focal_gamma": 2,           # DEFAULT 2 (float)
		"fbeta": 1                  # DEFAULT 1 (float)
	}

	weight_mode = None              # DEFAULT None ({inverse, sqrt} | None)
	weight_method = None            # DEFAULT None (loss | sampling | None)

	device_ids = [0]                # DEFAULT [0,] (list int 0-8)
	workers = 4                     # DEFAULT 8 (int 0-16)

	log_freq = 5                    # DEFAULT 5 (int)
	trainval_ratio = 0.80           # DEFAULT 0.80
	n_val_samples = None            # DEFAULT None (int | None)
	n_train_eval_samples = 64       # DEFAULT 64 (int | None)

	train_split = "train"           # DEFAULT train (train | val | trainval)
	val_split   = "val"             # DEFAULT val (train | val | trainval)

	img_size = 1024                 # DEFAULT None (None | int 224-4096)

	train_augmentation = tfms.Compose([
		tfms.HorizontalFlip(p=0.5),
		tfms.VerticalFlip(p=0.5),
		tfms.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20),
		tfms.RandomBrightnessContrast(),
		tfms.GaussNoise(var_limit=(2, 8))
	])

	##############################
	########### Test #############
	##############################

	test_augmentation = [         # DEFAULT [] (list)
		tfms.HorizontalFlip(p=1.0),
		tfms.VerticalFlip(p=1.0),
	]

	##############################
	########## Paths #############
	##############################

	datapath  = "./data/"
