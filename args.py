import torch
import albumentations as tfms

class Args:

	##############################
	###### Hyperparameters #######
	##############################

	epochs = 100                    # DEFAULT 100 (int >1)
	batch_size = 4                  # DEFAULT 4 (int >1)
	weight_decay = 1e-5             # DEFAULT 1e-5 (float >=0)

	arch = "inceptionv4"            # DEFAULT inceptionv4 (resnet50 | resnet152 | senet154 | inceptionv4 | ...)
	checkpoint = False              # DEFUALT False (bool | auto)

	initial_lr = 3e-5               # DEFAULT 3e-5 (float >0)
	lr_schedule = "poly"            # DEFAULT poly (None | poly | exp | step | multistep | cosine)
	lr_schedule_params = {          # DEFAULT {} (dict)
	}

	loss = "bce"                    # DEFAULT bce (bce | focal | fbeta | softmargin)
	loss_params = {
	}

	device_ids = [0]                # DEFAULT [0,] (list int 0-8)
	workers = 4                     # DEFAULT 4 (int >=0)

	pretraining = False             # DEFAULT False (bool)
	pretrained = False              # DEFAULT False (bool)
	pretrain_info = None            # DEFAULT None (None | tuple(str, str | int))

	log_freq = 5                    # DEFAULT 5 (int >0)
	save_freq = 10                  # DEFAULT best (best | None | int>0)
	n_val_samples = None            # DEFAULT None (int >0 | None)
	n_train_eval_samples = 64       # DEFAULT 64 (int >0 | None)
	debug = False                   # DEFAULT False (bool)
	logging_enabled = True          # DEFAULT True (bool)

	train_split = "train"           # DEFAULT train (train | val | trainval)
	val_split   = "val"             # DEFAULT val (train | val | trainval)

	img_size = 1024                 # DEFAULT 1024 (None | int 224-4096)

	include_laser = True            # DEFAULT True (bool)
	example_weighting = False       # DEFAULT False (bool)

	train_augmentation = tfms.Compose([
		tfms.GaussNoise(var_limit=(2, 8)),
		tfms.HorizontalFlip(p=0.5),
		tfms.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
	])

	##############################
	########### Test #############
	##############################

	test_augmentation = [
		tfms.HorizontalFlip(always_apply=True),
	]

	##############################
	########## Paths #############
	##############################

	datapath  = "./data/"
	kaggle_datapath = "/home/felix/data/kaggle/"
