import torch
import albumentations as tfms

class Args:

	##############################
	###### Hyperparameters #######
	##############################

	# general
	epochs = 100                    # DEFAULT 100 (int >1)
	batch_size = 4                  # DEFAULT 4 (int >1)

	# model
	arch = "inceptionv4"            # DEFAULT inceptionv4 (resnet50 | resnet152 | senet154 | inceptionv4 | ...)
	checkpoint = "auto"             # DEFUALT auto (bool | auto)
	pool_type = "avg"               # DEFAULT avg (avg | max)
	norm_type = "batchnorm"         # DEFAULT batchnorm (batchnorm | instancenorm | groupnorm | layernorm)

	# loss
	loss = "crossentropy"           # DEFAULT crossentropy (crossentropy | mse | l1 | multimargin | focal)
	loss_params = {
	}

	# optimization
	initial_lr = 5e-5               # DEFAULT 5e-5 (float >0)
	weight_decay = 1e-5             # DEFAULT 1e-5 (float >=0)
	lr_schedule = "poly"            # DEFAULT poly (None | poly | exp | step | multistep | cosine)
	lr_schedule_params = {          # DEFAULT {} (dict)
	}

	# dataset
	train_split = "train"           # DEFAULT train (train | val | trainval)
	val_split   = "val"             # DEFAULT val (train | val | trainval)
	n_vval_samples = None           # DEFAULT None (int >0 | None)
	n_tval_samples = 64             # DEFAULT 64 (int >0 | None)
	img_size = 1024                 # DEFAULT 1024 (None | int 224-4096)

	# class balancing
	balancing_method = None         # DEFAULT None (None | loss | sampling)
	balancing_params = {
	}

	# pretraining
	pretraining = False             # DEFAULT False (bool)
	pretrained_model = None         # DEFAULT None (None | tuple(str, str | int))

	# dataloader
	device_ids = [0]                # DEFAULT [0,] (list int 0-8)
	workers = 4                     # DEFAULT 4 (int >=0)

	# logging
	log_freq = 5                    # DEFAULT 5 (int >0)
	save_freq = 10                  # DEFAULT best (best | None | int>0)
	debug = False                   # DEFAULT False (bool)
	logging_enabled = True          # DEFAULT True (bool)
	display_graph = False           # DEFAULT False (bool)

	# data augmentation
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

	dr_datapath  = "/home/felix/data/kaggle_dr/"
	blindness_datapath = "/home/felix/data/kaggle_blindness/"
