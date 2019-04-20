# Human Protein Atlas Image Classification

This project contains code for multi-label classification on the kaggle subset of the Human Protein Atlas dataset.

Implemented features:
- loss functions
  - binary cross entropy
  - focal loss
  - soft F-beta loss
- CNN architectures
  - ResNet (50, 152)
  - Senet154
  - InceptionV4
- learning rate scheduling
- loss weighting
- sample weighting
- stratified split for train/val
- extensive data augmentation
- test-time augmentation
- rule-based post-processing
- adaptive f1 thresholding
