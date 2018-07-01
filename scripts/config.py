import os
import numpy as np
import cv2
import tensorflow as tf
from copy import copy
from tqdm import tqdm

project_root = os.getcwd()

train_img_root = project_root + "/dataset/train/images/*"
train_roi_root = project_root + "/dataset/train/labels/*"

val_img_root = project_root + "/dataset/val/images/*"
val_roi_root = project_root + "/dataset/val/labels/*"

train_img_root_b = project_root + "/dataset/train/images/benign"
train_roi_root_b = project_root + "/dataset/train/labels/benign"
train_img_root_m = project_root + "/dataset/train/images/malignant"
train_roi_root_m = project_root + "/dataset/train/labels/malignant"

val_img_root_b = project_root + "/dataset/val/images/benign"
val_roi_root_b = project_root + "/dataset/val/labels/benign"
val_img_root_m = project_root + "/dataset/val/images/malignant"
val_roi_root_m = project_root + "/dataset/val/labels/malignant"