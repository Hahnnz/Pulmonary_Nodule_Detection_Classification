import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

project_root = os.getcwd()

train_img_root = project_root + "/dataset/train/images/*"
train_roi_root = project_root + "/dataset/train/labels/*"

val_img_root = project_root + "/dataset/val/images/*"
val_roi_root = project_root + "/dataset/val/labels/*"

test_img_root = project_root + "/dataset/test/images/*"