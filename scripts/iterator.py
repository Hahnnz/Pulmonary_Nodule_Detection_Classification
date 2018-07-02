from tensorflow.contrib.data import Iterator
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
from scripts.config import *
from scripts.tf_tools import *

class create:
    def __init__(self, benign, malignant, none, batch_size=None, Shuffle=False):
        self.batch_size = batch_size

        self.be_img = convert_to_tensor(benign['img'], dtype=dtypes.int32)
        self.be_roi = RoI_OHE(convert_to_tensor(benign['roi'], dtype=dtypes.int32),'benign')
        
        self.ma_img = convert_to_tensor(malignant['img'], dtype=dtypes.int32)
        self.ma_roi = RoI_OHE(convert_to_tensor(malignant['roi'], dtype=dtypes.int32),'malignant')
        
        self.no_img = convert_to_tensor(none['img'], dtype=dtypes.int32)
        self.no_roi = RoI_OHE(convert_to_tensor(none['roi'], dtype=dtypes.int32),'none')
        
        
        self.img = tf.concat((self.be_img,self.ma_img,self.no_img),axis=0)
        self.roi = tf.concat((self.be_roi,self.ma_roi,self.no_roi),axis=0)
        
        if Shuffle:
            self.img=tf.random_shuffle(self.img,seed=777)
            self.roi=tf.random_shuffle(self.roi,seed=777)
            
        self.dataset = tf.data.Dataset.from_tensor_slices((img, roi))
        self.dataset = self.dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        
        self.iterator = Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        next_batch = self.iterator.get_next()