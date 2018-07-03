import tensorflow as tf
from tqdm import tqdm

def IoU_Loss(logits, mask):
    score_num_nodes=1
    for i in range(1,4): score_num_nodes*=int(logits.get_shape()[i])

    y_num_nodes=1
    for i in range(1,4): y_num_nodes*=int(mask.get_shape()[i])
    
    logits = tf.reshape(logits, (-1, score_num_nodes))
    mask = tf.reshape(tf.to_float(mask), (-1, y_num_nodes))

    inter = tf.reduce_sum(tf.multiply(logits, mask))
    union = tf.reduce_sum(tf.subtract(tf.add(logits, mask), tf.multiply(logits, mask)))

    return tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))

def RoI_OHE(roi_mask,Class):
    if Class.lower() == "benign": Class = tf.constant([0,1,0])
    elif Class.lower() == "malignant": Class = tf.constant([0,0,1])
    elif Class.lower() == "none": Class = tf.constant([1,0,0])
    else : raise ValueError("Class should 'Benign' or 'Malignant' or 'none'.")
    
    roi_set = None
    one_hot_roi = None
    num, roi_x, roi_y = list(map(int,roi_mask.get_shape()))[:3]
    
    processed_batchs = list()
    count = 0
        
    with tqdm(total = num) as pbar:
        for n in range(num):
            for i in range(roi_x):
                one_hot = tf.map_fn(lambda x: 
                                        tf.cond(tf.greater(x[0],0), 
                                                lambda: Class, lambda: tf.constant([1,0,0])),
                                        roi_mask[n,i])
                if i == 0 :
                    one_hot_roi = tf.expand_dims(one_hot,0)
                else :
                    one_hot_roi = tf.concat((one_hot_roi,tf.expand_dims(one_hot,0)),axis=0)
            if n == 0 :
                count+=1
                roi_set = tf.expand_dims(one_hot_roi,0)
            else :
                roi_set = tf.concat((roi_set,tf.expand_dims(one_hot_roi,0)),axis=0)
                count+=1
                if count==100:
                    processed_batchs.append(roi_set)
                    roi_set=None
                    count=0
            pbar.update(1)
        if roi_set is not None : processed_batchs.append(roi_set)
    return tf.concat(processed_batches, axis=0) if len(processed_batches)>1 else processed_batches[0]