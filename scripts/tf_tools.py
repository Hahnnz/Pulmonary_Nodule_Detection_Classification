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
