import tensorflow as tf
from tqdm import tqdm

def IoU_Loss(logits, mask):
    """
    Description
        Compute loss by how much pixels net output and mask are overlapping each other
        
    Arguments
        - logits : Output of network
        - num_classes : number of Classes 
        - mask : mask labels
    
    Output
        Loss of for output of network
    """
    logits = tf.argmax(tf.transpose(logits,(0,3,1,2)),1)
    masks = tf.argmax(tf.transpose(mask,(0,3,1,2)),1)
    
    score_num_nodes=1
    for i in range(1,3): score_num_nodes*=int(logits.get_shape()[i])

    y_num_nodes=1
    for i in range(1,3): y_num_nodes*=int(mask.get_shape()[i])
    
    logits = tf.reshape(logits, (-1, score_num_nodes))
    mask = tf.reshape(mask, (-1, y_num_nodes))

    inter = tf.reduce_sum(tf.multiply(logits, mask))
    union = tf.reduce_sum(tf.subtract(tf.add(logits, mask), tf.multiply(logits, mask)))

    return tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))

def PixelWiseSoftmax_with_Loss(logits, mask):
    """
    Description
        Compute Softmax_with_Loss pixel by pixel
        
    Arguments
        - logits : Output of network (64x64x3 or 512x512x3) 3 - None, Benign, Malignant
        - mask : mask labels (64x64x1 or 512x512x1)
    
    Output
        Loss of output of network
    """
    mask = tf.argmax(mask,axis=3)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=mask,logits=logits)
    return tf.reduce_mean(cross_entropy)

def RoI_Accuracy(logits, mask):
    """
    Description
        Compare predicted and roi mask pixel by pixel 
        whether output of network is predicted well or not
        
    Arguments
        - logits : Output of network (64x64x3 or 512x512x3) 3 - None, Benign, Malignant
        - mask : mask labels (64x64x1 or 512x512x1)
    
    Output
        Accuracy of output of network
    """
    correct =tf.equal(tf.argmax(logits,3),tf.argmax(mask,3))
    return tf.reduce_mean(tf.cast(correct, tf.float32))