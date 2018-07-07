from models.tensorflow.deconvnet import *
from scripts import dataset, iterator
from scripts.config import *
from scripts.tools import *
import scripts.preprocessing as pp
from scripts.tf_tools import *

# Set GPU Numbers to use
set_GPU("0")

# Load Train dataset
tr_benign = dataset.LMDB(Class='benign', Phase='train', mode='read').dataset
tr_malignant = dataset.LMDB(Class='malignant', Phase='train', mode='read').dataset
tr_none = dataset.LMDB(Class='none', Phase='train', mode='read').dataset

# Load Test dataset
te_benign = dataset.LMDB(Class='benign', Phase='test', mode='read').dataset
te_malignant = dataset.LMDB(Class='malignant', Phase='test', mode='read').dataset
te_none = dataset.LMDB(Class='none', Phase='test', mode='read').dataset

# Apply Mask RoI One Hot Encoding each class - Training
tr_benign['roi']=pp.RoI_OHE(tr_benign['roi'],'benign')
tr_malignant['roi']=pp.RoI_OHE(tr_malignant['roi'],'malignant')
tr_none['roi']=pp.RoI_OHE(tr_none['roi'],'none')

# Apply Mask RoI One Hot Encoding each class - Test
te_benign['roi']=pp.RoI_OHE(te_benign['roi'],'benign')
te_malignant['roi']=pp.RoI_OHE(te_malignant['roi'],'malignant')
te_none['roi']=pp.RoI_OHE(te_none['roi'],'none')

# Concatenate all of images and roi mask
train_img=np.concatenate((tr_benign['img'],tr_malignant['img'],tr_none['img']),axis=0)
train_roi=np.concatenate((tr_benign['roi'],tr_malignant['roi'],tr_none['roi']),axis=0)
test_img=np.concatenate((te_benign['img'],te_malignant['img'],te_none['img']),axis=0)
test_roi=np.concatenate((te_benign['roi'],te_malignant['roi'],te_none['roi']),axis=0)

# Shuffle Dataset
train_indices=np.random.permutation(len(train_img))
test_indices=np.random.permutation(len(test_img))
train_img, train_roi = train_img[train_indices], train_roi[train_indices]
test_img, test_roi = test_img[test_indices], test_roi[test_indices]

# Prepare Deconvnet
tf.reset_default_graph()
Deconvnet = deconvnet((64,64,3), (64,64,3))

# set Saver
with Deconvnet.graph.as_default():
    saver = tf.train.Saver()

# set Pixelwise softmax with Loss
loss = PixelWiseSoftmax_with_Loss(Deconvnet.score, Deconvnet.y)

# Accuracy setting
accuracy = RoI_Accuracy(Deconvnet.score, Deconvnet.y)

# Adadelta for optimizer
optimizer = tf.train.AdadeltaOptimizer(learning_rate=1,rho=0.95,epsilon=1e-09).minimize(loss)

# details
tr_batch_num = int(train_img.shape[0]/100)+1
te_batch_num = int(test_img.shape[0]/100)+1
num_epochs=10000
snapshot_step=1000
batch_size=100

# Training Loop Start
with Deconvnet.sess as sess:
    with tqdm(total = num_epochs) as pbar:
        with tf.device('/gpu:0'):
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                train_acc = 0.
                train_loss = 0.
                train_count = 0
                
                test_acc = 0.
                test_loss = 0.
                test_count = 0
                
                # Training epoch 
                for i in range(tr_batch_num):
                    data = train_img[i*batch_size:(i+1)*batch_size] if i != tr_batch_num-1 else train_img[i*batch_size:]
                    label = train_roi[i*batch_size:(i+1)*batch_size] if i != tr_batch_num-1 else train_roi[i*batch_size:]
                    acc,cost,op = sess.run([accuracy,loss,optimizer], feed_dict={Deconvnet.x: data,
                                                                                 Deconvnet.y: label, 
                                                                                 Deconvnet.keep_prob: 0.7})
                    train_acc += acc
                    train_loss += cost
                    train_count += 1
                
                # Test epoch    
                for i in range(te_batch_num):
                    data = test_img[i*batch_size:(i+1)*batch_size] if i != te_batch_num-1 else test_img[i*batch_size:]
                    label = test_roi[i*batch_size:(i+1)*batch_size] if i != te_batch_num-1 else test_roi[i*batch_size:]
                    acc,cost,op = sess.run([accuracy,loss,optimizer], feed_dict={Deconvnet.x: data,
                                                                                 Deconvnet.y: label, 
                                                                                 Deconvnet.keep_prob: 0.7})
                    test_acc += acc
                    test_loss += cost
                    test_count += 1
                
                # save Training state for each snapshot step
                if epoch%snapshot_step==0 and epoch !=0:
                    saver.save(sess, "./snapshot/tensorflow/Deconvnet_"+str(epoch)+".ckpt")

                pbar.set_description("[ Step : "+str(epoch+1)+"]")
                pbar.set_postfix_str(" Train - Accuracy : {:.5f}".format(train_acc/train_count if train_count !=0 else 0)+
                                     " Train - Loss : {:.5f}".format(train_loss/train_count if train_count !=0 else 0)+
                                     " Test - Accuracy : {:.5f}".format(test_acc/test_count if test_count !=0 else 0)+
                                     " Test - Loss : {:.5f}".format(test_loss/test_count if test_count !=0 else 0))
                
                pbar.update(1)
