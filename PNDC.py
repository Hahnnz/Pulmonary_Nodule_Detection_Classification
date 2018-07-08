from models.tensorflow.deconvnet import *
from scripts.tools import *
import cv2

class PNDC:
    def __init__(self, img_path, roi_path, interval, batch_size, snapshot_path,
               threshold=None, predictOnly='all', gpu=None):
        
        self.img=cv2.imread(img_path)
        self.roi=cv2.imread(roi_path)
        self.interval = interval
        self.batch_size = batch_size
        self.snapshot_path = snapshot_path
        self.threshold = threshold
        self.predictOnly = predictOnly
        self.gpu = gpu
        
        self.pred_img = self.img.copy()
        self.answer = np.zeros(self.roi.shape)
        self.masked_img = self.img.copy()
        
        self._roi_masking()
        
    def predict(self):

        if self.gpu is not None and str(self.gpu).isdigit():
            set_GPU(str(self.gpu))
        elif self.gpu==None:
            pass
        else :
            raise ValueError("Gpu setting error occured!")

        if self.threshold == None:
            self.threshold = round(64/self.interval)

        num_frame = int((512-128)/self.interval)


        tf.reset_default_graph()
        Deconvnet = deconvnet((64,64,3), (64,64,3))

        with Deconvnet.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(Deconvnet.sess, self.snapshot_path)

        test_set = list()

        for i in range(num_frame):
            for j in range(num_frame):
                test_set.append(self.img[(self.interval*i)+64: (self.interval*i)+128,
                                         (self.interval*j)+64: (self.interval*j)+128])

        test_set=np.array(test_set)

        if len(test_set) > self.batch_size:
            te_batch_num = int(len(test_set)/self.batch_size)+1
            score_list = []
            for num in range(te_batch_num):
                test_batch = test_set[num*self.batch_size:(num+1)*self.batch_size] if num != te_batch_num-1 else test_set[num*self.batch_size:]

                score = Deconvnet.sess.run([Deconvnet.score], 
                                           feed_dict={Deconvnet.x: test_batch,
                                                      Deconvnet.keep_prob: 1.0})[0]
                score_list.append(Deconvnet.sess.run(tf.argmax(score,3)))
            score = np.concatenate(score_list, axis=0)

        else :
            score = Deconvnet.sess.run([Deconvnet.score], 
                                       feed_dict={Deconvnet.x: test_set,
                                                  Deconvnet.keep_prob: 1.0})[0]
            score = Deconvnet.sess.run(tf.argmax(score,3))

        for n in range(num_frame):
            for k in range(num_frame):
                for i in range(score.shape[1]):
                    for j in range(score.shape[2]):
                        if score[(n*num_frame)+k,i,j]==1:
                            self.answer[self.interval*(n+1)+i+(64-self.interval),
                                        self.interval*(k+1)+j+(64-self.interval),1]+=1
                        elif score[(n*num_frame)+k,i,j]==2:
                            self.answer[self.interval*(n+1)+i+(64-self.interval),
                                        self.interval*(k+1)+j+(64-self.interval),2]+=1

    def scoring(self):
        self.pred_img = self.img.copy()
        
        if self.predictOnly.lower() == 'benign':
            b_activate = True
            m_activate = False
        elif self.predictOnly.lower() == 'malignant':
            b_activate = False
            m_activate = True
        elif self.predictOnly.lower() == 'all':
            b_activate = True
            m_activate = True
        else :
            raise ValueError("'predictOnly' must be 'benign', 'malignant', 'all'.")
            
        for i in range(self.pred_img.shape[0]):
            for j in range(self.pred_img.shape[1]):
                if b_activate and self.answer[i,j,1] > self.threshold and self.answer[i,j,1] > self.answer[i,j,2]:
                    self.pred_img[i,j,0]=255
                elif m_activate and self.answer[i,j,2] > self.threshold  and self.answer[i,j,2] > self.answer[i,j,1]:
                    self.pred_img[i,j,2]=255

    def _roi_masking(self):
        for i in range(self.masked_img.shape[0]):
            for j in range(self.masked_img.shape[1]):
                if self.roi[i,j,0] > 0:
                    self.masked_img[i,j,1]=255
