from scripts.config import *
from scripts.tools import *
import scripts.preprocessing as pp
from tqdm import tqdm
from multiprocessing import Pool


class load():
    def __init__(self, Class, Phase, img_root, roi_root, Rotate=False, theta_set={90},
                 Fliplr=False, Resize=False, Shuffle=False, roi_ohe=False, num_process=4):
        if Class.lower() not in ['none','benign','malignant']:
            raise ValueError('Class must be "benign" or "malignant".')
        if Phase.lower() not in ['train','test']:
            raise ValueError('Phase must be "train" or "test".')
        
        self.Class = Class.lower()
        self.Phase = Phase.lower()
        
        img_path = explore_dir(img_root)[np.newaxis,:,:]
        roi_path = explore_dir(roi_root)[np.newaxis,:,:]

        paths = np.concatenate((img_path,roi_path), axis=0).transpose(2,1,0)
        
        self.dataset=np.array(Pool(processes=num_process).map(self._im_and_roi_read,paths)).transpose(1,0,2,3,4)
        
        if Rotate:
            img_set=copy(self.dataset[0])
            roi_set=copy(self.dataset[1])
            
            rotated_img = list()
            rotated_roi = list()
            
            for i in range(len(roi_set)):
                for theta in theta_set:
                    rotated_img.append(pp.Rotate_img(img_set[i],theta))
                    rotated_roi.append(pp.Rotate_img(roi_set[i],theta))
            rotated = np.array([rotated_img, rotated_roi])
            self.dataset = np.concatenate((self.dataset,rotated),axis=1)
            
        if Fliplr:
            img_set=copy(self.dataset[0])
            roi_set=copy(self.dataset[1])
            
            fliplr_img = list()
            fliplr_roi = list()
            
            for i in range(len(roi_set)):
                fliplr_img.append(pp.Mirror_img(img_set[i]))
                fliplr_roi.append(pp.Mirror_img(roi_set[i]))
            fliplr = np.array([fliplr_img, fliplr_roi])
            self.dataset = np.concatenate((self.dataset,fliplr),axis=1)
        
        self.c_img=list()
        self.n_img=list()

        self.c_roi=list()
        self.n_roi=list()
        
        with tqdm(total=self.dataset.shape[1]) as pbar:
            pbar.set_description("[Phase : "+self.Phase+"] / [Class : "+self.Class+"]")
            for i in range(self.dataset.shape[1]):
                img = self.dataset[0,i]
                roi = self.dataset[1,i]
                bbox_coor = pp.detect(roi)

                if bbox_coor[-1]==1:
                    c_img, c_roi = pp.Crop_ROI(img,roi,bbox_coor)
                    self.c_img.append(c_img)
                    self.c_roi.append(c_roi)

                    c_img, c_roi = pp.Crop_None(img, roi, bbox_coor)
                    self.n_img.append(c_img)
                    self.n_roi.append(c_roi)
                pbar.update(1)
                
        
        if roi_ohe :
            for i in range(len(self.dataset[1])):
                self.dataset[1,i] = pp.RoI_OHE(self.dataset[1,i],self.Class)
        
        self.dataset = {self.Class+'_img':np.array(self.c_img),self.Class+'_roi':np.array(self.c_roi),
                       'none_img':np.array(self.n_img),'none_roi':np.array(self.n_roi)}
    def _im_and_roi_read(self, paths):
        return cv2.imread(paths[1,0]), cv2.imread(paths[1,1])
    
class LMDB():
    def __init__(self, Class, Phase, mode, dataset=None,
                 caffe_root = './caffe/', lmdb_root='./LMDB/'):
        if Class.lower() not in ['none','benign','malignant']:
            raise ValueError('Class must be "benign" or "malignant".')
        if Phase.lower() not in ['train','test']:
            raise ValueError('Phase must be "train" or "test".')
        if mode.lower() not in ['read','write']:
            raise ValueError('mode must be "read" or "write".')
        if mode.lower() == 'write' and dataset == None:
            raise ValueError('Dataset is required when mode is "write".')
        
        self.Class = Class.lower()
        self.Phase = Phase.lower()
            
        if mode == 'write':
            self._write(dataset, caffe_root, lmdb_root)
        elif mode == 'read':
            self.dataset = self._read(lmdb_root)
            
    def _write(self, dataset, caffe_root, lmdb_root):
        try:
            import lmdb
            import sys
            sys.path.insert(0, caffe_root + 'python')
            import caffe
            from caffe.proto import caffe_pb2
        except:
            raise ValueError("Library Error Occurred!")
            
        lmdb_env = lmdb.open(lmdb_root+self.Phase+'_'+self.Class+'_lmdb', map_size=int(1e12))
        # B X C X W X H
        for i in range(len(dataset[self.Class+'_img'])):
            datum_img = caffe.io.array_to_datum(dataset[self.Class+'_img'][i].transpose(2,0,1))
            datum_roi = caffe.io.array_to_datum(dataset[self.Class+'_roi'][i].transpose(2,0,1))

            with lmdb_env.begin(write=True) as lmdb_txn:
                lmdb_txn.put((self.Class+'_img_'+'{:08}'.format(i+1)).encode('ascii'),datum_img.SerializeToString())
                lmdb_txn.put((self.Class+'_roi_'+'{:08}'.format(i+1)).encode('ascii'),datum_roi.SerializeToString())
                
    def _read(self, lmdb_root):
        try:
            import lmdb
        except:
            raise ValueError("Library Error Occurred!")
            
        lmdb_env = lmdb.open(lmdb_root+self.Phase+"_"+self.Class+"_lmdb")
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
    
        img =list()
        roi = list()
    
        n_counter=0
        with lmdb_env.begin() as lmdb_txn:
            with lmdb_txn.cursor() as lmdb_cursor:
                for key, value in lmdb_cursor:  
                    if(self.Class+'_img' in str(key)):
                        img.append(np.fromstring(value, dtype=np.uint8)[9:].reshape(3,64,64).transpose(1,2,0))
                    if(self.Class+'_roi' in str(key)):
                        roi.append(np.fromstring(value, dtype=np.uint8)[9:].reshape(3,64,64).transpose(1,2,0))
    
        return {'img': np.array(img), 'roi': np.array(roi)}
