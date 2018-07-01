from scripts.config import *
from scripts.tools import *
import scripts.preprocessing as pp
from tqdm import tqdm
from multiprocessing import Pool


class load():
    def __init__(self, Class, img_root, roi_root, 
                 Rotate=False, theta_set={90},Fliplr=False, Resize=False, Shuffle=False, num_process=4):
        if Class.lower() not in ['benign','malignant']:
            raise ValueError('Class must be "benign" or "malignant".')
        
        self.Class = Class.lower()
        
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
            pbar.set_description("[Class : "+self.Class+"]")
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
        self.dataset = {self.Class+'_img':np.array(self.c_img),self.Class+'_roi':np.array(self.c_roi),
                       'none_img':np.array(self.n_img),'none_roi':np.array(self.n_roi)}
    def _im_and_roi_read(self, paths):
        return cv2.imread(paths[1,0]), cv2.imread(paths[1,1])
    
#class save():
    # lmdb saver needed