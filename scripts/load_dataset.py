import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import cv2

def set_benign(mask):
    for j in range(64):
        for k in range(64):
            mask[j,k] = [0,1,0] if mask[j,k,0] >0 else [1,0,0]
    return mask

def set_Maligant(mask):
    for j in range(64):
        for k in range(64):
            mask[j,k] = [0,1,0] if mask[j,k,0] >0 else [1,0,0]
    return mask


def load(num_process=4):
    Test_Nodule_None = list()
    Test_Nodule_Benign = list()
    Test_Nodule_Malignant = list()

    Test_Mask_None = list()
    Test_Mask_Benign = list()
    Test_Mask_Malignant = list()
    
    NN = open("./anotations/None.txt").readlines()
    NN = list(NN[i][:-1] for i in range(len(NN)))
    NB = open("./anotations/Benign.txt").readlines()
    NB = list(NB[i][:-1] for i in range(len(NB)))
    NM = open("./anotations/Malignant.txt").readlines()
    NM = list(NM[i][:-1] for i in range(len(NM)))

    MN = open("./anotations/n_mask.txt").readlines()
    MN = list(MN[i][:-1] for i in range(len(MN)))
    MB = open("./anotations/b_mask.txt").readlines()
    MB = list(MB[i][:-1] for i in range(len(MB)))
    MM = open("./anotations/m_mask.txt").readlines()
    MM = list(MM[i][:-1] for i in range(len(MM)))
    
    
    Nodule_None = Pool(processes=num_process).map(cv2.imread,NN)
    Nodule_Benign = Pool(processes=num_process).map(cv2.imread,NB)
    Nodule_Malignant = Pool(processes=num_process).map(cv2.imread,NM)
    Mask_None = Pool(processes=num_process).map(cv2.imread,MN)
    Mask_Benign = Pool(processes=num_process).map(cv2.imread,MB)
    Mask_Benign = Pool(processes=num_process).map(set_benign,Mask_Benign)
    Mask_Malignant = Pool(processes=num_process).map(cv2.imread,MM)
    Mask_Malignant = Pool(processes=num_process).map(set_benign,Mask_Malignant)
    
    
    test_indices = np.random.choice(range(0,59912),5000,replace=False)
    
    for i, idx in enumerate(list(reversed(sorted(test_indices)))):
        Test_Nodule_None.append(Nodule_None.pop(idx))
        Test_Nodule_Benign.append(Nodule_Benign.pop(idx))
        Test_Nodule_Malignant.append(Nodule_Malignant.pop(idx))

        Test_Mask_None.append(Mask_None.pop(idx))
        Test_Mask_Benign.append(Mask_Benign.pop(idx))
        Test_Mask_Malignant.append(Mask_Malignant.pop(idx))
    
    
    Train_dataset = np.array(Nodule_None + Nodule_Benign + Nodule_Malignant)
    Train_roi_mask = np.array(Mask_None + Mask_Benign + Mask_Malignant)
    
    Test_dataset = np.array(Test_Nodule_None + Test_Nodule_Benign + Test_Nodule_Malignant)
    Test_roi_mask = np.array(Test_Mask_None + Test_Mask_Benign + Test_Mask_Malignant)
    
    return Train_dataset, Train_roi_mask, Test_dataset, Test_roi_mask