import numpy as np
import cv2

def detect(mask):
    """
    Arguments
        - mask : roi mask data
    
    Output
        Bbox ractangle coordinates and Existence of roi
    """
    x_axis, y_axis=mask.shape[:2]
    coors=[0,0,0,0,0] # x1,x2,y1,y2,exist
    
    threshold=0
    
    for i in range(x_axis) :
        for j in range(y_axis) :
            if mask[i][j][0] > threshold or mask[i][j][1] > threshold or mask[i][j][2] > threshold :    
                coors[4] =1 # exist
                if coors[0] == 0 and coors[2] ==0:
                    coors[0] = coors[1] = i
                    coors[2] = coors[3] = j
                    
                elif coors[0]<i and coors[1] < i: 
                    coors[1] = i
                elif coors[1]>i :
                    coors[0] = i
                    
                elif coors[2]<j and coors[3] < j: 
                    coors[3] = j
                elif coors[3]>j :
                    coors[2] = j
    return coors

def Crop_ROI(image, mask, coors):
    """
    Arguments
        - image : CT original image shape of (512,512,3)
        - mask : roi mask data shape of (512,512,3)
        - coors : roi bbox coordinates (x1,x2,y1,y2,existance)
    
    Output
        cropped image and roi as (64,64,3) 
    """
    
    # if roi exist,
    if coors[4] == 1:                
        x_size = coors[1] - coors[0] +1
        y_size = coors[3] - coors[2] +1
        
        # reframe roi bbox to 64x64 with placing roi center
        if (x_size) !=64 :
            if x_size%2 == 0:
                coors[0]-=(64-x_size)/2
                coors[1]+=(64-x_size)/2
            elif x_size%2 != 0:
                coors[0]-=int((64-x_size)/2) 
                coors[1]+=int((64-x_size)/2) +1          

        if (y_size) !=64 :
            if y_size%2 == 0:
                coors[2]-=(64-y_size)/2
                coors[3]+=(64-y_size)/2
            elif y_size%2 !=0 :
                coors[2]-=int((64-y_size)/2) 
                coors[3]+=int((64-y_size)/2) +1
        coors = list(map(int,coors))
        
        cropI = image[int(coors[0]):int(coors[1]+1), int(coors[2]):int(coors[3]+1)]
        cropM = mask[int(coors[1]):int(coors[1]+1), int(coors[2]):int(coors[3]+1)]
        return cv2.resize(cropI,(64,64)), cv2.resize(cropM,(64,64))
    else :
        raise ValueError("RoI doesn't exist on this images.")
        
def Rotate_img(img,theta):
    """
    Arguments
        - img : CT original image
        - theta : degree for rotating
    
    Output
        image that rotated with given theta
    """
    x, y =img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((x/2,y/2),theta,1)
    return cv2.warpAffine(img, rotation_matrix, (x,y))
    
def Mirror_img(img):
    """
    Arguments
        - img : CT original image
    
    Output
        mirrored image
    """
    x, y =img.shape[:2]
    src_point = np.float32([[0,0], [y-1,0], [0,x-1]])
    dst_point = np.float32([[y-1,0], [0,0], [y-1, x-1]])
    affine_matrix = cv2.getAffineTransform(src_point, dst_point)
    return cv2.warpAffine(img, affine_matrix, (y,x))

def Resizing_img(image,mask,coors):
    """
    Description
        Resize roi bbox to (64x64) without refreming
    
    Arguments
        - image : CT original image
        - mask : roi mask
        - coors : roi bbox coordinates (x1,x2,y1,y2,existance)
        
    Output
        resized 
    """
    
    cropI = image[coors[0]:(coors[1]+1), coors[2]:(coors[3]+1)]
    resizedI=cv2.resize(cropI, (64, 64))
    cropM = mask[coors[0]:(coors[1]+1), coors[2]:(coors[3]+1)]
    resizedM=cv2.resize(cropM, (64, 64))
    return resizedI, resizedM

def Crop_None(image,mask,bbox_coors):
    """
    Description
        Crop None RoI image
        
    Arguments
        - image : CT original image shape of (512,512,3)
        - mask : roi mask data shape of (512,512,3)
        - coors : roi bbox coordinates (x1,x2,y1,y2,existance)
    
    Output
        cropped image and roi as (64,64,3) 
    """    

    Check_overwrapping=True
    bbox_coors=list(map(int,bbox_coors))
    while Check_overwrapping:
        x_coor, y_coor=np.random.choice(range(64,384),2)
        if x_coor not in range(bbox_coors[0],bbox_coors[1]+1) and y_coor not in range(bbox_coors[2],bbox_coors[3]+1):
            Check_overwrapping=False
            
    coors_img=[x_coor,x_coor+64,y_coor,y_coor+64]
    
    image = image[coors_img[0]:coors_img[1],coors_img[2]:coors_img[3]]
    mask = mask[coors_img[0]:coors_img[1],coors_img[2]:coors_img[3]]
    return image, mask

def RoI_OHE(roi_mask,Class, threshold=10):
    """
    Description
        Region Of Interest One How Encoding. This function is suitted for PNDC-project only.
        
    Arguments
        - roi_mask : roi mask dataset
        - Class : insert 'Benign' or 'Malignant' or 'none'
    
    Output
        one hot encoded roi mask
    """
    if Class.lower() == "benign": ohe = [0,1,0]
    elif Class.lower() == "malignant": ohe = [0,0,1]
    elif Class.lower() == 'none': ohe = [1,0,0]
    else : raise ValueError("Class should 'Benign' or 'Malignant' or 'none'.")
    
    roi_set = None
    one_hot_roi = None
    num, roi_x, roi_y = roi_mask.shape[:3]
    
    for n in range(num):
        for i in range(roi_x):
            for j in range(roi_y):
                roi_mask[n, i,j] = ohe if roi_mask[n,i,j,0] > threshold else [1,0,0]
    return roi_mask