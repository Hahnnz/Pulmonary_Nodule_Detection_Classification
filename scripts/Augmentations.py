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

def Crop_ROI(image, mask):
    """
    Arguments
        - image : CT original image (512,512,3)
        - mask : roi mask data (512,512,3)
    
    Output
        cropped image and roi as (64,64,3) 
    """
    # get roi bbox
    coors=detect(mask_img)
    
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
                coors[0]-=((64-x_size)/2) 
                coors[1]+=((64-x_size)/2) +1          

        if (y_size) !=64 :
            if y_size%2 == 0:
                coors[2]-=(64-y_size)/2
                coors[3]+=(64-y_size)/2
            elif y_size%2 !=0 :
                coors[2]-=((64-y_size)/2) 
                coors[3]+=((64-y_size)/2) +1 
        
        cropI = image[int(coors[0]):int(coors[1]+1), int(coors[2]):int(coors[3]+1)]
        cropM = mask_img[int(coors[1]):int(coors[1]+1), int(coors[2]):int(coors[3]+1)]
        return cropI, cropM,
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

def Resizing_img(image,mask):
    """
    Description
        Resize roi bbox to (64x64) without refreming
    
    Arguments
        - image : CT original image
        - mask : roi mask
        
    Output
        resized 
    """
    coors=detect(mask_img)
    
    cropI = image[coors[0]:(coors[1]+1), coors[2]:(coors[3]+1)]
    resizedI=cv2.resize(cropI, (64, 64))
    cropM = mask_img[coors[0]:(coors[1]+1), coors[2]:(coors[3]+1)]
    resizedM=cv2.resize(cropM, (64, 64))
    return resizedI, resizedM

