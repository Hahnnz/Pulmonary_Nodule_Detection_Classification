import numpy as np
import os, glob

def set_GPU(device_num):
    if type(device_num) is str:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=device_num
    else : raise ValueError("devuce number should be specified in str type")
        
def get_filename(fulllocation):
    i=1
    while True:
        if fulllocation[-i] is not "/":i += 1
        else:
            name=fulllocation[-(i-1):]
            break
    return name

def explore_dir(dir,extension=None,count=0):
    if count==0:
        global n_dir, n_file, filenames, filelocations
        n_dir=n_file=0
        filenames=filelocations=np.array([])
    for img_path in sorted(glob.glob(os.path.join(dir,'*'))):
        if os.path.isdir(img_path):
            n_dir +=1
            explore_dir(img_path,count+1)
        elif os.path.isfile(img_path):
            n_file += 1
            loc=np.array([img_path])
            name=np.array([get_filename(img_path)])
            filelocations=np.concatenate((filelocations, loc), axis=0)
            filenames=np.concatenate((filenames, name), axis=0)
    return np.array([filenames,filelocations])
