import os

def set_GPU(device_num):
    if type(device_num) is str:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=device_num
    else : raise ValueError("devuce number should be specified in str type")