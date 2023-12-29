import numpy as np
import cv2
import os
import sys
import glob
import random

def data_append(old_data, new_data):
    if old_data.size == 0:
        old_data = new_data
    else:
        old_data = np.concatenate((old_data, new_data), axis = 0)
    return old_data  

def get_txt_files_from_img_files(img_files):
    txt_files = []
    for img_file in img_files:
        txt_files.append(os.path.splitext(img_file)[0] + '.txt')
    return txt_files

def get_img_data(img_files, img_size, data):
    for img_file in img_files:
        img = cv2.imread(img_file)
        img = cv2.resize(img, img_size)
        img = img[np.newaxis, :] / 255.
        data = data_append(data, img)
    return data   

def get_txt_data(txt_files, data):
    for txt_file in txt_files:
        txt = np.loadtxt(txt_file, delimiter=' ')
        txt = txt[np.newaxis, :]
        data = data_append(data, txt)
    return data

def split_shuffle(data_size):
    order = list(range(data_size))
    random.shuffle(order)
    return order 
        
def split_train_test(data, split, sf_list):
    split = int(split * data.shape[0])
    train = data[0:split, :]
    test = data[split::, :]
    train_order = sf_list[0:split]
    test_order = sf_list[split::]
    for i, train_o in enumerate(train_order):
        train[i] = data[train_o]
    for i, test_o in enumerate(test_order):
        test[i] = data[test_o]
    
    return train, test
    

def rcnn_format(root, img_size):
    x = np.array([])
    y = np.array([])
    if len(img_size) == 3: img_size = img_size[0:-1]
    subfolders = glob.glob(os.path.join(root, '*'))
   
    if len(subfolders) == 0: 
        sys.exit('請檢查資料集是否已經下載並放到根目錄下')
        
    for subfolder in subfolders:
        if not os.path.isdir(subfolder): continue
        img_files = glob.glob(os.path.join(subfolder, "*.jpg"))
        txt_files = get_txt_files_from_img_files(img_files)
        print('Load dataset:', subfolder)

        x = get_img_data(img_files, img_size, x)
        y = get_txt_data(txt_files, y)

    sf_list = split_shuffle(x.shape[0])
    train_x, test_x = split_train_test(x, 0.8, sf_list)
    train_y, test_y = split_train_test(y, 0.8, sf_list)
    
    return (train_x, train_y), (test_x, test_y)

def cnn_format(root, img_size):
    (train_x, train_y), (test_x, test_y) = rcnn_format(root, img_size)
    train_y = train_y[:, 0]
    test_y = test_y[:, 0]
    return (train_x, train_y), (test_x, test_y)
