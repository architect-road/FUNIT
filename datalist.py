# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @File    :   mytest.py
# @Time    :   2019/12/01 12:16:37
# @Author  :   yinpeng

import time     
import os  
import shutil

 
def readFilename(path, allfile):
    filelist = os.listdir(path)
 
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            readFilename(filepath, allfile) #递归读取文件夹
        else:
            allfile.append(filepath)
    return allfile
 
 
 
if __name__ == '__main__':
    path_class_train="datasets/class/train"
    path_class_test="datasets/class/test"
    path_content_train="datasets/content/train"
    path_content_test="datasets/content/test"

    txtpath_class_train="datasets/human_train_class.txt"
    txtpath_class_test="datasets/human_test_class.txt"
    txtpath_content_train="datasets/human_train_content.txt"
    txtpath_content_test="datasets/human_test_content.txt"

    allfile_class_test = []
    allfile_class_train = []
    allfile_content_test = []
    allfile_content_train = []

    allfile_class_train=readFilename(path_class_train,allfile_class_train)
    allfile_class_test=readFilename(path_class_test,allfile_class_test)
    allfile_content_train=readFilename(path_content_train,allfile_content_train)
    allfile_content_test=readFilename(path_content_test,allfile_content_test)

    with open(txtpath_class_train,'a+') as fp:
        for name in allfile_class_train:
            name_seq = name.split('/')
            write_name = os.path.join(name_seq[-2] , name_seq[-1])
            fp.write("".join(write_name)+"\n")
    
    with open(txtpath_class_test,'a+') as fp:
        for name in allfile_class_test:
            name_seq = name.split('/')
            write_name = os.path.join(name_seq[-2] , name_seq[-1])
            fp.write("".join(write_name)+"\n")
    
    with open(txtpath_content_train,'a+') as fp:
        for name in allfile_content_train:
            name_seq = name.split('/')
            write_name = os.path.join(name_seq[-2] , name_seq[-1])
            fp.write("".join(write_name)+"\n")

    with open(txtpath_content_test,'a+') as fp:
        for name in allfile_content_test:
            name_seq = name.split('/')
            write_name = os.path.join(name_seq[-2], name_seq[-1])
            fp.write("".join(write_name)+"\n")