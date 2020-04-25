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
    path_train="datasets/train"
    path_test="datasets/test"

    txtpath_train="datasets/human_train.txt"
    txtpath_test="datasets/human_test.txt"

    allfile_test = []
    allfile_train = []

    allfile_train = readFilename(path_train,allfile_train)
    allfile_test = readFilename(path_test,allfile_test)

    with open(txtpath_train,'a+') as fp:
        for name in allfile_train:
            name_seq = name.split('/')
            write_name = os.path.join(name_seq[-2] , name_seq[-1])
            fp.write("".join(write_name)+"\n")
    
    with open(txtpath_test,'a+') as fp:
        for name in allfile_test:
            name_seq = name.split('/')
            write_name = os.path.join(name_seq[-2] , name_seq[-1])
            fp.write("".join(write_name)+"\n")