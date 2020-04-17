#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:04:57 2018

@author:dongming.wei@sjtu.edu.cn 
"""
import numpy as np
import sys
import SimpleITK as sitk

def dice(FIX,MOVED,label):
    label=int(label)
    if label==0:
        fix=sitk.ReadImage(FIX)
        fix = sitk.ReadImage(FIX)
        moved = sitk.ReadImage(MOVED)
        
        fix_data = sitk.GetArrayFromImage(fix)
        moved_data = sitk.GetArrayFromImage(moved)
        MO = np.sum(moved_data[(fix_data>label) * (moved_data>label)])*2.0/(np.sum(fix_data[fix_data>label])
                + np.sum(moved_data[moved_data>label]))
        print(MO)
        return MO
    else:
        label=int(label)
        fix = sitk.ReadImage(FIX)
        moved = sitk.ReadImage(MOVED)
        
        fix_data = sitk.GetArrayFromImage(fix)
        moved_data = sitk.GetArrayFromImage(moved)
        if np.sum(fix_data[fix_data==label]) + np.sum(moved_data[moved_data==label]) ==0:
            print("[Warning]: No this label!")
        else:
            MO = np.sum(moved_data[(fix_data==label) * (moved_data==label)])*2.0/(np.sum(fix_data[fix_data==label])
                    + np.sum(moved_data[moved_data==label]))
            print(MO)
            return MO

def IoU(FIX,MOVED,label):
   label=int(label)
   fix = sitk.ReadImage(FIX)
   moved = sitk.ReadImage(MOVED)

   fix_data = sitk.GetArrayFromImage(fix)
   moved_data = sitk.GetArrayFromImage(moved)

   fix_data[fix_data!=label]=0
   fix_data[fix_data==label]=1
   
   moved_data[moved_data!=label]=0
   moved_data[moved_data==label]=1

   IoU=np.float32(np.sum(moved_data*fix_data))*1.0/((np.sum(moved_data)+np.sum(fix_data)-np.sum(moved_data*fix_data)))
   print(IoU)

if __name__== "__main__":
    print(sys.argv[1])
    dice(sys.argv[1],sys.argv[2],sys.argv[3])
    IoU(sys.argv[1],sys.argv[2],sys.argv[3])
