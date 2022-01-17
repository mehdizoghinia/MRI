#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 20:21:08 2021

@author: mehdi
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import kimimaro

folders = ['sub-03','sub-11','sub-15','sub-17','sub-18','sub-19']




 # 268-256              289-329           274-241
 # 240-258              307-341           232-201
 # 275-261              313-349           241-220
 # 267-255              310-348           258-225
 # 265-257              302-341           244-219
 # 255-276              293-323           258-226

folders = ['sub-03','sub-01','sub-06','sub-15','sub-19','sub-10']


swi = nib.load('swi.nii.gz').get_fdata() # 
tof = nib.load('tof.nii.gz').get_fdata() 

 # 256-249              147-170           272-241
 # 263-257              132-157           265-240
 # 259-267              115-131           254-217
 # 249-258              128-150           279-240
 # 258-249              135-153           264-241
 # 252-258              147-166           284-255


cube1 = swi [254:263,362:384,240:279]  
tcube1 = tof [254:263,362:384,240:279] 

from skimage.util import invert

cube1 = np.add(invert(cube1), 200) 

nib.save(nib.Nifti1Image(cube1.astype(np.float32), nib.load('tof.nii.gz').affine), 'swi_cube3.nii.gz')
nib.save(nib.Nifti1Image(tcube1.astype(np.float32), nib.load('tof.nii.gz').affine), 'tof_cube3.nii.gz')





t = nib.load('tof_cube3.nii.gz').get_fdata() 
s = nib.load('swi_cube3.nii.gz').get_fdata() 

from scipy import ndimage
close_t = ndimage.binary_closing(getLargestCC(t>164).astype(np.float32), structure=np.ones((1,1,1))).astype(np.float)#opening(c, structure=np.ones((2,2,2))).astype(int)
nib.save(nib.Nifti1Image(close_t, nib.load('tof.nii.gz').affine), 'tof_thr3.nii.gz')

nib.save(nib.Nifti1Image(getLargestCC(t>90).astype(np.float32), nib.load('tof.nii.gz').affine), 'tof_thr2.nii.gz')


close_s = ndimage.binary_dilation(getLargestCC(s>151).astype(np.float32), structure=np.ones((1,1,1))).astype(np.float)#opening(c, structure=np.ones((2,2,2))).astype(int)
nib.save(nib.Nifti1Image(close_s, nib.load('tof.nii.gz').affine), 'swi_thr3.nii.gz')

close_s = ndimage.binary_closing(s>149, structure=np.ones((3,3,3))).astype(np.float)#opening(c, structure=np.ones((2,2,2))).astype(int)
nib.save(nib.Nifti1Image(getLargestCC(close_s).astype(np.float32), nib.load('tof.nii.gz').affine), 'swi_thr2.nii.gz')


#---------skel







from skimage.measure import label   

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC