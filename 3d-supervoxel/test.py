# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:22:59 2019

@author: Administrator
"""


from os import path
from skimage import io, color
from tqdm import trange,tqdm
import numpy as np
import nibabel as nib
filename = path.abspath('.') + '\\' + 'mprage_3T_bet_dr.nii'
im = nib.load(filename)
data = im.get_fdata()


#归一化
gray = data/data.max()*255
rgb = color.grey2rgb(gray)
rgb = rgb/rgb.max()
lab = color.rgb2lab(rgb)




'''
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        for k in range(im.shape[2]):
            if lab[i,j,k,1]>1:
                print(i,j,k,1,lab[i,j,k,1],sep=' ')
            if lab[i,j,k,2]>1:
                print(i,j,k,2,lab[i,j,k,2],sep=' ')
'''