# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:22:59 2019

@author: Administrator
"""


from os import path
from skimage import  transform
import nibabel as nib
filename = path.abspath('.') + '\\' + 'mprage_3T_bet_dr.nii'
im = nib.load(filename)
data = im.get_fdata()
shape_new = (int(im.shape[0]/3),int(im.shape[1]/3),int(im.shape[2]/3))
data_new = transform.resize(data,shape_new)
#此时data_new数据类型为float64
#需要转化为int16
#data_new = data_new.astype(np.int16)

#把仿射矩阵和头文件都存下来
affine = im.affine.copy()
hdr = im.header.copy()
 
#形成新的nii文件
new_nii = nib.Nifti1Image(data_new, affine, hdr)


filename_new = path.abspath('.') + '\\' + 'mprage_3T_bet_dr_new.nii'
#保存nii文件，后面的参数是保存的文件名
nib.save(new_nii, filename_new)


