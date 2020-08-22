# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:43:48 2019

@author: Administrator
"""
from os import path
from skimage import io, color
from tqdm import trange,tqdm
import numpy as np
import nibabel as nib

                

class Cluster:
    def __init__(self, h, w ,d , l, a, b):
        self.h = h
        self.w = w
        self.d = d

        self.l = l
        self.a = a
        self.b =b
        self.pixels = []
        
        
    

class SLIC:
    def __init__(self,filename, n, m, k, p = 1/5000):
        self.n = n #N as super pixel num
        self.m = m # m as the maximum value of LAB distance
        self.k = k # K as iterate times
        self.p = p 
        '''
        if the number of pixels in cluster/total pixels number is below p, 
        the cluster needs to be combined to its nearest cluster.
        '''
        
    def start_SLIC(self):
        #get LAB in the form of np.ndarray, p_height, p_height, pixel_num
        self.get_lab(filename)
        
        self.S = int((self.pixel_num/self.n)**(1/3))
        
        
        #accelerate the calculation
        self.S2 = self.S**2  
        self.m2 = self.m**2
        
        
        self.clusters = []
        self.init_clusters()
        self.move_clusters()
        self.iterate()
        
        
    def get_lab(self, filename):
        im = nib.load(filename)
        #把仿射矩阵和头文件都存下来
        self.affine = im.affine.copy()
        self.hdr = im.header.copy()
        
        gray = im.get_fdata()
        rgb = color.gray2rgb(gray)
        rgb = rgb/rgb.max()  #归一化
        self.lab = color.rgb2lab(rgb)   
        self.p_height = im.shape[0]
        self.p_width = im.shape[1]
        self.p_depth = im.shape[2]
        self.pixel_num = self.p_height * self.p_width *self.p_depth
        
    
    def init_clusters(self):
        h = int(self.S /2)
        while(h < self.p_height ):
            w = int(self.S /2)
            while(w < self.p_width):
                d = int(self.S/2)
                while d<self.p_depth:
                    self.clusters.append(Cluster(h,w,d,self.lab[h][w][d][0],
                    self.lab[h][w][d][1],self.lab[h][w][d][2]))
                    d+=self.S
                w += self.S
            h += self.S
            
            
    
    def move_clusters(self):
        for k in range(len(self.clusters)):
            h = self.clusters[k].h
            w = self.clusters[k].w
            d = self.clusters[k].d
            min_gradient = self.get_gradient(h, w, d)
            for i in range(-1,2):
                if(h+i >= self.p_height): continue
                for j in range(-1,2):
                    if(w+j >= self.p_width): continue
                    for l in range(-1,2):
                        if(d+l >= self.p_depth): continue
                        current = self.get_gradient(h + i, w + j, d+l)
                        if  current<min_gradient:
                            min_gradient = current
                            self.clusters[k].h = h+i
                            self.clusters[k].w = w+j
                            self.clusters[k].d = d+l
                            self.clusters[k].l = self.lab[h+i][w+j][d+l][0]
                            self.clusters[k].a = self.lab[h+i][w+j][d+l][1]
                            self.clusters[k].b = self.lab[h+i][w+j][d+l][2]
            
        
        
    def get_gradient(self, h, w, d):
        if (w < self.p_width-1 and h < self.p_height-1 and d < self.p_depth-1):
            return abs(self.lab[h][w+1][d][0] - self.lab[h][w-1][d][0])\
                    +abs(self.lab[h][w+1][d][1] - self.lab[h][w-1][d][1])\
                    +abs(self.lab[h][w+1][d][2] - self.lab[h][w-1][d][2])\
                    +abs(self.lab[h+1][w][d][0] - self.lab[h-1][w][d][0])\
                    +abs(self.lab[h+1][w][d][1] - self.lab[h-1][w][d][1])\
                    +abs(self.lab[h+1][w][d][2] - self.lab[h-1][w][d][2])\
                    +abs(self.lab[h][w][d+1][0] - self.lab[h][w][d-1][0])\
                    +abs(self.lab[h][w][d+1][1] - self.lab[h][w][d-1][1])\
                    +abs(self.lab[h][w][d+1][2] - self.lab[h][w][d-1][2])
        else:
            return float('inf')
                
    
           
    
    def iterate(self):
        
        self.distance = np.full([self.p_height, self.p_width, self.p_depth],np.inf)
        self.match = {}
        
        for i in trange(self.k):               
            
            '''
            if distance, cluster.pixels, match are reset beofre every iteration
            the output is very strange. It seems many squares.
            
            
            self.distance = np.full([self.p_height, self.p_width],np.inf)
            for cluster in self.clusters:  # reset cluster.pixels as empty list
                cluster.pixels = []
            self.match = {}
            ''' 
           
            
            self.assign()
            self.renew_cluster()
            final_name = path.abspath('.') + '\\' +\
            'mprage,lab,n={n},m={m},k={k}.nii'.format(n = self.n,m =self.m,k=i)
            self.save_image(final_name)
        
        
            

    def assign(self):
        for cluster in self.clusters:
            for h in range(cluster.h- self.S, cluster.h + self.S):
                if(h<0 or h>= self.p_height): continue
                for w in range(cluster.w - self.S, cluster.w + self.S):
                    if(w<0 or w>=self.p_width):continue
                    for d in range(cluster.d - self.S, cluster.d + self.S):        
                        if(d<0 or d>=self.p_depth):continue
                        
                        D = self.cal_dis(h,w,d,cluster)   
                        
                        if(D <self.distance[h][w][d]):
                            self.distance[h][w][d] = D
                            
                            if (h,w,d) in self.match:
                                self.match[(h,w,d)].pixels.remove((h,w,d))
                                
                            self.match[(h,w,d)] = cluster
                            cluster.pixels.append((h,w,d))
                        
                       
        
    def cal_dis(self,h, w, d, cluster):
         
      return    ((self.lab[h][w][d][0] - cluster.l)**2 +\
                (self.lab[h][w][d][1] - cluster.a)**2 +\
                (self.lab[h][w][d][2] - cluster.b)**2 )*self.S2 +\
                 ((h - cluster.h)**2 +(w - cluster.w)**2 + (d - cluster.d)**2)*self.m2
                 

    def renew_cluster(self):
        for cluster in self.clusters:
            h_sum = 0
            w_sum = 0
            d_sum = 0
            l_sum = 0
            a_sum = 0
            b_sum = 0
            num = 0
            for pixel in cluster.pixels:
                h = pixel[0]
                w = pixel[1]
                d = pixel[2]
                h_sum += h
                w_sum += w
                d_sum += d
                l_sum += self.lab[h][w][d][0]
                a_sum += self.lab[h][w][d][1]
                b_sum += self.lab[h][w][d][2]
                num += 1
                  
            cluster.h = int(h_sum/num)
            cluster.w = int(w_sum/num)
            cluster.d = int(d_sum/num)
            cluster.l = l_sum/num
            cluster.a = a_sum/num
            cluster.b = b_sum/num

    
    def save_image(self,final_name):
        lab_copy = self.lab.copy()
        for cluster in self.clusters:
            for pixel in cluster.pixels:
                lab_copy[pixel[0]][pixel[1]][pixel[2]] = \
                [cluster.l, cluster.a, cluster.b]
            #lab_copy[cluster.h][cluster.w][cluster.d]=[0,0,0]
            

        #形成新的nii文件
       # new_rgb = color.lab2rgb(lab_copy)
       # new_gray = color.rgb2gray(new_rgb)
        new_nii = nib.Nifti1Image(lab_copy, self.affine, self.hdr)
        
        
        nib.save(new_nii, final_name)
        
        
if __name__ =='__main__':   # testbench
    filename = path.abspath('.') + '\\' + 'mprage_3T_bet_dr_new.nii'
    n, m, k  = 1000, 20, 10
    #由平面距离变为空间距离， 可能需要调小m的值
    #灰度图转化为lab, a,b的值都很小
    
    picture = SLIC(filename, n, m, k) 
    #N as super pixel num, 
    #m as max value of lab distance
    # K as iterate time
    picture.start_SLIC()