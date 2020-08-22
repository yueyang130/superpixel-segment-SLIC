# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 18:49:41 2019

@author: Yueyang
"""
from os import path
from skimage import io, color
from tqdm import trange
import numpy as np

class Cluster:
    def __init__(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.w =w
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
        
        self.S = int((self.pixel_num/self.n)**0.5)
        
        #accelerate the calculation
        self.S2 = self.S**2  
        self.m2 = self.m**2
        
        self.clusters = []
        self.init_clusters()
        self.move_clusters()
        self.iterate()
        
        
    def get_lab(self, filename):
        rgb = io.imread(filename)
        self.lab = color.rgb2lab(rgb)
        #self.xyz = color.rgb2xyz(rgb)
        self.p_height = rgb.shape[0]
        self.p_width = rgb.shape[1]
        self.pixel_num = self.p_height * self.p_width
        
    
    def init_clusters(self):
        h = int(self.S /2)
        while(h < self.p_height ):
            w = int(self.S /2)
            while(w < self.p_width):
                self.clusters.append(Cluster(h,w,self.lab[h][w][0],
                self.lab[h][w][1],self.lab[h][w][2]))
                w += self.S
            h += self.S
            
            
    
    def move_clusters(self):
        for k in range(len(self.clusters)):
            h = self.clusters[k].h
            w = self.clusters[k].w
            min_gradient = self.get_gradient(h, w)
            for i in range(-1,2):
                if(h+i >= self.p_height): continue
                for j in range(-1,2):
                    if(w+j >= self.p_width): continue
                    current = self.get_gradient(h + i, w + j)
                    if  current<min_gradient:
                        min_gradient = current
                        self.clusters[k].h = h+i
                        self.clusters[k].w = w+j
                        self.clusters[k].l = self.lab[h+i][w+j][0]
                        self.clusters[k].a = self.lab[h+i][w+j][1]
                        self.clusters[k].b = self.lab[h+i][w+j][2]
            
        
        
    def get_gradient(self, h, w):
        if (w < self.p_width-1 and h < self.p_height-1):
            return abs(self.lab[h][w+1][0] - self.lab[h][w-1][0])\
                    +abs(self.lab[h][w+1][1] - self.lab[h][w-1][1])\
                    +abs(self.lab[h][w+1][2] - self.lab[h][w-1][2])\
                    +abs(self.lab[h+1][w][0] - self.lab[h-1][w][0])\
                    +abs(self.lab[h+1][w][1] - self.lab[h-1][w][1])\
                    +abs(self.lab[h+1][w][2] - self.lab[h-1][w][2])
        else:
            return float('inf')
                
    
           
    
    def iterate(self):
        
        self.distance = np.full([self.p_height, self.p_width],np.inf)
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
            'lenna,n={n},m={m},k={k}.jpg'.format(n = self.n,m =self.m,k=i)
            self.save_image(final_name)
        
        
            

    def assign(self):
        for cluster in self.clusters:
            for h in range(cluster.h- self.S, cluster.h + self.S):
                if(h<0 or h>= self.p_height): continue
                for w in range(cluster.w - self.S, cluster.w + self.S):
                    if(w<0 or w>=self.p_width):continue
                
                    D = self.cal_dis(h,w,cluster)
                    
                    if(D <self.distance[h][w]):
                        self.distance[h][w] = D
                        
                        if (h,w) in self.match:
                            self.match[(h,w)].pixels.remove((h,w))
                            
                        self.match[(h,w)] = cluster
                        cluster.pixels.append((h,w))
                        
                       
        
    def cal_dis(self,h, w ,cluster):
         
      return    ((self.lab[h][w][0] - cluster.l)**2 +\
                (self.lab[h][w][1] - cluster.a)**2 +\
                (self.lab[h][w][2] - cluster.b)**2 )*self.S2 +\
                 ((h - cluster.h)**2 +(w - cluster.w)**2)*self.m2
                 

    def renew_cluster(self):
        for cluster in self.clusters:
            h_sum = 0
            w_sum = 0
            l_sum = 0
            a_sum = 0
            b_sum = 0
            num = 0
            for pixel in cluster.pixels:
                h = pixel[0]
                w = pixel[1]
                h_sum += h
                w_sum += w
                l_sum += self.lab[h][w][0]
                a_sum += self.lab[h][w][1]
                b_sum += self.lab[h][w][2]
                num += 1
                  
            cluster.h = int(h_sum/num)
            cluster.w = int(w_sum/num)
            cluster.l = l_sum/num
            cluster.a = a_sum/num
            cluster.b = b_sum/num
            
            

    
    def save_image(self,final_name):
        lab_copy = self.lab.copy()
        for cluster in self.clusters:
            for pixel in cluster.pixels:
                lab_copy[pixel[0]][pixel[1]] = \
                [cluster.l, cluster.a, cluster.b]
            lab_copy[cluster.h][cluster.w]=[0,0,0]
            cluster.l=0
            
        
        rgb = color.lab2rgb(lab_copy)
        io.imsave(final_name, rgb)
        
    
    
                
                
        
             
        

if __name__ =='__main__':   # testbench
    filename = path.abspath('.') + '\\' + 'lenna.jpg'
    n, m, k  = 1000, 30, 10
    
    
    picture = SLIC(filename, n, m, k) 
    #N as super pixel num, 
    #m as max value of lab distance
    # K as iterate time
    picture.start_SLIC()