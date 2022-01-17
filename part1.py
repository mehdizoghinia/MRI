#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 09:49:26 2021

@author: mehdi
"""



import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import kimimaro


swi = nib.load('swi.nii.gz').get_fdata() # 
tof = nib.load('tof.nii.gz').get_fdata() 

 # 230-275              364-417           234-171

cube1 = swi [237:282,80:148,171:234]  
tcube1 = tof[237:282,80:148,171:234] 

from skimage.util import invert

cube1 = np.add(invert(cube1), 200) 

from scipy import ndimage 
cube1 = ndimage.zoom(cube1, 2)
tcube1 = ndimage.zoom(tcube1, 2)

from scipy import ndimage
close_t = ndimage.binary_closing(nib.load('tof_thr.nii.gz').get_fdata(), structure=np.ones((5,5,5))).astype(np.float)#opening(c, structure=np.ones((2,2,2))).astype(int)



nib.save(nib.Nifti1Image(close_t.astype(np.float32), nib.load('swi.nii.gz').affine), 'tof_thr.nii.gz')

nib.save(nib.Nifti1Image(t.astype(np.float32), nib.load('tof_cube2.nii.gz').affine), 'tof_thr2.nii.gz')
nib.save(nib.Nifti1Image((nib.load('tof_cube2.nii.gz').get_fdata()>135).astype(np.float32), nib.load('tof_cube2.nii.gz').affine), 'tof_thr2.nii.gz')



thresh_tof = threshold_otsu(t)
thresh_swi = threshold_otsu(s)
print("tof-swi",thresh_tof,thresh_swi)

from skimage.filters import threshold_otsu

#-----thre

folders = ['sub-01','sub-02','sub-03','sub-06','sub-07','sub-10','sub-13']
folders = ['sub-01','sub-02','sub-03','sub-06','sub-07','sub-13']
 
#17 - 11 - 15 - 3-18-19-20

for i, folder in enumerate(folders):
    thresh_tof = threshold_otsu(tcube1)
    thresh_swi = threshold_otsu(cube1)

#------------- resize 
from skimage.transform import resize

base_shape = nib.load("sub-02/tof_thr.nii.gz").get_fdata().shape
for folder in folders:
    # print(folder, nib.load(folder + "/res_tof.nii.gz").get_fdata().shape)    
    resized_tof = resize( nib.load(folder + "/tof_thr.nii.gz").get_fdata(), base_shape)
    resized_swi = resize( nib.load(folder + "/swi_thr.nii.gz").get_fdata(), base_shape)
    nib.save(nib.Nifti1Image(resized_tof.astype(np.float32), nib.load(folder + '/swi.nii.gz').affine), folder + '/res_tof.nii.gz')
    nib.save(nib.Nifti1Image(resized_swi.astype(np.float32), nib.load(folder + '/swi.nii.gz').affine), folder + '/res_swi.nii.gz')
    print(folder, nib.load(folder + "/res_tof.nii.gz").get_fdata().shape,nib.load(folder + "/res_swi.nii.gz").get_fdata().shape)    


#----------------------
 
swif = nib.load(folders[4] + '/res_swi.nii.gz').get_fdata()# 
t = nib.load(folders[4] + '/res_tof.nii.gz').get_fdata()

'''t[np.where(t>0.1)] = 1   
swif[np.where(swif>0.1)] = 1   
'''
shape_x, shape_y, shape_z = swif.shape

kimimaro_skel_s = kimimaro.skeletonize(swif.astype(int) ,fix_borders=True)
skel_swi = kimimaro_skel_s[1]
vertices_swi = skel_swi.vertices.astype(int)
radius_swi = skel_swi.radii.astype(float)
    
kimimaro_skel_t = kimimaro.skeletonize(t.astype(int),fix_borders=True)
skel_tof = kimimaro_skel_t[1]
vertices_tof = skel_tof.vertices.astype(int)
radius_tof = skel_tof.radii.astype(float)
    
    #draw skeleton points
sk_swi = np.zeros([shape_x,shape_y,shape_z])
for vertex in vertices_swi:
    sk_swi[vertex[0]][vertex[1]][vertex[2]] = 1
    
sk_tof = np.zeros([shape_x,shape_y,shape_z])
for vertex in vertices_tof:
    sk_tof[vertex[0]][vertex[1]][vertex[2]] = 1      
    
    
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d.axes3d import Axes3D
%matplotlib auto 


# Twice as wide as it is tall.
fig = plt.figure(figsize=plt.figaspect(0.5))


#---- First subplot---plot swi
ax = fig.add_subplot(2, 2, 1, projection='3d')
cords = np.where(sk_swi)


pnt3d= ax.scatter(cords[0], cords[1], cords[2], c= radius_swi, marker='.')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_title("swi")
cbar=plt.colorbar(pnt3d)
cbar.set_label("diameter ")



#---- Second subplot---plot tof
ax = fig.add_subplot(2, 2, 2, projection='3d')
cords = np.where(sk_tof)


pnt3d= ax.scatter(cords[0], cords[1], cords[2], c= radius_tof, marker='.')

ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')

ax.set_title("tof")
cbar=plt.colorbar(pnt3d)
cbar.set_label("diameter ")


#---- 3rd subplot---plot tof
ax = fig.add_subplot(2, 2, 3, projection='3d')
cords = np.where(swif)


pnt3d= ax.scatter(cords[0], cords[1], cords[2],  marker='.')

ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')

ax.set_title("swi")
cbar=plt.colorbar(pnt3d)
cbar.set_label("diameter ")



#---- 3rd subplot---plot tof
ax = fig.add_subplot(2, 2, 4, projection='3d')
cords = np.where(t)


pnt3d= ax.scatter(cords[0], cords[1], cords[2],  marker='.')

ax.set_xlabel('X ')
ax.set_ylabel('Y ')
ax.set_zlabel('Z ')

ax.set_title("tof")
cbar=plt.colorbar(pnt3d)
cbar.set_label("diameter ")





plt.show()       
       
#-----------------------    

Z = np.zeros((1,4,2)) # the array containing  swi&tof cordination and diameters like below

for i , folder in enumerate(folders):
    
    swif = nib.load(folder + '/res_swi.nii.gz').get_fdata().astype(np.int_) # 
    t = nib.load(folder + '/res_tof.nii.gz').get_fdata().astype(np.int_)
    
    shape_x, shape_y, shape_z = swif.shape

    kimimaro_skel_s = kimimaro.skeletonize(swif ,fix_borders=True, dust_threshold=1000)
    skel_swi = kimimaro_skel_s[1]
    vertices_swi = skel_swi.vertices.astype(int)
    radius_swi = skel_swi.radii.astype(float)
    
    kimimaro_skel_t = kimimaro.skeletonize(t,fix_borders=True, dust_threshold=1000)
    skel_tof = kimimaro_skel_t[1]
    vertices_tof = skel_tof.vertices.astype(int)
    radius_tof = skel_tof.radii.astype(float)
    
    #draw skeleton points
    sk_swi = np.zeros([shape_x,shape_y,shape_z])
    for vertex in vertices_swi:
        sk_swi[vertex[0]][vertex[1]][vertex[2]] = 1
    
    sk_tof = np.zeros([shape_x,shape_y,shape_z])
    for vertex in vertices_tof:
        sk_tof[vertex[0]][vertex[1]][vertex[2]] = 1  
    
    
    import sklearn.neighbors as ne
    
    
    X = np.asanyarray(np.where(sk_swi)).T
    tree = ne.KDTree(X, leaf_size=2)  
    
    Y = np.asanyarray(np.where(sk_tof)).T       
    
    
    for i in range (0,Y.shape[0]):
        dist, ind = tree.query(Y[i:i+1], k=1)
        if dist < 6:
            if abs(radius_tof[i] - radius_swi[ind])<6:
                xx = np.zeros((1,4,2))
                xx[0,0:3,0] = X[ind] 
                xx[0,3,1] = radius_tof[i]
                xx[0,3,0] = radius_swi[ind]
                xx[0,0:3,1] = Y[i]
                print(Z)
                Z = np.append(Z, xx,axis= 0)
    Z = np.delete(Z, Z[0],axis= 0)

  
#----------------------bar chart
    
import math
step = 10
x = np.arange(math.floor(np.min(Z[:,1,1])), math.ceil(np.max(Z[:,1,1])),step)
error_swi = []
error_tof = []
num = []
mean_swi = [] #  containing all the swi 
mean_tof = [] #  containing all the tof
for i in x :
    
    swi_dia= Z[:,3,0][ np.where((Z[:,1,1]<i+step) & (Z[:,1,1]>= i)) ]
    tof_dia= Z[:,3,1][ np.where((Z[:,1,1]<i+step) & (Z[:,1,1]>= i)) ]
    
    mean_swi.append(np.mean(swi_dia))
    error_swi.append(np.std(swi_dia))
    num.append(swi_dia.shape[0])

    
    mean_tof.append(np.mean(tof_dia))  
    error_tof.append(np.std(tof_dia))             
print(num)
width = 3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_swi, width, label='swi', yerr=error_swi)
rects2 = ax.bar(x + width/2, mean_tof, width, label='tof' , yerr=error_tof)
# num_bar = ax.bar(x , num, width, label='num')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('mean')
ax.set_title('mean diameter in different y axis(coronal)')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

fig.tight_layout()

plt.show()   

#----------------------bar chart
    
import math
step = 3
x = np.add(np.arange(len(list_dia_s)),1)
error_swi = []
error_tof = []
num = []
mean_swi = [] #  containing all the swi 
mean_tof = [] #  containing all the tof
for i in range(len(list_dia_s)-1,-1,-1) :
    print(i)
    
    mean_swi.append(np.mean(list_dia_s[i]))
    error_swi.append(np.std(list_dia_s[i]))
    num.append(len(list_dia_s[i]))

    
    mean_tof.append(np.mean(list_dia_t[i]))  
    error_tof.append(np.std(list_dia_t[i]))             
print(num)
width =0.45  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_swi, width, label='swi', yerr=error_swi)
rects2 = ax.bar(x + width/2, mean_tof, width, label='tof' , yerr=error_tof)
# num_bar = ax.bar(x , num, width, label='num')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('mean')
ax.set_title('mean diameter in different y axis(coronal)')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

fig.tight_layout()

plt.show()   


#------draw
%matplotlib auto 

deep =17
#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(nrows=2, ncols=len(folders), sharex=True, sharey=True) 

for i , folder in enumerate(folders):
    t =  np.flip(nib.load(folder + "/tof_cube.nii.gz").get_fdata(), (1,2)) 
    s =   np.flip(nib.load(folder + "/swi_cube.nii.gz").get_fdata(), (1,2))
    axarr[0,i].set_title('Subject '+str(i+1))
    
    # axarr[0,i].imshow( t[deep,:,:])
    # axarr[1,i].imshow(np.flip(nib.load(folder + "/tof_thr.nii.gz").get_fdata(), (1,2)) [deep,:,:] )
    
    axarr[0,i].imshow( s[deep,:,:])
    axarr[1,i].imshow(np.flip(nib.load(folder + "/swi_thr.nii.gz").get_fdata(), (1,2)) [deep,:,:] )
    
    # nib.save(nib.Nifti1Image((s> (threshold_otsu(s) +20)).astype(np.float32), nib.load(folder + "/tof_cube.nii.gz").affine), folder+'/swi_thr.nii.gz')
    # nib.save(nib.Nifti1Image((t > threshold_otsu(t)).astype(np.float32),nib.load(folder + "/tof_cube.nii.gz").affine), folder+'/tof_thr.nii.gz')


# plt.subplots_adjust(wspace=0.05)
plt.show()  


#--------------------
list_dia_s = [[] for rows in range(7)]
list_dia_t = [[] for rows in range(7)]

for i , folder in enumerate(folders):
    t = nib.load(folder + '/tof_thr.nii.gz').get_fdata()
    s = nib.load(folder + '/swi_thr.nii.gz').get_fdata()
    #print(cut, "cut")
    cut = round (t.shape[1]/6)
    t2 = t[:,cut:-cut,:]
    s2 = s[:,cut:-cut,:]

    shape_x, shape_y, shape_z = s.shape

    kimimaro_skel_s = kimimaro.skeletonize(s2.astype(int) ,fix_borders=True,dust_threshold=5)
    skel_swi = kimimaro_skel_s[1]
    vertices_swi = skel_swi.vertices.astype(int)
    radius_swi = skel_swi.radii.astype(float)
        
    kimimaro_skel_t = kimimaro.skeletonize(t2.astype(int),fix_borders=True,dust_threshold=5)
    skel_tof = kimimaro_skel_t[1]
    vertices_tof = skel_tof.vertices.astype(int)
    radius_tof = skel_tof.radii.astype(float)
        
        #draw skeleton points
    sk_swi = np.zeros([shape_x,shape_y,shape_z])
    for i , vertex in enumerate(vertices_swi):
        sk_swi[vertex[0]][vertex[1]][vertex[2]] = radius_swi[i]
        
    sk_tof = np.zeros([shape_x,shape_y,shape_z])
    for i, vertex in enumerate(vertices_tof):
        sk_tof[vertex[0]][vertex[1]][vertex[2]] = radius_tof[i]
            
    list_s= np.amax(sk_swi,axis=(0,2))
    list_t= np.amax(sk_tof,axis=(0,2))
    
    aaa=np.array_split(list_s, 7)
    bbb=np.array_split(list_t, 7)
    
    for i,part in enumerate(aaa): 
        list_dia_s[i].extend(aaa[i]) 
        list_dia_t[i].extend(bbb[i]) 

for j in range(len(list_dia_s)):
    
    for i in range(len(list_dia_s[j])-1,-1,-1):
    
        if (list_dia_t[j][i] < list_dia_s[j][i]/2) or  (list_dia_s[j][i] < 3 ):
            #print(list_dia_t[j][i] , list_dia_s[0][i])
            list_dia_t[j].remove(list_dia_t[j][i])
            list_dia_s[j].remove(list_dia_s[j][i])    
    
    
    
    
#-----------------------
           
from skimage.measure import label   

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
    
import math
def get_r(sq):
    return  math.sqrt(sq/math.pi)    
    
plt.figure()
%matplotlib auto 

    #subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(len(folders),6) 
    
for i , folder in enumerate(folders):
    t = nib.load(folder + '/tof_thr3.nii.gz').get_fdata()
    s = nib.load(folder + '/swi_thr3.nii.gz').get_fdata()
    coronal = t.shape[1]
    
    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    
    axarr[i,0].imshow(mid_point(getLargestCC(s[:,:,round(coronal/4)])))
    axarr[i,1].imshow(mid_point(getLargestCC(s[:,:,round(coronal/2)])))
    axarr[i,2].imshow(mid_point(getLargestCC(s[:,:,round(coronal/4*3)])))

    
    
    axarr[i,3].imshow(mid_point(getLargestCC(t[:,:,round(coronal/4)])))
    axarr[i,4].imshow(mid_point(getLargestCC(t[:,:,round(coronal/2)])))
    axarr[i,5].imshow(mid_point(getLargestCC(t[:,:,round(coronal/4*3)])))

plt.show()  

    

def mid_point(img):
    x_center, y_center =  np.argwhere(img==1).sum(0)/np.sum(img)
    img[int(round(x_center)),int(round(y_center))] = 0
    return img







list_dia_s = [[] for rows in range(3)]
list_dia_t = [[] for rows in range(3)]

for folder in folders:
    t = nib.load(folder + '/tof_thr3.nii.gz').get_fdata()
    s = nib.load(folder + '/swi_thr3.nii.gz').get_fdata()
    coronal = t.shape[1]
    
    list_dia_s[0].append(get_r(np.sum(getLargestCC(s[:,:,round(coronal/4)]))))
    list_dia_s[1].append(get_r(np.sum(getLargestCC(s[:,:,round(coronal/2)]))))
    list_dia_s[2].append(get_r(np.sum(getLargestCC(s[:,:,round(coronal/4*3)]))))
    
    list_dia_t[0].append(get_r(np.sum(getLargestCC(t[:,:,round(coronal/4)]))))
    list_dia_t[1].append(get_r(np.sum(getLargestCC(t[:,:,round(coronal/2)]))))
    list_dia_t[2].append(get_r(np.sum(getLargestCC(t[:,:,round(coronal/4*3)]))))
    
    
    
    
pol = nib.load('swi_thr.nii.gz').get_fdata()
img = pol[:,round(pol.shape[1]/2),:]
x_center, y_center =  np.argwhere(img==1).sum(0)/np.sum(img)
print(round(x_center), y_center)
    

