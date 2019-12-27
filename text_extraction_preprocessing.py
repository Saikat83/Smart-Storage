import numpy as np
import cv2 as cv
import glob
import h5py
def FindMean(img):
    dict1={}
    sum_b,sum_g,sum_r=[0,0,0]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val=img[i,j]
            sum_b+=val[0]
            sum_g+=val[1]
            sum_r+=val[2]
    dict1["b"]=sum_b/(img.shape[0]*img.shape[1])
    dict1["g"]=sum_g/(img.shape[0]*img.shape[1])
    dict1["r"]=sum_r/(img.shape[0]*img.shape[1])
    return dict1
w,h=[256,256]
new_img=[]
mean=[]
for image in glob.glob('C:/Users/User/Dropbox/Smart Storage/Combined/*.*'):
    img=cv.imread(image)
    (H,W)=img.shape[:2]
    img=cv.resize(img,(w,h))
    new_img.append(img)
    dict1=FindMean(img)
    li=[]
    li.append(dict1['r'])
    li.append(dict1['g'])
    li.append(dict1['b'])
    mean.append(li)
h5f=h5py.File('data.h5','w')
h5f.create_dataset('dataset_1',data=new_img)
h5f.create_dataset('dataset_2',data=mean)
h5f.close()
