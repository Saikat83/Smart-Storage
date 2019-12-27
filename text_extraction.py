from imutils.object_detection import non_max_suppression
import numpy as np
import h5py
import cv2
import pytesseract
import os
h5f=h5py.File('data.h5','r')
new_img=h5f['dataset_1'][:]
mean=h5f['dataset_2'][:]
h5f.close()
layers = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
east=cv2.dnn.readNet('frozen_east_text_detection.pb')
list1=['morning','good','ood','evening','']
list2=['happy','diwali','christmas','y wali','appy']
path1='C:/Users/User/Dropbox/Smart Storage/Test0'
path2='C:/Users/User/Dropbox/Smart Storage/Test1'
path3='C:/Users/User/Dropbox/Smart Storage/Others'
count1,count2,count3=[0,0,0]
for k in range(0,10):
        image=new_img[k]
        (H,W)=image.shape[:2]
        blob=cv2.dnn.blobFromImage(image,1.0,(W,H),mean[k],swapRB=True,crop=False)
        east.setInput(blob)
        (s,g)=east.forward(layers)
        (r,c)=s.shape[2:4]
        rec=[]
        prob=[]
        padding=0.0
        for i in range(0,r):
                sd=s[0,0,i]
                x0=g[0,0,i]
                x1=g[0,1,i]
                x2=g[0,2,i]
                x3=g[0,3,i]
                angles=g[0,4,i]
                for j in range(0,c):
                        if sd[j]<0.5:
                                continue
                        (oX,oY)=(j*4.0,i*4.0)
                        angle=angles[j]
                        cos=np.cos(angle)
                        sin=np.sin(angle)
                        h=x0[j]+x2[j]
                        w=x1[j]+x3[j]
                        endX=int(oX+(cos*x1[j])+(sin*x2[j]))
                        endY=int(oY-(sin*x1[j])+(cos*x2[j]))
                        startX=int(endX-w)
                        startY=int(endY-h)
                        rec.append((startX,startY,endX,endY))
                        prob.append(sd[j])
        boxes=non_max_suppression(np.array(rec),probs=prob)
        for (startX,startY,endX,endY) in boxes:
                cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),2)
        cv2.imshow("Text Detection",image)
        cv2.waitKey(0)
        (p1,p2,p3)=(0,0,0)
        for (startX,startY,endX,endY) in boxes:
            dX=int((endX-startX)*padding)
            dY=int((endY-startY)*padding)
            startX=max(0,startX-dX)
            startY=max(0,startY-dY)
            endX=min(W,endX+(2*dX))
            endY=min(H,endY+(2*dY))
            roi=image[startY:endY,startX:endX]
            pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
            c=("-l eng --oem 1 --psm 11")
            text=pytesseract.image_to_string(roi,config=c)
            if text.lower() in list1:
                    p1=1
            elif text.lower() in list2:
                    p2=1
            print(text)
        if p1==0 and p2==0:
                p3=1
        print(p1,p2,p3)
        if p1==1:
                count1+=1
                cv2.imwrite(os.path.join(path1,'image'+str(count1)+'.jpg'),image)
        elif p2==1:
                count2+=1
                cv2.imwrite(os.path.join(path2,'image'+str(count2)+'.jpg'),image)
        else:
                count3+=1
                cv2.imwrite(os.path.join(path3,'image'+str(count3)+'.jpg'),image)
