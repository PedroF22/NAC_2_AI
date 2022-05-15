#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

minKPMatch=10

sift=cv2.SIFT_create()

refImg=cv2.imread("aOuros.png",0)

refKP,refDesc = sift.detectAndCompute(refImg,None)

vc=cv2.VideoCapture("q1.mp4")

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:

    frameImg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    frameKP, frameDesc = sift.detectAndCompute(frameImg,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(refDesc,frameDesc, k=2) 

    goodMatch=[]
    for m,n in matches:
        if(m.distance < 0.75*n.distance):
            goodMatch.append(m)
       
    if(len(goodMatch)> minKPMatch):

        tp=[]
        qp=[]
        for m in goodMatch:
            qp.append(refKP[m.queryIdx].pt) 
            tp.append(frameKP[m.trainIdx].pt)
        tp,qp=np.float32((tp,qp))

        H,status=cv2.findHomography(qp,tp,cv2.RANSAC,3.0)
        
        h,w=refImg.shape

        refBorda=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        
        frameBorda=cv2.perspectiveTransform(refBorda,H)
        
        cv2.polylines(frame,[np.int32(frameBorda)],True,(0,255,0),5)
        cv2.putText(frame, 'CARTA DETECTADA', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(200,50,0),2,cv2.LINE_AA)
    else:
        print ("Não encontrado bom match - %d/%d"%(len(goodMatch),minKPMatch))

    cv2.imshow("resultado", frame)
    
    rval, frame = vc.read()

    key = cv2.waitKey(10)
    if key == 27:
        break

vc.release()
cv2.destroyAllWindows()