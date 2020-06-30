import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#This function is not optimized for multiple faces
def getFaceImage(frame,FaceCoordinates):
    x,y,w,h = FaceCoordinates
    face_crop = frame[y:y+h,x:x+w]
    return face_crop

def draw_rectangle(frame,FaceCoord):
    x,y,w,h = FaceCoord
    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)
    return frame


def getCords(img,scale=1.2,singleFace = True):
    face_rects = extractfaces(img,singleFace=singleFace,scale=scale)
    x,y = 0,0
    XYList = []
    if face_rects is not None:
        if singleFace:
            return face_rects[0]
        else:
            for (x,y,w,h) in face_rects:
                XYList.append((x,y))
            return XYList
    else:
        return None


def extractfaces(img,scale = 1.3,write = False,path = 'Data/',singleFace = True):
    face_img = img.copy()
    gray = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray,scaleFactor = scale,minNeighbors = 3)
    if not singleFace:
        return face_rects
    face_rects = sorted(face_rects,key = lambda x: x[3])
    face_crop = None
    x,y,w,h = 0,0,0,0
    if len(face_rects) == 0:
        return None
    if len(face_rects) != 0:
        x,y,w,h = face_rects[-1]
        face_crop = img[y:y+h,x:x+w]
    if write:
        cv2.imwrite(path + str(time.time()) + '.jpg',face_crop)
    #print("Face Shape is " + str(face_crop.shape))
    return [(x,y,w,h)]

def showFace(img,decRatio = 1):
    img = cv2.resize(img,(img.shape[0]//decRatio,img.shape[1]//decRatio))
    while True:
        cv2.imshow('Title',img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
    
def putText(img,text,coords = (0,0),fonts = 3,thick = 3,colorn = (0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,text = text,org = (coords[0],coords[1]-50),fontFace = font,fontScale = fonts,color = colorn,thickness = thick,lineType = cv2.LINE_AA)
    return img

if __name__ == '__main__':
    pass