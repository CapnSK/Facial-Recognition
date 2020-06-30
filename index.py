import keras_vggface
from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
from scipy.spatial import distance
import FaceOps


embedings = []
knownEmbeddings={}
#available options for model : vgg16 resnet50 senet50
model = VGGFace(model='resnet50',include_top=False, input_shape=(224, 224, 3), pooling='avg')

#make this boolean true when you want to extract FaceData from Data.
addFaceData=True
if addFaceData:
    for file in glob.glob(".\Data\*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        print("Extracting face of",person_name)
        Img = face = cv2.imread(file)
        FaceCoords = FaceOps.getCords(Img)
        FaceImg = FaceOps.getFaceImage(Img,FaceCoords)
        cv2.imwrite('.\FaceData\\'+person_name+'.jpg',FaceImg)


for file in glob.glob(".\FaceData\*"):
    person_name = os.path.splitext(os.path.basename(file))[0]
    face = cv2.imread(file, 1)
    faceReduced = cv2.resize(face,(224,224))
    faceReduced = np.array([faceReduced])
    knownEmbeddings[person_name]=model.predict(faceReduced)






cap = cv2.VideoCapture(0)
while True:
    ret,Img = cap.read()
    if ret:
        results = FaceOps.extractfaces(Img) 
        if results is not None and len(results) !=0:
            for i in range(len(results)):
                x1,y1,w,h = results[i]               
                x2,y2=x1+w,y1+h
                #cv2.rectangle(Img,(x,y),(x+w,y+h),(255,255,255),5)
                face = Img[y1:y2,x1:x2]
                if face.shape[0] == 0 or face.shape[1] ==0:
                    continue        
                faceReduced = cv2.resize(face,(224,224))
                faceReduced = np.array([faceReduced])
                embedings = model.predict(faceReduced)
                distances = []
                for name,knownEmbedding in knownEmbeddings.items():
                    d=distance.cosine(knownEmbedding,embedings)
                    distances.append([name,d])
                distances = sorted(distances,key=lambda x:x[1])
                #print(distances)
                if len(distances)>0 and distances[0][1]<=0.35:
                    cv2.putText(Img, text=distances[0][0], org=(x1,y1-50), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.1, color=(255,255,255), thickness = 3, lineType = cv2.LINE_AA)
                else:
                    cv2.putText(Img, text="unknown", org=(x1,y1-50), fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.1, color=(255,255,255), thickness = 3, lineType = cv2.LINE_AA)
        cv2.imshow('Title',Img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()