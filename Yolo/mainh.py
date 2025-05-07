import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import*
# Using YoloV8 pre trained model for object detection
model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# Opening video file named 'tf'
cap=cv2.VideoCapture('tf_1.mp4')

# Read the class names from text file	 
my_file = open("coco.txt", "r")
data = my_file.read()
# Split class names into a list
class_list = data.split("\n") 

count=0
# Initialize tracker instances for different types of vehicles
tracker=Tracker()
tracker1=Tracker()
tracker2=Tracker()
# y coordinate for Green Line(where we start our detection)
cy1=184
# y coordinate for Red Line
cy2=209
offset=8
# Dictionary for all classes( car truck bus)
upcar={}		# save unique id and current position of car
downcar={}
countercarup=[]		# save carid for crosscheck
countercardown=[]
downbus={}
counterbusdown=[]
upbus={}
counterbusup=[]
downtruck={}
countertruckdown=[]

# loop for processing video frames
while True:    
    ret,frame = cap.read()		# read the current frame from the video
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))		# Resize the frame for Consistent processing
   

# Use the YOLO model to detect objects in the frame
    results=model.predict(frame)
# Extract bounding box data from predictions
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    # list for car    
    list=[]
    # list1 for bus
    list1=[]
    #list2 for truck
    list2=[]
    for index,row in px.iterrows():

        # Rectangle frame coordinates 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
           list.append([x1,y1,x2,y2])			#appending the coordinates in the list
          
        elif'bus' in c:
            list1.append([x1,y1,x2,y2])
          
        elif 'truck' in c:
             list2.append([x1,y1,x2,y2])
            

    bbox_idx=tracker.update(list)
    bbox1_idx = tracker1.update(list1)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id1=bbox
        # center points cx3 cy3
        cx3=int(x3+x4)//2
        cy3=int(y3+y4)//2
        # if the car class touch the centre point of the green line we have detection->enter if
        if cy1<(cy3 + offset) and cy1>(cy3-offset):
            upcar[id1] = (cx3, cy3)	# save unique id and current position of car
        # Same car is crossing red line(crosscheck)
        if id1 in upcar:
            if cy2<(cy3 + offset) and cy2>(cy3-offset):
                cv2.circle(frame,(cx3,cy3),4,(255,0,0),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                if countercarup.count(id1) == 0:
                    countercarup.append(id1)
                    
######################### Car Down##############################
                    
        # if the car class touch the centre point of the red line we have detection->enter if
        if cy2<(cy3 + offset) and cy2>(cy3-offset):
            downcar[id1] = (cx3, cy3)	# save unique id and current position of car
        # Same car is crossing green line(crosscheck)
        if id1 in downcar:
            if cy1<(cy3 + offset) and cy1>(cy3-offset):
                cv2.circle(frame,(cx3,cy3),4,(255,0,255),-1)
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                cvzone.putTextRect(frame,f'{id1}',(x3,y3),1,1)
                if countercardown.count(id1) == 0:
                    countercardown.append(id1)
                    
############################## Up Bus ########################
    
    for bbox1 in bbox1_idx:
        x5, y5, x6, y6, id2 = bbox1
        cx4 = int(x5+x6)//2
        cy4 = int(y5+y6)//2
        if cy1<(cy4 + offset) and cy1>(cy4 - offset):
            upbus[id2] = (cx4,cy4)
        if id2 in upbus:
            if cy2<(cy4 + offset) and cy2>(cy4-offset):
                cv2.circle(frame,(cx4,cy4),4,(255,0,0),-1)
                cv2.rectangle(frame,(x5,y5),(x6,y6),(255,0,255),2)
                cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1)
                if counterbusup.count(id2) == 0:
                    counterbusup.append(id2)
                    
                    
#################################### Bus down ###################
        
        if cy2<(cy4 + offset) and cy2>(cy4 - offset):
            downbus[id2] = (cx4,cy4)
        if id2 in downbus:
            if cy1<(cy4 + offset) and cy1>(cy4-offset):
                cv2.circle(frame,(cx4,cy4),4,(255,0,255),-1)
                cv2.rectangle(frame,(x5,y5),(x6,y6),(255,0,0),2)
                cvzone.putTextRect(frame,f'{id2}',(x5,y5),1,1)
                if counterbusdown.count(id2) == 0:
                    counterbusdown.append(id2)
                    
# Creating a counter                 
    cv2.line(frame,(1,cy1),(1018,cy1),(0,255,0),2)	#Green Line
    cv2.line(frame,(3,cy2),(1016,cy2),(0,0,255),2)	#Red Line
    
    cup = len(countercarup)	#counting the number of car in the list
    cdown = len(countercardown)
    cbusup = len(counterbusup)
    cbusdown = len(counterbusdown)
    
    cvzone.putTextRect(frame,f'Left Lane(Car):-{cup}',(50,60),2,2)
    cvzone.putTextRect(frame,f'Right Lane(Car):-{cdown}',(50,160),2,2)
    cvzone.putTextRect(frame,f'LL(bus):-{cbusup}',(792,43),2,2)
    cvzone.putTextRect(frame,f'RL(bus):-{cbusdown}',(792,100),2,2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
