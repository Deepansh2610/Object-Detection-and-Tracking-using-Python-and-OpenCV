import cv2 #used for object detection
from tracker import *


#create tracker object

tracker = EuclideanDistTracker()
capture = cv2.VideoCapture("highway.mp4")

#object detection from stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40) #history = keeps past frames in memory(higher the number more accurate for stable cameras not for moving cameras), varThreshold = higher the number less detection but more accurate lower the value less accurate but more detection)

#To extract each frame from the video
while True:
    ret, frame = capture.read()
    height, width, _ = frame.shape
    print(height, width)

    #Extract region of interest
    region_of_interest = frame[200: 1200 ,500 : 850] #height(up: down), width(left: right)


    #1. object detection
    mask = object_detector.apply(region_of_interest) #applying white mask on the objects
    
    #cleaning the mask to remove detection of shadows
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) #keeping the values of the mask which are between 245 and 255 as 255 is white and towards 0 is black

    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #marking the object's boundries and then coloring them 
    detections = []
    for cnt in contours:
        #Ca;culate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 150:
            #cv2.drawContours(region_of_interest, [cnt], -1, (255, 0, 0), 2)
            #for making a box around the object
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(region_of_interest, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #print(x, y, w, h)

            detections.append([x, y, w, h])


    #2. Object tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(region_of_interest, str(id), (x, y -15), cv2.FONT_HERSHEY_PLAIN, 2, (0 , 0 ,0), 2) #Putting the tracking id and other info. 15px above the object
                   #In what region,  what string, (position of string on the video, font, size of string, color of string


        cv2.rectangle(region_of_interest, (x, y), (x + w, y + h), (255, 0, 0), 2)



    print(boxes_ids)

    print(detections)        
    cv2.imshow("ROI", region_of_interest)
    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27: #key to stop the program (s)
        break

capture.release()
capture.destroyAllWindows()

