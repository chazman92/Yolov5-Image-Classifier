import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from time import time
from datetime import datetime
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/chuck/Models/Yolov5SmallBest.pt', force_reload=True)
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
#model.classes = 0   # Only pedestrian
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

url = 'rtsp://chazman:navy92@192.168.1.15:88/videoMain'
cap = cv2.VideoCapture(url)
#Video Capture
date_time = datetime.now().strftime("%H:%M:%S")
out_file=f"/home/chuck/Models/{date_time}_Labeled_Video.avi"
x_shape = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
y_shape = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
four_cc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(out_file, four_cc, 20, (x_shape, y_shape))

while cap.isOpened():
    start_time = time()
    ret, frame = cap.read()
    
    # Make detections 
    try:
        results = model(frame)
    except (IOError, SyntaxError, AttributeError) as e:
        print('Bad file:', e)
    #results.print()  
    #results.save()
    #results.show()
    #results.xyxy[0]
    # Results
    
    #print(results.pandas().xyxy[0])
    # info = results.pandas().xyxy[0]
    # info2 = results.pandas().xyxy[0].to_dict(orient = "records")
    # savedImagePath = "/home/chuck/SDCard/savedImages/"
    # if len(info2) != 0:
    #     for result in info2:
    #         confidence = result['confidence']
    #         label = result['name']
    #         # if (confidence > .5):
    #         #     print (label, confidence)
    #         #     #results = result.score_frame(frame)
    #         #     frame = result.plot_boxes(results, frame)
    #         #     out.write(frame) #only write frames where confidence >.5
    #         #     if (label == 'ModelX'):
    #         #         print('Found a ModelX')
    #         #     #     imgPath = f"{savedImagePath}{time.strftime("%Y%m%d-%H%M%S")}.png"
    #             #     isWritten = cv2.imwrite(imgPath, img) 
    #             #     if isWritten:
	#             #         print('The image is successfully saved.')
    img = results.render()
    cv2.imshow('YOLO', np.squeeze(img))
    end_time = time()
    fps = 1/np.round(end_time - start_time, 3)

    #print(f"Frames Per Second : {fps}")
    #out.write(frame) write every frame to avi

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()