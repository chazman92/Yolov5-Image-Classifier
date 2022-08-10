import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
from time import time
from datetime import datetime
import os
import sys
import argparse

class TeslaDetection:
    def __init__(self, url, boolSaveImg, boolShowVid):
        self._URL = url
        self.boolSaveImages = boolSaveImg
        self.boolShowVideo = boolShowVid
        self.model = self.cls_load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.previous_time = time()
        self.delta_time = 0
        self.lastImageCoord = [0,0]
        self.currImageCoord = [0,0]
    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = cv2.VideoCapture(self._URL)
        while cap.isOpened():
            start_time = time()
            
            #slow down FPS with target
            fps = 4
            counter = 0 
            while True: 
                if counter == fps: 
                    ret, frame = cap.read() 
                    counter = 0
                    break
                else: 
                    ret = cap.grab() 
                    counter += 1 

            if not ret: # break if no frame is returned and restart
                break

            # Make detections 
            try:
                results = self.cls_score_frame(frame)
                frameOut = self.cls_plot_boxes(results, frame, False) #last boolean is for plotting boxes or not boolBoxes

                results_string = ""
                info = results[2]
                if len(info) != 0:
                    for result in info:
                        confidence = result['confidence']
                        label = result['name']
                        if (confidence > .5):
                            results_string += f"{label} {round(confidence, 4)} \r\n"
                            if label == 'car' or label == 'truck':
                                if self.boolSaveImages:
                                    if self.cls_check_repeat:
                                        self.cls_export_image(frameOut) #Image with results plotted based on boolean in plot_boxes

                                break #break out of for result loop if car found
                                
            except (IOError, SyntaxError, AttributeError) as e:
                print('Bad file:', e)
            
            if self.boolShowVideo:
                resized = self.cls_resize_image(frameOut, 50)
                cv2.imshow('YOLO', np.squeeze(resized))

            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)

            results_string += f"FPS : {round(fps, 4)}"
            #   The \x1b[f sequence moves the cursor to 1,1, and \x1b[J clears all content from the cursor position to the end of the screen.
            sys.stdout.write("\x1b[f\x1b[J" + results_string + "\n")
            #print(results_string)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    def cls_load_model(self):  
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        #model = torch.hub.load('ultralytics/yolov5', 'yolov7', pretrained=True)
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/chuck/Models/50EpochBest.pt', force_reload=True)
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/chuck/Models/Yolov5MediumBest.pt', force_reload=True)

        model.conf = 0.5  # NMS confidence threshold
        model.iou = 0.45  # NMS IoU threshold
        # model.classes = 0   # Only pedestrian
        model.multi_label = False  # NMS multiple labels per box
        #model.max_det = 1000  # maximum number of detections per image
        return model
    def cls_score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        info = results.pandas().xyxy[0].to_dict(orient = "records")
        return labels, cord, info, results
    def cls_export_image(self, frame):
        # Get the current time, increase delta and update the previous variable
        current_time = time()
        self.delta_time += current_time - self.previous_time
        self.previous_time = current_time
        path_date = datetime.now().strftime(f"%b_%d")
        path = f"/home/chuck/SDCard/savedImages/{path_date}/"

        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(path)

        # Check if 3 (or some other value) seconds passed
        if self.delta_time > 3:
            # Operations on image
            # Reset the time counter
            self.delta_time = 0
            date_time = datetime.now().strftime("%H_%M_%S")
            img_file=f"{path}{date_time}_image.png"
            cv2.imwrite(img_file, frame)
            self.lastImageCoord = self.currImageCoord
            return
        else:
            return
    def cls_plot_boxes(self, results, frame, boolBoxes):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results[0], results[1]
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                self.currImageCoord = [x1, y1]
                if boolBoxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.cls_class_to_label(labels[i]), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2) #y1-20 moves label up 20pixels

        return frame
    def cls_class_to_label(self, cls):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(cls)]
    def cls_resize_image(self, src_image, scale_percent):

        #calculate the 50 percent of original dimensions
        width = int(src_image.shape[1] * scale_percent / 100)
        height = int(src_image.shape[0] * scale_percent / 100)

        dsize = (width, height)
        resized_img = cv2.resize(src_image, dsize, interpolation=cv2.INTER_CUBIC)

        return resized_img
    def cls_check_repeat(self):
        buffer = 50
        if self.lastImageCoord[1] > self.currImageCoord[1+buffer] and self.lastImageCoord[1] < self.currImageCoord[1-buffer]:
            if self.lastImageCoord[2] > self.currImageCoord[2+buffer] and self.lastImageCoord[2] < self.currImageCoord[2-buffer]:
                print('Repeat Image')
                return False
        
        return True
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    #parser.add_argument("url_path", help="Path To Video", type=string)

    # Optional arguments
    parser.add_argument("-u", "--UrlPath", help="URL Path.", type=bool, default='rtsp://chazman:navy92@192.168.1.15:88/videoMain')       
    parser.add_argument("-s", "--saveImages", help="Bool Save Images.", type=bool, default=True)
    parser.add_argument("-v", "--showVideo", help="Bool Show Video", type=bool, default=True)

    # Parse arguments
    args = parser.parse_args()

    return args
def run_analysis(path_to_video):
    while True:
        a = TeslaDetection(path_to_video, True, True) #args (PathURL, boolSaveImages, boolShowVideo)
        a()
        print('Delete Object and Restarted in run_analysis loop...')
        a.instance = None

        key = cv2.waitKey(25)
        if key == ord('q'):
           break

##All Python scripts start here ....
if __name__ == "__main__":
    # Parse the arguments
    #args = parseArguments()

    path = 'rtsp://chazman:navy92@192.168.1.15:88/videoMain'
    #path = '/home/chuck/Models/Model3Chinatown.mp4'
    try:
        run_analysis(path)
    finally:
        print("Finally")