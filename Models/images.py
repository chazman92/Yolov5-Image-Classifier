import torch
from torchvision.utils import save_image
import numpy as np
import cv2
#import pafy
from matplotlib import pyplot as plt
from time import time
from datetime import datetime
import sys
import os
import copy


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, url):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        #Video Capture
        #date_time = datetime.now().strftime("%H_%M_%S")
        #out_file=f"/home/chuck/Models/{date_time}_Labeled_Video.avi"

        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        #self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.previous_time = time()
        self.delta_time = 0

        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        #play = pafy.new(self._URL).streams[-1]
        #assert play is not None
        #return cv2.VideoCapture(play.url) //this is for youtube
        #return cv2.VideoCapture(self._URL)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        #model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/chuck/Models/Yolov5SmallBest.pt', force_reload=True)
        #model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/chuck/Models/Yolov5MediumBest.pt', force_reload=True)

        # model.conf = 0.25  # NMS confidence threshold
        # model.iou = 0.45  # NMS IoU threshold
        # model.classes = 0   # Only pedestrian
        # model.multi_label = False  # NMS multiple labels per box
        # model.max_det = 1000  # maximum number of detections per image
        return model

    def score_frame(self, frame):
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

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, boolBoxes):
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
                if boolBoxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2) #y1-20 moves label up 20pixels

        return frame

    def export_image(self, frameOut):
        # Get the current time, increase delta and update the previous variable
        self.current_time = time()
        self.delta_time += self.current_time - self.previous_time
        self.previous_time = self.current_time
        
        path_date = datetime.now().strftime(f"%b_%d")
        path = f"/home/chuck/SDCard/savedImages/{path_date}/"
        #path = f"/home/chuck/SDCard/savedImages/{path_date}/" #Moved to SDCARD

        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist 
            print('trying to make directory here ... because dir does not exist')
            os.makedirs(path)


        # Check if 3 (or some other value) seconds passed
        if self.delta_time > 3:
            # Operations on image
            # Reset the time counter
            self.delta_time = 0
            date_time = datetime.now().strftime("%H_%M_%S")
            img_file=f"{path}{date_time}_image.png"
            cv2.imwrite(img_file, frameOut)
            #old PIL code
            #im = Image.fromarray(frameOut)
            #im.save(img_file)
            return
        else:
            return

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        #player = self.get_video_from_url() #not using youtube grap here.
        player = cv2.VideoCapture(self._URL)
        if not player.isOpened():
            print("Error: Could not open file: %s" % (self.URL))
            return

        assert player.isOpened()

        while player.isOpened():
            try:
                start_time = time()
                ret, frame = player.read()
                
                target = 4
                counter = 0 
                while True: #slow down FPS with target
                    if counter == target: 
                        ret, frame = player.read() 
                        counter = 0
                        break
                    else: 
                        ret = player.grab() 
                        counter += 1 

                assert ret
                
                results = self.score_frame(frame) #modified tuple look at funciton to get results array restult[x]

                #play video live
                #img = resultsx.render()
                #cv2.imshow('YOLO', np.squeeze(img))
                #cv2.resize(img, (800, 600))


                #plot boxes regardless of results?
                #frameOut = self.plot_boxes(results, frame, false) #last boolean is for plotting boxes or not boolBoxes

                results_string = ""
                info = results[2]
                line_count = 0
                if len(info) != 0:
                    for result in info:
                        confidence = result['confidence']
                        label = result['name']
                        if (confidence > .5):
                            frameRaw = copy.copy(frame) #makes a copy of frame before results applied
                        
                            #plot boxes only when results meet criteria
                            frameOut = self.plot_boxes(results, frame, False) #last boolean is for plotting boxes or not boolBoxes
                            results_string += f"{label} {round(confidence, 4)} \r\n"
                            #self.export_image(frameOut)  #exports every image
                            #cv2.imshow('YOLO', np.squeeze(frameOut))
                            if label == 'car' or label == 'truck':
                                #self.export_image(frameRaw) #Raw Frame Copy without results
                                self.export_image(frameOut) #Image with results plotted based on boolean in plot_boxes
                                #out.write(frameOut)
                            
                end_time = time()
                fps = 1/np.round(end_time - start_time, 3)
                results_string += f"FPS : {round(fps, 4)}"
                #The \x1b[f sequence moves the cursor to 1,1, and \x1b[J clears all content from the cursor position to the end of the screen.
                #sys.stdout.write("\x1b[f\x1b[J" + results_string + "\n")
                print(results_string)

            except (IOError, SyntaxError, AttributeError, AssertionError) as e:
                print('Bad file:', e)
                player.release()
                print('Released player and exit via try except')
                break

def run_analysis(path_to_video):
    while True:
        a = ObjectDetection(path_to_video)
        a()
        print('Delete Object and Restarted in run_analysis loop...')
        a.instance = None

        #key = cv2.waitKey(25)
        #if key == ord('q'):
        #    break

if __name__ == "__main__":
    path = 'rtsp://chazman:navy92@192.168.1.15:88/videoMain'
    #path = '/home/chuck/Models/Model3Chinatown.mp4'
    try:
        run_analysis(path)
    finally:
        print("Finally")





