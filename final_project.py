import cv2
import numpy as np
from sort import *
from ECEG301 import VideoIdentification
from ECEG301 import FrameIdentification


#define network and video
weights = "dnn_model/yolov4.weights"
config = "dnn_model/yolov4.cfg"
filename = "WeekOfData/IP95 - GWML Parking Lot - Camera - 01 (11_13_2021 9_00_00 AM EST).png"
path = "WeekOfData/*.jpg"

class ObjectDetection:
    
    
    def __init__(self, weights, config, filename):
        
        self.weights = weights
        
        self.config = config
        
        self.filename = filename
        
    
    def load_network(self, weights, config):
    
        
        net = cv2.dnn.readNet(weights, config)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        model = cv2.dnn_DetectionModel(net)
        
        model.setInputParams(size=(608, 608), scale=1/255)
        
        return model


    def create_classes(self):
        
        
        classes = []
        
        with open("dnn_model/classes.txt", "r") as file_object:
        
            for class_name in file_object.readlines():
            
                class_name = class_name.strip()
                    
                classes.append(class_name)

        
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        
        return classes, colors 

    
    def obtain_video(self, filename = None):
        
        #if no file name, open camera
        if filename == None:
            cap = cv2.VideoCapture(0)
        
        else:
            cap = cv2.VideoCapture(filename)
            
        return cap
    
    def play_video(self, img):
            
        #play video with recognition
        cv2.imshow("Img", img)
        key = cv2.waitKey(1)
            
        #press esc key to exit
        if key == 27:
            return 27

    def detect_vehicles(self):
        
        model = self.load_network(weights, config)
        classes, colors = self.create_classes()
        cap = self.obtain_video(filename)
        

        while True:
            # Load Image
            ret, img = cap.read()
            if ret is False:
                break
    
            # Detect Objects
            class_ids, scores, boxes = model.detect(img, nmsThreshold=0.4)
            for (class_id, score, box) in zip(class_ids, scores, boxes):
                
                #make box
                x, y, w, h = box
                x2 = x + w
                y2 = y + h
                
                #obtain classes and bound with distinct colors
                class_name = classes[class_id]
                color = colors[class_id]
                
                #bring score to a percentage
                score = round(score * 100)
                
                #obtain coordinates of bounding vbox and add information about objects above it
                cv2.putText(img, "{} {}".format(class_name.upper(), str(score) + "%"), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                
                #draw bounding boxes for each type of vehicle
                VideoIdentification.draw_contours(class_name, x, y, x2, y2, img)
                
            
            #play video
            if self.play_video(img) == 27:
                break
            
            cv2.waitKey(0)
            
        cv2.destroyAllWindows()


class WeeklyData:
    
    
    def __init__(self, weights, config, path):
        
        self.weights = weights
        
        self.config = config
        
        self.path = path
        
        self.total_vehicles = 0
        
    
    def analyze_folder(self):
        
        fi = FrameIdentification()
        
        dataset_folder = fi.read_folder(path)
        
        for image_path in dataset_folder:
            print("Path:", image_path)
            image = cv2.imread(image_path)
            
            bounding_boxes = fi.detect_vehicles(image, weights, config)
            vehicle_count = len(bounding_boxes)
            
            self.total_vehicles += vehicle_count
            
            for box in bounding_boxes:
                
                x, y, w, h = box

                cv2.rectangle(image, (x, y), (x + w, y + h), (25, 0, 180), 3)

                cv2.putText(image, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

            cv2.imshow("Snapshot", image)
            
            fi.write_to_thingspeak(str(vehicle_count))
            
            cv2.waitKey(1)

        print("Total current count", self.total_vehicles)


#video = ObjectDetection(weights, config, filename)
#video.detect_vehicles()

frame = WeeklyData(weights, config, path)
frame.analyze_folder()