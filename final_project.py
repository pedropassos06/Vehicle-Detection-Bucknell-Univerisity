import cv2
import numpy as np
from ECEG301 import VideoIdentification
from ECEG301 import FrameIdentification


#define global variables to be used throught the project
weights = "dnn_model/yolov4.weights"
config = "dnn_model/yolov4.cfg"
filename = "Videos/4K Road traffic video.mp4"
path = "WeekOfData/*.jpg"
KEY = '74X5ZXKXZLM3U0IQ'
vehicle_classes = [2, 3, 5, 7]

class ObjectDetection:
    
    
    def __init__(self):
        pass

    def detect_vehicles(self):
    
        '''
        detects vehicle in a video using sort to predict motion
        
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        '''        
        
        #create video identification object
        vi = VideoIdentification()
        
        #obtain model to be used
        model = vi.load_network(weights, config)
        
        #get classes and a random color for its boxes
        classes, colors = vi.create_classes()
        
        #start camera recording or run a video
        cap = vi.obtain_video(filename)
        

        while True:
            
            #load image
            success, img = cap.read()
            
            #ensure image is loaded
            if success is False:
                break
    
            #detect objects
            class_ids, scores, boxes = model.detect(img, nmsThreshold=0.4)
            
            for (class_id, score, box) in zip(class_ids, scores, boxes):
                
                #only looking for appropriate vehicles
                if class_id not in vehicle_classes:
                    continue
                
                #box cordinates
                x, y, w, h = box
                
                #obtain classes and boxes with distinct colors
                class_name = classes[class_id]
                color = colors[class_id]
                
                #bring score to a percentage
                score = round(score * 100)
                
                #obtain coordinates of bounding vbox and add information about objects above it
                cv2.putText(img, "{} {}".format(class_name.upper(), str(score) + "%"), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                
            
            #play video
            if vi.play_video(img) == 27:
                break
            
            cv2.waitKey(1)
            
        #destroy all windows after escape is pressed
        cv2.destroyAllWindows()


class WeeklyData:
    
    
    def __init__(self):
        
        self.total_vehicles = 0
        
    
    def analyze_folder(self):

        '''
        Loops through all images in a folder, detects cars in each image
        and send the number of vehicles in the lot to a thingspeak channel
        
        Parameters
        ----------
        None
    
        Returns
        -------
        None
        '''        
        
        #initialize frame identification object
        fi = FrameIdentification()
        
        #open folder with all images
        dataset_folder = fi.read_folder(path)
        
        #loop through all images
        for image_path in dataset_folder:
            
            print("Path:", image_path)
            
            #read image
            image = cv2.imread(image_path)
            
            #detect vehicles with boxes - put all boxes in a list
            bounding_boxes = fi.detect_vehicles(image, weights, config)
            
            #number of vehicles in the image = number of boxes
            vehicle_count = len(bounding_boxes)
            
            #keep track of total count
            self.total_vehicles += vehicle_count
            
            #loop through all boxes
            for box in bounding_boxes:
                
                #box coordinates
                x, y, w, h = box
                
                #draw rectangle around each vehicle
                cv2.rectangle(image, (x, y), (x + w, y + h), (25, 0, 180), 3)

                #add text informing vehicle's class
                cv2.putText(image, "Vehicles: " + str(vehicle_count), (20, 50), 0, 2, (100, 200, 0), 3)

            #display the image
            cv2.imshow("Snapshot", image)
            
            #write to thingspeak
            fi.write_to_thingspeak(str(vehicle_count), KEY)
            
            #wait a slight moment before going to the next image
            cv2.waitKey(1)

        print("Total current count", self.total_vehicles)




video = ObjectDetection()
video.detect_vehicles()


#Please do not uncomment this line unless you are using your own thingspeak channel
#edit line 12

#frame = WeeklyData()
#frame.analyze_folder()