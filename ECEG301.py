# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:48:38 2021

@author: pccp0

This is a file that contains many helper functions.
"""

import cv2
import numpy as np
from time import sleep
import urllib.request
import glob

class VideoIdentification:
    
    def __init__(self):
        pass
    
    def load_network(self, weights, config):

        '''
        Loads the specific neural network using weights and configuration files
        
        Parameters
        ----------
        weights : a string
            file that contains the pre trained weights of the neural network
        config : a string
            file that contains the model architechture of the neural network
    
        Returns
        -------
        model, the overall model of the neural network that is to be used
        '''
        
        net = cv2.dnn.readNet(weights, config)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        #create model from deep learning network
        model = cv2.dnn_DetectionModel(net)
        
        #set input parameters
        model.setInputParams(size=(832, 832), scale=1/255)
        
        return model
    
    
    def create_classes(self):
        
        '''
        creates a list with all classes and also generates a random color to go with such class
        
        Parameters
        ----------
        None
    
        Returns
        -------
        classes, a list with all 80 classes that our algorithm can identify
        colors, a tuple that represents a random rgb color
        '''

        classes = []
        
        #loop through all class names that are in a text file
        with open("dnn_model/classes.txt", "r") as file_object:
        
            for class_name in file_object.readlines():
            
                class_name = class_name.strip()
                    
                #add all names to the list classes
                classes.append(class_name)

        #generate random color
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        
        return classes, colors 
    
    
    def obtain_video(self, filename = None):
        
        '''
        Loads video or opens webcam/camera
        
        Parameters
        ----------
        None if webcam is to be used
        
        filename : a string
            contains path to a video
    
        Returns
        -------
        classes, a list with all 80 classes that our algorithm can identify
        colors, a tuple that represents a random rgb color
        '''
        
        #if no file name, open camera
        if filename == None:
            cap = cv2.VideoCapture(0)
        
        else:
            cap = cv2.VideoCapture(filename)
            
        return cap
    
    
    def play_video(self, img):

        '''
        displays specific video or displays webcam view
        
        Parameters
        ----------
        img : an array
            contains the the frame of the video (will go through all frames)
    
        Returns
        -------
        27, an integer, if escape key is pressed
        '''        

        #play video with recognition
        cv2.imshow("Img", img)
        key = cv2.waitKey(1)
            
        #press esc key to exit
        if key == 27:
            return 27


class FrameIdentification:

    def __init__(self):
        pass
    
    def load_network(self, weights, config):        
        
        '''
        Loads the specific neural network using weights and configuration files
        
        Parameters
        ----------
        weights : a string
            file that contains the pre trained weights of the neural network
        config : a string
            file that contains the model architechture of the neural network
    
        Returns
        -------
        self.model, the overall model of the neural network that we are using
        '''
        
        network = cv2.dnn.readNet(weights, config)
        
        self.model = cv2.dnn_DetectionModel(network)
        
        self.model.setInputParams(size = (832, 832), scale = 1 / 255)
        
        return self.model
        
        
    def read_folder(self, path):
        
        '''
        Reads a folder with one or more images
        
        Parameters
        ----------
        path : a string
            represents the path to a folder that contains all the images we need to use    
        
        Returns
        -------
        dataset_folder, a list with all image paths from a folder
        '''        
        
        dataset_folder = glob.glob(path)
        
        return dataset_folder
    
    
    def write_to_thingspeak(self, data, KEY):

        '''
        Sends data to thingspeak channel
        
        Parameters
        ----------
        data : an integer
            represents the number of cars in the parking lot
        KEY : a string
            represents a thingspeak channel's write to feed key
    
        Returns
        -------
        an url that writes to our thingspeak channel
        '''             

        sleep(30)
        
        #simply write to thingspeak using channel feed url
        return urllib.request.urlopen('https://api.thingspeak.com/update?api_key='+KEY+'&field1='+str(data))
    

    def detect_vehicles(self, img, weights, config):
        
        '''
        Detects vehicles with at least 50% confidence
        
        Parameters
        ----------
        weights : a string
            file that contains the pre trained weights of the neural network
        config : a string
            file that contains the model architechture of the neural network
        img : Numpy Array
            represents each frame of the video
    
        Returns
        -------
        vehicle_boxes, a list with top left coordinate, width, and length, 
                       of every box to draw the contours
        '''    
    
        model = self.load_network(weights, config)
        
        # Allow classes containing Vehicles only - cars, motorcycles, buses, trucks
        vehicle_classes = [2, 3, 5, 7]
        
        vehicles_boxes = []
        
        #detect() gives us class id's, confidences, and also a set of bounding boxes
        class_ids, confidences, boxes = model.detect(img, nmsThreshold=0.4)
        
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            
            #at least 50% confidence
            if confidence > 0.5 and class_id in vehicle_classes:
                
                vehicles_boxes.append(box)


        return vehicles_boxes