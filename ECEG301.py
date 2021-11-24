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
from sort import *
import glob

class VideoIdentification:
    
    def __init__(self):
        
        #initialize parameters of time - timestamp function
        self.seconds = int(0)
        self.minutes = int(0)
        self.hours = int(0)
        
        #initialize list of specific vehicles - vehicle counting function
        self.cars = list()
        self.trucks = list()
        self.motorbikes = list()
    
        #initialize object tracker - object tracking function
        self.cars_tracker = Sort()
        self.trucks_tracker = Sort()
        self.motorbikes_tracker = Sort()

    
    def convert_to_timestamp(self):
        
        '''
        Converts hours, minutes, and seconds into timestamp
        
        Parameters
        ----------
        None
    
        Returns
        -------
        timestamp, a string that represents a time stamp of how much time has passed
    
        '''
        
        if self.seconds > 59:
            self.seconds = 0
            self.minutes += 1
        
        if self.minutes > 59:
            self.minutes = 0
            self.hours += 1
            
        self.seconds += 1       
        
        timestamp = str(self.hours) + ':' + str(self.minutes) + ':' + str(self.seconds)
        
        return timestamp
    
    def start_timer(self, number_of_spots, number_of_cars):
        '''
        stars a timer and sends information to thingspeak every 30 seconds
        
        Parameters
        ----------
        hnumber_of_spots : an integer
            total number of parking spots in the lot
        number_of_cars : an integer
            number of cars parked in the lot
    
        Returns
        -------
        None
    
        '''
        
        while True:
            
            #send a 30 to thingspeak if lot is not full
            if (self.seconds == 30 or self.seconds == 0) and self.number_of_spots > number_of_cars:
                self.write_to_thingspeak(30)
            
            #send a 0 to thingspeak if lot is full
            else:
                self.write_to_thingspeak(0)
            
            #convert to timestamp
            self.convert_to_timestamp()
            
            #wait one second
            sleep(1)
    
    
    def draw_contours(self, class_name, x, y, x2, y2, img):
        
        '''
        Draws contours on specific objects -> cars, trucks, motorbikes
        
        Parameters
        ----------
        class_name : a string
            contains class of object
        x, y, x2, y2 : integers
            represents top left and right bottom coordinate of bounding countor
        img : Numpy Array
            represents each frame of the video
    
        Returns
        -------
        None
        '''
        
        # Select only cars
        if class_name == "car":
            self.cars.append([x, y, x2, y2])
        elif class_name == "truck":
            self.trucks.append([x, y, x2, y2])
        elif class_name == "motorbike":
            self.motorbikes.append([x, y, x2, y2])
        
        # Update cars tracking
        if self.cars:
            cars_bbs_ids = self.cars_tracker.update(np.array(self.cars))
            for car in cars_bbs_ids:
                x, y, x2, y2, id = np.array(car, np.int32)
                cv2.putText(img, str( ), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 3)

        # Update trucks tracking
        if self.trucks:
            trucks_bbs_ids = self.trucks_tracker.update(np.array(self.trucks))
            for truck in trucks_bbs_ids:
                x, y, x2, y2, id = np.array(truck, np.int32)
                cv2.putText(img, str( ), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)

        # Update motorbikes tracking
        if self.motorbikes:
            motorbikes_bbs_ids = self.motorbikes_tracker.update(np.array(self.motorbikes))
            for motorbike in motorbikes_bbs_ids:
                x, y, x2, y2, id = np.array(motorbike, np.int32)
                cv2.putText(img, str( ), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 3)



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
    
    
    def write_to_thingspeak(self, data):

        '''
        Sends data to thingspeak channel
        
        Parameters
        ----------
        data : an integer
            represents the number of cars in the parking lot
    
        Returns
        -------
        an url that writes to our thingspeak channel
        '''             

        sleep(30)
        
        return urllib.request.urlopen('https://api.thingspeak.com/update?api_key=74X5ZXKXZLM3U0IQ&field1='+str(data))
    

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
        
        class_ids, confidences, boxes = model.detect(img, nmsThreshold=0.4)
        
        for class_id, confidence, box in zip(class_ids, confidences, boxes):
            
            #at least 50% confidence
            if confidence > 0.5 and class_id in vehicle_classes:
                
                vehicles_boxes.append(box)


        return vehicles_boxes