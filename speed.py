import torch
import numpy as np
import cv2
from time import time



class tennisDetection:
    

    # distance from camera to object measured
    #Known_distance = 76.2  # centimeter
    Known_distance = 76  # centimeter
    # width of object in the real world or Object Plane
    #tennis/baseball: 7cm, basket: 30 cm, soccer: 27 cm
    Known_width = 7  # centimeter

    #the object width (pixel) of reference image 
    ref_image_object_width_int = 125

    #fonts = cv2.FONT_HERSHEY_COMPLEX
    initialTime = 0
    initialDistance = 0
    changeInTime = 0
    changeInDistance = 0


    listDistance = []
    listSpeed = []


    def FocalLength(self, measured_distance, real_width, width_in_rf_image):
        focal_length = (width_in_rf_image * measured_distance) / real_width
        return focal_length

    def Distance_finder(self, Focal_Length, real_object_width, object_width_in_frame):
        distance = (real_object_width * Focal_Length)/object_width_in_frame
        return distance


    def speedFinder(self, coveredDistance, timeTaken):

        speed = coveredDistance / timeTaken

        return speed


    def averageFinder(self, completeList, averageOfItems):

        # finding the length of list.
        lengthOfList = len(completeList)

        # calculating the number items to find the average of
        selectedItems = lengthOfList - averageOfItems

        # getting the list most recent items of list to find average of .
        selectedItemsList = completeList[selectedItems:]

        # finding the average .
        average = sum(selectedItemsList) / len(selectedItemsList)

        return average

    def speedEstimate(self, object_pixel_width, frame_in):
        Focal_length_found = self.FocalLength(self.Known_distance, self.Known_width, self.ref_image_object_width_int)
        #print("focal: " + str(Focal_length_found))
        #start_time  = time()
        #result = speedDetect.score_frame(frame)

        object_width_in_frame = object_pixel_width
        #print("object width in frame: " + str(object_width_in_frame))

        if object_width_in_frame != 0:
            Distance = self.Distance_finder(Focal_length_found, self.Known_width, object_width_in_frame)
            #print("distance: " + str(Distance))

            self.listDistance.append(Distance)
            averageDistance = self.averageFinder(self.listDistance, 2)
            #print("average distance: " + str(averageDistance))

            distanceInMeters = averageDistance/100

            if self.initialDistance != 0:
                changeInDistance = self.initialDistance - distanceInMeters
                #print("change in distance: " + str(changeInDistance))
            
                if(time() - self.initialTime == 0):
                    changeInTime = 0.01
                    #print("change in time: " + str(changeInTime))
                else:
                    changeInTime = time() - self.initialTime
                    #print("change in time: " + str(changeInTime))

                speed = self.speedFinder(coveredDistance=changeInDistance, timeTaken=changeInTime)

                self.listSpeed.append(speed)
                averageSpeed = self.averageFinder(self.listSpeed, 10)
                if averageSpeed < 0:
                    averageSpeed = averageSpeed * -1

                speedFill = int(45+(averageSpeed) * 130)
                if speedFill > 235:
                    speedFill = 235

                #cv2.line(frame, (45, 70), (235, 70), (0, 255, 0), 35)
                # speed dependent line
                #cv2.line(frame, (45, 70), (speedFill, 70), (255, 255, 0), 32)
                #cv2.line(frame, (45, 70), (235, 70), (0, 0, 0), 22)
                # print()
                cv2.putText(
                    frame_in, f"Speed: {round(averageSpeed, 2)} m/s", (50, 75), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 220), 2) 
                

                #print("AVERAGE SPEED: " + str(round(averageSpeed, 2)))

            #inital distance and time
            self.initialDistance = distanceInMeters
            #print(initialDistance)
            self.initialTime = time() 


