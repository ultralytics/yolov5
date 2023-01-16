# RealSense D435i Camera 📷 by GTG
"""
Class to import the information of the camera
"""
import pyrealsense2 as rs
import numpy as np

class RealSense:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #Definincion de la configuración de la cámara
        self.h_color = 640 ; self.w_color = 480 ; self.fps_color = 60
        self.h_depth = 640 ; self.w_depth  = 480 ; self.fps_depth = 60
        self.config.enable_stream(rs.stream.color, self.h_color, self.w_color, rs.format.bgr8, self.fps_color)
        self.config.enable_stream(rs.stream.depth, self.h_depth , self.w_depth , rs.format.z16, self.fps_depth )
        #Comienzo del streaming
        profile = self.pipeline.start(self.config)  
        self.align_to = rs.stream.color 
        self.align = rs.align(self.align_to)
    
    def values(self):
        return self.h_color, self.w_color, self.fps_color  
    
    def get_frames(self):
        self.frames = self.pipeline.wait_for_frames()
        self.aligned_frames = self.align.process(self.frames)
        self.color_frame = self.aligned_frames.get_color_frame()
        self.depth_frame = self.aligned_frames.get_depth_frame()
        
        if not self.depth_frame or not self.color_frame:
            assert('Error: No se encuentran los frames de la camara D435i')
            
        self.color_image= np.asanyarray(self.color_frame.get_data())
        self.depth_image = np.asanyarray(self.depth_frame.get_data())         
   
        return self.color_image, self.depth_image
    
    
    def get_distance(self, x, y):
        #Obtención de la distancia
        if (x <= self.h_color and y <= self.w_color): #Se especifica para que no de error en la llegada de parametros invalidos
            # self.dist = self.depth_frame.get_distance(x + 4, y + 8)
            self.dist = self.depth_frame.get_distance(x, y)
            return self.dist
        else:
            return 0
    
    
    def release(self):
        self.pipeline.stop()
    
    