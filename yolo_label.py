# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:16:30 2020

@author: cagda
"""

import math
#60.24	  37.55	   0.65	  69.04	  87.60

def yolo_form(major_axis, minor_axis, angle, x_pos, y_pos):

    x2=int(major_axis* float(math.cos(angle)))
    y2=int(major_axis * float(math.sin(angle)))
    
    x1=int(minor_axis* float(math.sin(angle)))
    y1=int(minor_axis* float(math.cos(angle)))
    
    x1=abs(x1)
    x2=abs(x2)
    y1=abs(y1)
    y2=abs(y2)    

    if(x2>=x1):
        x_top= int(x_pos + x2)
        x_down= int(x_pos - x2)
        
    else:    
        x_top= int(x_pos + x1)
        x_down= int(x_pos - x1)
        
    if(y2>=y1):  
        y_top=int( y_pos + y2)
        y_down=int( y_pos - y2)
    else:
        y_top=int( y_pos + y1)
        y_down=int( y_pos - y1)
    if(x_down<0):
        x_downn=x_down
        x_down=1
    else:
        x_downn=x_down
        
    if(y_down<0):
        y_downn=y_down
        y_down=1
    else:
        y_downn=y_down
    
    return   x_down,y_down,x_top,y_top  #x1,y1,x2,y2 

    

        
        
        
        