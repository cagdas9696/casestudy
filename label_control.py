#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: can
"""


import os
import cv2
import argparse


FLAGS = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('folder', type=str, default='', help='path to folder that contains images and labels')
    parser.add_argument('names', type=str, help='path to label names file')
    FLAGS = parser.parse_args()

    folder = FLAGS.folder
    if not folder.endswith('/'):
        folder = folder + '/'

    control_folder = folder+'control/'
    if not os.path.exists(control_folder):
        os.makedirs(control_folder)

    names_path = FLAGS.names
    with open(names_path, 'r') as f:
        label_names = [line.replace('\n','') for line in f.readlines()]


    image_list = os.listdir(folder+'images/')
    image_without_label = []
    for image in image_list:
        image_name = image.split('.png')[0]
        label_path = folder+'labels/'+image_name+'.txt'
        print(label_path)
        if os.path.exists(label_path):
            image = cv2.imread(folder+'images/'+image)
            height,width,_ = image.shape
            with open(label_path, 'r') as f:
                lines = f.readlines()
        
            for line in lines:
                bbox = line.split(' ')
                if float(bbox[1])<0 or float(bbox[2])<0 or float(bbox[1])>1 or float(bbox[2])>1 or float(bbox[3])<0 or float(bbox[4])<0:
                    lines = 'del'
                    break
                klas = bbox[0]
                x0 = int(width * float(bbox[1]))
                y0 = int(height * float(bbox[2]))
                w = int(width * float(bbox[3]))
                h = int(height * float(bbox[4]))
                x1 = int(x0 - w/2)
                y1 = int(y0 - h/2)
                
                thickness = 1+int(w/50.0)
                cv2.rectangle(image, (x1,y1),(x1 + w, y1 + h),(0,255,0),thickness)
                
                cv2.putText(image,
                    label_names[int(klas)],
                    (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 0, 255], 2)
    
    
    
            cv2.imwrite(control_folder+image_name+'.png', image)
        else:
            if not os.path.exists(folder+'unlabeled'):
                os.makedirs(folder+'unlabeled')
            os.rename(folder+'images/'+ image_name+'.png', folder+'unlabeled/'+ image_name+'.png')
