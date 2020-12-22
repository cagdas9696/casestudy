from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle as pkl
import random

folder = "raw/Class1_def/train/"
for i in range(600):
    pas = False
    while not pas:
        k = (i % 120) + 1
        img = cv2.imread(folder+str(k)+".png")
        
        with open(folder+str(k)+".txt", "r") as f:
            lines = f.readlines()
        bboxes = np.ndarray(shape=(len(lines), 5), dtype=float)
        for n, line in enumerate(lines):
            line = line.replace("\n", "")
            c, x1, y1, x2, y2 = list(map(float, line.split(" ")))
            bboxes[n] = [x1, y1, x2, y2, c]
        
        check = False
        while not check:
            try:
                img_, bboxes_ = RandomHorizontalFlip(0.5)(img.copy(), bboxes.copy())
                img_, bboxes_ = RandomTranslate(p=0.5, diff=True)(img_, bboxes_)
                if not (bboxes_[0][0] > 0 and bboxes_[0][1] > 0 and bboxes_[0][2] < img.shape[1] and bboxes_[0][3] < img.shape[0]):
                    continue
                img_, bboxes_ = RandomScale(p=0.5, diff=True)(img_, bboxes_)
                if not (bboxes_[0][0] > 0 and bboxes_[0][1] > 0 and bboxes_[0][2] < img.shape[1] and bboxes_[0][3] < img.shape[0]):
                    continue
                img_, bboxes_ = RandomRotate(p=0.5, angle=20)(img_, bboxes_)
                if not (bboxes_[0][0] > 0 and bboxes_[0][1] > 0 and bboxes_[0][2] < img.shape[1] and bboxes_[0][3] < img.shape[0]):
                    continue
                img_, bboxes_ = RandomShear(p=0.5, shear_factor=0.2)(img_, bboxes_)
                if not (bboxes_[0][0] > 0 and bboxes_[0][1] > 0 and bboxes_[0][2] < img.shape[1] and bboxes_[0][3] < img.shape[0]):
                    continue
                check = True
            except:
                break
        
        if not check:
            print(folder+str(k)+'_'+str(l)+".png")
            continue
        
        l = (i // 120) + 1
        name = str(k)+'_'+str(l)
        cv2.imwrite(folder+name+".png", img_)

        new_lines = list()
        for box in bboxes_:
            new_line = "{} {} {} {} {}\n".format(
                int(box[4]), int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            new_lines.append(new_line)
        
        with open(folder+name+".txt", "w") as f:
            f.writelines(new_lines)
        
        pas = True
