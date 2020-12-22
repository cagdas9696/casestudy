import cv2
import sys
import os

folder = sys.argv[1]
names = [f.replace(".png","") for f in os.listdir(folder+"images/")]
for name in names:
    image = cv2.imread(folder+"images/"+name+".png")
    height, width, _ = image.shape
    with open(folder+"label_x1y1x2y2/"+name+".txt", "r") as f:
        lines = f.readlines()
    new_lines = list()
    for line in lines:
        line = line.replace("\n", "")
        c, x1, y1, x2, y2 = list(map(float, line.split(" ")))
        x = (x2 + x1) / (2 * width)
        y = (y2 + y1) / (2 * height)
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        new_line = "{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(
            int(c), x, y, w, h)
        new_lines.append(new_line)

    with open(folder+"labels/"+name+".txt", "w") as f:
        f.writelines(new_lines)
