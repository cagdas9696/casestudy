import sys
import torch
from detection.models import Yolov4Tiny
import cv2
from detection.tool.torch_utils import do_detect
from detection.tool.utils import load_class_names, plot_boxes_cv2


def testDetection(weightsfile="detection/weights_best.pth", img_path="data/test/class2_withdef/121.png"):
    model = Yolov4Tiny(n_classes=3, inference=True)
    print("loading model weights...")
    pretrained_dict = torch.load(weightsfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)
    if torch.cuda.is_available():
        model.cuda()
    img = cv2.imread(img_path)
    sized = cv2.resize(img, (512, 512))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(model, sized, 0.25, 0.45, True)
    class_names = load_class_names('detection/labels.txt')

    print("number of detections:", len(boxes))
    for box in boxes:
        class_id = box[0][-1]
        score = box[0][-2]
        print('type: %s, score: %f\n' % (class_names[class_id], score))
    plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)
    
    return boxes

if __name__=='__main__':
    weightsfile = sys.argv[1]
    img_path = sys.argv[2]
    boxes = testDetection(weightsfile, img_path)