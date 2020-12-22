import sys
from test_classifier import testClassifier
from test_detection import testDetection

classification_model_path = sys.argv[1]
detection_weights_path = sys.argv[2]
input_image_path = sys.argv[3]

l1, l2 = testClassifier(classification_model_path, input_image_path)

if l2 == 1:
    print("Now running the detection model to find out where defection is...\n")
    boxes = testDetection(detection_weights_path, input_image_path)
