import os
import sys
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from classification.model import ResnetTwoHead


def testClassifier(model_path="checkpoints/model_epoch_23.pth", img_path="data/test/class3_withdef/149.png"):
    labels_1 = ["class_id_1", "class_id_2", "class_id_3"]
    labels_2 = ["without_defect", "with_defect"]

    data_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    image = Image.open(img_path)
    image = image.convert('RGB')
    image = data_transforms(image)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    image = torch.autograd.Variable(image.unsqueeze(0))

    print("\nloading multi-output classification...")
    model = torch.load(model_path)
    model.to(device)
    output = model(image)

    label1_out=output['class'].to(device)
    label1_out = torch.argmax(label1_out, dim=1).unsqueeze(1)[0][0]
    label_class = labels_1[label1_out]

    label2_out=output['def'].to(device)
    label2_out = torch.sigmoid(label2_out)
    label2_out = torch.round(label2_out).to(torch.int)[0][0]
    label_def = labels_2[label2_out]

    print("Predictions: {} and {}\n".format(label_class, label_def))

    return [label1_out, label2_out]

if __name__=='__main__':
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    l1, l2 = testClassifier(model_path, img_path)
