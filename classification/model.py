import torch
from torch import nn, Tensor
from .resnet import resnet18, conv1x1, conv3x3

class Classifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_planes: int,
        out_planes: int,
        stride: int = 1
        ) -> None:
        super(Classifier, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(out_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = conv3x3(out_planes, out_planes, stride)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_planes, n_classes)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



class ResnetTwoHead(nn.Module):
    def __init__(self, n_classes = [3, 1], pretrained = True):
        super(ResnetTwoHead, self).__init__()
        
        self.backbone = resnet18(pretrained=pretrained, backbone=True)
        self.classifier1 = Classifier(n_classes[0], 512, 1024, 1)
        self.classifier2 = Classifier(n_classes[1], 512, 1024, 1)

        for m in self.classifier1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.classifier2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        y1 = self.classifier1(x)
        y2 = self.classifier2(x)
        return {'class': y1, 'def': y2}

