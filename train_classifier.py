import os
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torch import optim
import torch.nn.functional as F
from torchvision import transforms

from classification.model import ResnetTwoHead


class DataGenerator(Dataset):
    def __init__(self, folder, transform=None):
        self.X = list()
        self.y_class = list()
        self.y_def = list()
        subfolders = os.listdir(folder)
        for subfolder in subfolders:
            image_list = [f for f in os.listdir(os.path.join(folder, subfolder)) if f.endswith(".png")]
            for image in image_list:
                path = os.path.join(folder, subfolder, image)
                self.X.append(path)

                cls_id, def_id = subfolder.split("_")

                if cls_id == "class1":
                    self.y_class.append(0)
                if cls_id == "class2":
                    self.y_class.append(1)
                if cls_id == "class3":
                    self.y_class.append(2)
                
                if def_id == "nodef":
                    self.y_def.append(0)
                if def_id == "withdef":
                    self.y_def.append(1)         
        
        self.transform=transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # image = cv2.imread(self.X[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # h,w,c = image.shape
        # image = image.reshape((c,h,w))
        image = Image.open(self.X[idx])
        image = image.convert('RGB')
        label1 = np.array([self.y_class[idx]]).astype('float')
        label2 = np.array([self.y_def[idx]]).astype('float')
        
        sample={'image': image, \
                'label_class': torch.from_numpy(label1), \
                'label_def': torch.from_numpy(label2)}
        
        #Applying transformation
        if self.transform:
            sample['image']=self.transform(sample['image'])
            
        return sample


data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
train_data = DataGenerator('data/train', transform=data_transforms)
test_data = DataGenerator('data/test', transform=data_transforms)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = ResnetTwoHead(n_classes = [3, 1], pretrained = True)
# model = model.to(device)
model = torch.load('checkpoints/model_last.pth')
model.cuda()

# Loss Functions
criterion_class = nn.CrossEntropyLoss().to(device)
criterion_def = nn.BCEWithLogitsLoss().to(device)
# Optimizer
optimizer = optim.SGD([{'params': model.parameters(), 'initial_lr': 0.01}], lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,10,20], gamma=0.1)

def train_model(model, criterion1, criterion2, optimizer, scheduler, n_epochs=25):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        # train the model #
        model.train()
        for batch_idx, sample_batched in enumerate(train_dataloader):
            # importing data and moving to GPU
            image, label1, label2 = sample_batched['image'].to(device),\
                                             sample_batched['label_class'].to(device),\
                                              sample_batched['label_def'].to(device)
            image = torch.autograd.Variable(image)
            label1 = torch.autograd.Variable(label1)
            label1_ = label1.squeeze().type(torch.LongTensor).to(device)
            label2 = torch.autograd.Variable(label2)
            label2_ = label2.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            output=model(image)
            label1_out=output['class'].to(device)
            label2_out=output['def'].to(device)
            # calculate loss
            loss1=criterion1(label1_out, label1_)
            loss2=criterion2(label2_out, label2_)   
            loss=loss1+loss2
            # back prop
            loss.backward()
            # grad
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            if batch_idx % 50 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
        # validate the model #
        model.eval()
        for batch_idx, sample_batched in enumerate(test_dataloader):
            image, label1, label2 = sample_batched['image'].to(device),\
                                             sample_batched['label_class'].to(device),\
                                              sample_batched['label_def'].to(device)
            image = torch.autograd.Variable(image)
            label1 = torch.autograd.Variable(label1)
            label1_ = label1.squeeze().type(torch.LongTensor).to(device)
            label2 = torch.autograd.Variable(label2)
            label2_ = label2.to(device)
            
            output = model(image)
            label1_out=output['class'].to(device)
            label2_out=output['def'].to(device)           
            # calculate loss
            loss1=criterion1(label1_out, label1_)
            loss2=criterion2(label2_out, label2_)
            loss=loss1+loss2
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model, 'checkpoints/model_epoch_{}.pth'.format(epoch))
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
        scheduler.step()
    # return trained model
    return model

def test_model(model):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for sample_batched in test_dataloader:
            image, label1, label2 = sample_batched['image'].to(device),\
                                             sample_batched['label_class'].to(device),\
                                              sample_batched['label_def'].to(device)
            image = torch.autograd.Variable(image)                                              
            output = model(image)
            label1_out=output['class'].to(device)
            label2_out=output['def'].to(device)
            label1_out = torch.argmax(label1_out, dim=1).unsqueeze(1).to(torch.float)
            label2_out = torch.sigmoid(label2_out)
            label2_out = torch.round(label2_out)
            print("a")
            # y_pred_tag = torch.round(y_test_pred)
            # y_pred_list.append(y_pred_tag.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

# trained_model = train_model(model, criterion_class, criterion_def, optimizer, scheduler, 25)


model = torch.load('checkpoints/model_epoch_23.pth')
model.cuda()
test_model(model)