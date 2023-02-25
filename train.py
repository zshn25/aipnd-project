
# Author: Zeeshan Khan Suri

# Imports here
import os
import json
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms

from model_utils import initialize_model

print("Torch version:",torch.__version__, "\nTorchvision version:",torchvision.__version__)


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="data_dir", type=str)
parser.add_argument("--save_dir", help="save_dir", type=str, default="")
parser.add_argument("--arch", help="Choose a model architecture from torchvision models",
                    type=str, choices=["resnet", "alexnet", "vgg", "densenet", "squeezenet"], default="resnet")
parser.add_argument("--learning_rate", help="learning_rate", type=float, default=0.01)
parser.add_argument("--hidden_units", help="Num hidden units", type=int, default=128)
parser.add_argument("--epochs", help="Num epochs", type=int, default=15)
parser.add_argument("--gpu", help="Use GPU?", action='store_true')

args = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

batch_size = 16
num_workers = 4
epochs = args.epochs
print("Batch size:", batch_size, "num_workers:", num_workers)

device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# TODO: Define your transforms for the training, validation, and testing sets
common_transforms = transforms.Compose([
                                        transforms.Resize(size=224),
                                        transforms.CenterCrop(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])
train_transforms = transforms.Compose([
                                       transforms.RandomResizedCrop(size=224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomAffine(180),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                      ])
# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(valid_dir, transform=common_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=common_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers, pin_memory=True)

train_total = len(train_dataset)
val_total = len(val_dataset)
test_total = len(test_dataset)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
num_classes = len(cat_to_name.keys())

model = initialize_model(args.arch, num_classes, args.hidden_units, feature_extract=True, use_pretrained=True).to(device)

params_to_update = []
print("Params to learn:")
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)

print("**Model initialized with the new classifier. Starting training**")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params_to_update, lr=args.learning_rate, momentum=0.9)

train_losses = []
val_losses = []
for epoch in range(epochs):  # loop over the dataset multiple times
    # Running stats
    train_loss = 0.0
    val_loss = 0.0
    val_correct = 0
    train_correct = 0
    
    # Training loop
    model.train()
    for i, data in enumerate(train_dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()
            
    train_losses.append(train_loss/len(train_dataset))
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            
    val_losses.append(val_loss/len(val_dataset))
    
    # Print running stats after each epoch
    print((f'Epoch {epoch + 1}: train loss: {train_loss/len(train_dataset):.3f}',
    f'val loss: {val_loss/len(val_dataset):.3f}',
    f'train accuracy: {100 * float(train_correct) // len(train_dataset)}%',
    f'val accuracy: {100 * float(val_correct) // len(val_dataset)}%'))

print('Finished Training')
    
    
# TODO: Save the checkpoint 
model.class_to_idx = train_dataset.class_to_idx

torch.save({
            'model': model,
            'optimizer': optimizer,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_to_idx': train_dataset.class_to_idx,
            }, os.path.join(args.save_dir, "checkpoint.pt"))

print('Model saved at', os.path.join(args.save_dir, "checkpoint.pt"))
    
    
