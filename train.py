# Imports necessory tools
import torch
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import time
from tqdm import tqdm
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse





def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./flowers/', help='the directory to pull information from')
    parser.add_argument('--arch', type=str, default='vgg16', help='architecture to utilize, choose one from CNN models: vgg16, alexent')
    parser.add_argument('--learning_rate', type=float, default=.001, help='the rate at which the model does its learning')
    parser.add_argument('--hidden_layer', type=int, default=4096, help='Dictates the hidden units for the hidden layer')
    parser.add_argument('--gpu', default='cuda', type=str, help='device where to run model: CPU vs. GPU')
    parser.add_argument('--epochs', type=int, default=5, help='number of cycles to train the model')
    parser.add_argument('--dropout', type=float, default=0.3, help='probability rate for dropouts')
    parser.add_argument('--save_dir', type=str, default='./checkpoint.pth', help='choose directory for saving')

    in_args = parser.parse_args()

    return in_args




data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#Define your transforms for the training, validation, and testing sets
mean =[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

data_transforms = {

    'training': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),

    'validation': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),


    'testing':transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        #transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

}

#Load the datasets with ImageFolder
datasets = {
    'training': datasets.ImageFolder(train_dir, transform = data_transforms['training']),

    'validation': datasets.ImageFolder(valid_dir, transform = data_transforms['validation']),

    'testing': datasets.ImageFolder(test_dir, transform = data_transforms['testing'])
    
    }

#Using the image datasets and the trainforms, define the dataloaders
dataloaders = {

    'training': DataLoader(datasets['training'], batch_size = 64, shuffle = True, num_workers = 4),

    'validation': DataLoader(datasets['validation'], batch_size = 32, shuffle = False),

    'testing': DataLoader(datasets['testing'], batch_size = 32, shuffle = False)

}





#Imports cat_to_name.json file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)





# Building and training network

device = torch.device("cuda" if in_args.gpu == "cuda" and torch.cuda.is_available() else "cpu")
print("device:{}".format(device))

the_pretrained_model = models.in_args.arch(pretrained=True)



for param in the_pretrained_model.parameters():
    param.requires_grad = False 



the_pretrained_model.classifier = nn.Sequential(
    nn.Linear(25088, in_args.hidden_layer),
    nn.ReLU(), 
    nn.Dropout(in_args.dropout),
    nn.Linear(in_args.hidden_layer, 102)

)

pretrained_model = the_pretrained_model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.classifier.parameters(), lr=in_args.learning_rate)


def train_one_epoch(model, dataloader, optimizer, device, criterion):
 
    model.train()
    running_loss = 0.0
    print("training on", device)

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(inputs.float())
        loss = criterion(outputs, labels)


        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)




def validating_model(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy




num_epochs = 5
best_val_loss = float('inf')
starting_time = time.time()

for epoch in range(num_epochs):
    

    print(f"Epoch {epoch+1}/{num_epochs}")


    train_loss = train_one_epoch(pretrained_model, dataloaders['training'], optimizer, device, criterion)


    val_loss, val_accuracy = validating_model(pretrained_model, dataloaders['validation'], device, criterion)


    print(f"Epoch number {epoch+1}/{num_epochs}, Training Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}")


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(pretrained_model.state_dict(), 'trained_model.pth')

time_end = time.time()
print(f'Training time: {time_end - starting_time} seconds')





# validation on the test set
pretrained_model.load_state_dict(torch.load('trained_model.pth'))
pretrained_model.eval()


# Testing the model
test_loss, test_accuracy = validating_model(pretrained_model, dataloaders['testing'], device, criterion)
print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy:.2f}")



#Saving the checkpoint 
pretrained_model.class_to_idx = datasets['training'].class_to_idx

checkpoint = {
    'arch': in_args.arch,
    'state_dict': pretrained_model.state_dict(),
    'class_to_idx': pretrained_model.class_to_idx,
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'learning_rate': in_args.learning_rate
}
torch.save(checkpoint, f'checkpoint_{test_accuracy:.2f}.pth')
