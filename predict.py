import torch
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os




parser = argparse.ArgumentParser()

parser.add_argument('--test_file', type=str, default='flowers/train/30/image_03460.jpg', help='Run the prediction function on a given image')
parser.add_argument('--json_file', type=str, default='cat_to_name.json', help='JSON file that maps the class values to other category names')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pth', help='a checkpoint file that load/build a pre-trained model from')
parser.add_argument('--topk', type=int, default=5, help='the top K classes with probabilities')
parser.add_argument('--gpu', default='cuda', type=str, help='device where to run model: CPU vs. GPU')

in_args = parser.parse_args()





test_file = in_args.test_file
json_file = in_args.json_file
checkpoint_file = in_args.checkpoint_file
topk = in_args.topk
gpu = in_args.gpu



# Checking if JSON file exists
if os.path.exists(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
else:
    print(f"Error: {json_file} not found.")






# Writing a function that loads a checkpoint and rebuilds the model

def load_checkpoint(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        arch = checkpoint['arch']
        hidden_layer = checkpoint['hidden_layer']
        dropout = checkpoint['dropout']
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("device:{}".format(device))


        rebuild_model = getattr(models, arch)(pretrained=False)
        input_features = rebuild_model.classifier[0].in_features
        rebuild_model.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_layer),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_layer, 102),
            nn.LogSoftmax(dim=1)
            )

        rebuild_model.load_state_dict(checkpoint['state_dict'])
        rebuild_model.class_to_idx = checkpoint['class_to_idx']
        rebuild_model.to(device)
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)

        nn.CrossEntropyLoss()
        optimizer = optim.Adam(rebuild_model.classifier.parameters(), lr=lr)

        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)    

        epoch = checkpoint.get('epoch', None)
        start_epoch = epoch if epoch is not None else 0
        print(f'Model loaded. Last trained epoch: {epoch}')


        return rebuild_model, optimizer, start_epoch
    
    
rebuild_model = load_checkpoint(checkpoint_file)








# image preprocessing function
def process_image(test_file):

    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    the_preprocessed_image = Image.open(test_file).convert("RGB")
    
    the_transformed_images = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    the_processed_image = the_transformed_images(the_preprocessed_image)
    
    return the_processed_image
    

#replace "image" with the image path
image_path = process_image(test_file)
print(image_path)
    




def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

imshow(process_image(test_file))








# prediction class function

def predict(test_file, model, topk=5):
    device = torch.device("cuda" if gpu == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    img = process_image(test_file)
    img = img.unsqueeze(0)
    img = img.float()
    
    with torch.no_grad():
        output = model.forward(img.cuda())

    probs = torch.exp(output)
    top_probs, top_indices = probs.topk(topk, dim = 1) 
    top_probs = top_probs.cpu().numpy().squeeze()
    top_indices = top_indices.cpu().numpy().squeeze()

    class_to_idx_inv = {v: k for k, v in model.class_to_idx.items()}
    mapped_classes = [class_to_idx_inv[idx] for idx in top_indices]

    #mapped_classes = list()
    return top_probs, mapped_classes


probs, classes = predict(test_file, rebuild_model)
print(probs)
print(classes)
print(f"Top {topk} classes and probabilities:")
