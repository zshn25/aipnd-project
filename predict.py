
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

from PIL import Image

print("Torch version:",torch.__version__, "\nTorchvision version:",torchvision.__version__)


parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="image_path", type=str)
parser.add_argument("save_dir", help="checkpoint path", type=str)
parser.add_argument("--category_names", help="category_names", type=str, default="cat_to_name.json")
parser.add_argument("--top_k", help="Return top KK most likely classes", type=int, default=1)
parser.add_argument("--gpu", help="Use GPU?", action='store_true')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

common_transforms = transforms.Compose([
                                        transforms.Resize(size=256),
                                        torchvision.transforms.CenterCrop(size=(224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])

checkpoint = torch.load(args.save_dir, "cpu")
model = checkpoint['model'].to(device)
model.load_state_dict(checkpoint['model_state_dict'])
class_to_idx = model.class_to_idx

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    num_classes = len(cat_to_name.keys())

# Predict the class from an image file
image = Image.open(args.image_path)
image = common_transforms(image)[:3,:,:].unsqueeze(0)
logits = model(image)
logits, classes = logits.topk(args.top_k)
probs = torch.nn.functional.softmax(logits, dim=1)

probs = probs[0].tolist()
classes = classes[0].tolist()

# Print Top K classes and their probabilities
[print(cat_to_name[str(class_+1)], ": ", "{:.2f}".format(probs[i])) for i,class_ in enumerate(classes)]
