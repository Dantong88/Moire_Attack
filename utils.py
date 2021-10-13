import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
# Standard libraries
import numpy as np
import os
# PyTorch
import torch
import torch.nn as nn
from PIL import Image

def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    
def image_folder_custom_label(root, transform, idx2label) :
    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    
    new_data = dsets.ImageFolder(root=root, transform=transform, 
                                 target_transform=lambda x : idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data

def create_dir(dir, print_flag = False):
    if not os.path.exists(dir):
        os.makedirs(dir)
        if print_flag:
            print("Create dir {} successfully!".format(dir))
    elif print_flag:
        print("Directory {} is already existed. ".format(dir))

def save_img(input_img, target_save_path):
    if isinstance(input_img, np.ndarray):
        pil_img = Image.fromarray(input_img.astype(np.uint8))
    else:
        pil_img = input_img
    pil_img.save(target_save_path, "JPEG", quality = 100)

def adjust_contrast_and_brightness(input_img, beta = 30):
    input_img = torch.clamp(input_img + beta, 0, 255)
    return input_img