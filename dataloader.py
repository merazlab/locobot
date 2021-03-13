import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform
from PIL import Image 
import cv2
import ast
# conda install -c conda-forge scikit-imagey
from torch import Tensor
import torchvision

class LocoData(Dataset):
    def __init__(self, root):
        images, labels = [], []
        folders = os.listdir(root) 
    
        for folder in folders:
            folder_path = os.path.join(root, folder)
            img_path = os.path.join(folder_path, 'rgb.png')
            odom_path = os.path.join(folder_path, 'odom.txt')
            arr = []
            if os.path.exists(img_path) and os.path.exists(odom_path):
                f = open(odom_path, "r")
                list_str = f.readline()
                list_str = list_str.replace('[', '')
                list_str = list_str.replace(']', '')
                # f.close()
                for word in list_str.split():
                    arr.append(float(word))
                labels.append(arr)
                images.append(img_path)
        labels = np.array(labels)

        data = [(x, y) for x, y in zip(images, labels)]
        self.data = data

        # The number of items in the dataset
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data[index][0]
        img_as_img = Image.open(img_name)
        img_as_tensor = torchvision.transforms.ToTensor()(img_as_img)
        label = self.data[index][1]
        return img_as_tensor, label

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'dataset_2')
    print(DATA_DIR)
    train = LocoData(DATA_DIR)
