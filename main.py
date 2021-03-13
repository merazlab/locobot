import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import LocoData
import matplotlib.pyplot as plt



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset_2')

train_dataset = LocoData(DATA_DIR)
print("Loaded data")

# Creating a dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=4)



def train():
	epoch_loss = 0
	for step, data in enumerate(train_loader):
		train_x, train_y = data
		print(train_x.shape)
		print(train_y.shape)
		# print(train_y)
		# exit()
train()