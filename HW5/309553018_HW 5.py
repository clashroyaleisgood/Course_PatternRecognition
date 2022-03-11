#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Hyper Param
BatchSize = 500
LearningRate = 0.001
EpochCounts = 10
# loss_func = nn.NLLLoss()
MODELPATH = 'Myresnet.pt'

# ## Load data

x_train = np.load(r"dataset/x_train.npy")   # 50000, 32, 32, 3
y_train = np.load(r"dataset/y_train.npy")   # 50000, 1
train_data_size = 50000
# x_train = np.rollaxis(x_train, 3, 1)      # 50000, 3, 32, 32
y_train = y_train.squeeze()
print(f'Shape: {y_train.shape}')

# for i in range(10):
#     plt.figure(figsize=(1.5, 1.5))
#     plt.imshow(x_train[i])
#     plt.show()

x_test = np.load(r"dataset/x_test.npy")     # 10000, 32, 32, 3
y_test = np.load(r"dataset/y_test.npy")     # 10000, 1
y_test = y_test.squeeze()
# x_test = np.rollaxis(x_test, 3, 1)          # 10000, 3, 32, 32

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# It's a multi-class classification problem
class_index = {
    'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
}
print(np.unique(y_train))

# ## Data preprocess

preprocess = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.Resize(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])
# input_tensor = preprocess(x_train[4])
# plt.imshow(input_tensor.permute(1, 2, 0))
# plt.show()
# print()


class MyDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform:
            img = self.transform(img)
        return img, self.label[index]

    def __len__(self):
        return len(self.data)

TransformTrain = MyDataset(x_train, y_train, transform=preprocess)
TransformTest = MyDataset(x_test, y_test, transform=preprocess)
TrainDataLoader = DataLoader(
    dataset=TransformTrain, batch_size=BatchSize, shuffle=True
)
TestDataLoader = DataLoader(
    dataset=TransformTest, batch_size=BatchSize, shuffle=False
)

num_classes = 10

# ## Build model & training (Keras)
# import warnings
# warnings.filterwarnings('ignore')

# Builde model
resnet50 = None
if os.path.isfile(MODELPATH):
    print(f'Load Model from {MODELPATH}')
    resnet50 = torch.load(MODELPATH)
else:
    resnet50 = models.resnet50(pretrained=True)
    for param in resnet50.parameters():
        param.requires_grad = False

    fc_inputs = resnet50.fc.in_features
    resnet50.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
    )
resnet50 = resnet50.to('cuda')

optimizer = optim.Adam(resnet50.parameters(), lr=LearningRate)

# Compile the model with loss function and optimizer


def train_model(model, train_data, loss_function, optimizer, epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'to device: {device}')
    history = []

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            if i % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {i+1}/100')
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 因為這裡的梯度是累加的，所以每次記得清零
            optimizer.zero_grad()

            outputs = model(inputs)
            # outputs = torch.argmax(outputs, 1) # one hot to normal encoding
            # outputs = outputs.view([-1, 1])
            # outputs = outputs.to(torch.int64)

            # labels = labels.to(torch.int64)
            # labels = labels.squeeze()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            train_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size

        # avg_valid_loss = valid_loss/valid_data_size
        # avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_train_acc])

        print(f'Epoch: {epoch+1}, Loss: {avg_train_loss:.4f}, \
                Accuracy: {avg_train_acc*100:.4f}')

        torch.save(model, MODELPATH)
    return model, history


def test_model(model, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred = []
    for i, (inputs, _) in enumerate(test_data):
        if i % 5 == 0:
            print(f'Batch: {i+1}/20')
        inputs = inputs.to(device)
        # labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        ret, predictions = torch.max(outputs.data, 1)
        y_pred += [predictions.cpu().numpy()]

    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape((-1))

    # ## DO NOT MODIFY CODE BELOW!
    # please screen shot your results and post it on your report

    assert y_pred.shape == (10000,)

    y_test = np.load("dataset/y_test.npy")
    print("Accuracy of my model on test-set: ", accuracy_score(y_test, y_pred))

# Train or Test

trained_model, history = train_model(
    model=resnet50,
    train_data=TrainDataLoader,
    loss_function=nn.NLLLoss(),
    optimizer=optimizer,
    epochs=EpochCounts
)

test_model(resnet50, TestDataLoader)

'''
    train 1 -> Epochs: 5, BatchSize: 500
    train 2 -> Epochs: 5, BatchSize: 500
    train 3 -> Epochs: 10, BatchSize: 500
    train 4 -> Epochs: 10, BatchSize: 500
'''
