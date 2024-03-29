# -*- coding: utf-8 -*-
**QMNIST**
"""

# import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load Fashion QMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.QMNIST(root='./data', train=True, download=True, transform=transform) # load the training data
test_dataset = torchvision.datasets.QMNIST(root='./data', train=False, download=True, transform=transform) # load the test data

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# labels for the data
labels_map = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}

# display the training data
figure = plt.figure(figsize=(10, 10))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

# display the test data
figure = plt.figure(figsize = (8,8))
cols, rows = 3, 3
for i in range(1, cols * rows +1):
    sample = torch.randint(len(test_dataset), size = (1,)).item()
    img, label = test_dataset[sample]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap = "gray")
plt.show()

"""**Build a simple neural network**"""

#Define the neural network architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network
model = MLP()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model on test data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: { correct / total * 100}%')

# Evaluate the model on training data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on train set: { correct / total * 100}%')

# Evaluate the model and store predictions
model.eval()
predictions = []
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

import random
import matplotlib.pyplot as plt

# Get a random index for the test data
random_index = random.randint(0, len(test_dataset) - 1)

# Get the image and label at the random index
image, label = test_dataset[random_index]

# Reshape the image tensor to a 28x28 shape
image = image.view(28, 28)

# Convert the image tensor to a numpy array for visualization
image_numpy = image.numpy()

# Assuming 'predictions' contains the predicted labels for the test data
predicted_label = predictions[random_index]

# Display the image with both predicted and actual labels
plt.imshow(image_numpy, cmap='gray')
plt.title(f'Predicted Label: {predicted_label}, Actual Label: {label}')
plt.axis('off')
plt.show()

"""Hypothesiz Changes:

Modification 1: We add another dense layer of 128 nodes.

Effect: Adding another layer makes the network more dense. This has potential positive and negative effects. A deeper network enhances its capacity to learn complex representations. It can capture more finer details in the input data and also better represent the underlying function.

But there is risk of the network overfitting.
"""

class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)  # New layer with 128 nodes
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the neural network
model1 = MLP1()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001, weight_decay=0.001)

# Train the neural network
num_epochs = 5
for epoch in range(num_epochs):
    model1.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model
model1.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: { correct / total * 100}%')

# Evaluate the model
model1.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on train set: { correct / total * 100}%')

# Evaluate the model and store predictions
model1.eval()
predictions1 = []
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model1(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions1.extend(predicted.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

import random
import matplotlib.pyplot as plt

# Get a random index for the test data
random_index = random.randint(0, len(test_dataset) - 1)

# Get the image and label at the random index
image, label = test_dataset[random_index]

# Reshape the image tensor to a 28x28 shape
image = image.view(28, 28)

# Convert the image tensor to a numpy array for visualization
image_numpy = image.numpy()

# Assuming 'predictions' contains the predicted labels for the test data
predicted_label1 = predictions1[random_index]

# Display the image with both predicted and actual labels
plt.imshow(image_numpy, cmap='gray')
plt.title(f'Predicted Label: {predicted_label1}, Actual Label: {label}')
plt.axis('off')
plt.show()

"""Though we assumed adding another dense layer will increase the performance of the model drastically, we observed that there was no significant improvement in the performance of the model.

# **Experimentation.**

1. Model with SGD Optimizer
"""

#Define the neural network architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network
model = MLP()

model2 = MLP()
optimizer2 = optim.SGD(model2.parameters(), lr=0.01, weight_decay=0.01)
criterion2 = nn.CrossEntropyLoss()

# Train the neural network
num_epochs = 7
for epoch in range(num_epochs):
    model2.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion2(outputs, labels)
        loss.backward()
        optimizer2.step()
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model test
model2.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: { correct / total * 100}%')


# Evaluate the model train
model2.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model2(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on train set: { correct / total * 100}%')

# Evaluate the model and store predictions
model2.eval()
predictions2 = []
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model2(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions2.extend(predicted.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

import random
import matplotlib.pyplot as plt

# Get a random index for the test data
random_index = random.randint(0, len(test_dataset) - 1)

# Get the image and label at the random index
image, label = test_dataset[random_index]

# Reshape the image tensor to a 28x28 shape
image = image.view(28, 28)

# Convert the image tensor to a numpy array for visualization
image_numpy = image.numpy()

# Assuming 'predictions' contains the predicted labels for the test data
predicted_label1 = predictions2[random_index]

# Display the image with both predicted and actual labels
plt.imshow(image_numpy, cmap='gray')
plt.title(f'Predicted Label: {predicted_label1}, Actual Label: {label}')
plt.axis('off')
plt.show()

"""1. We observe a drop in accuracy when we use SGD in Optimizer

**2. Model with RMSProp Optimizer**
"""

model3 = MLP()
optimizer3 = optim.RMSprop(model3.parameters(), lr=0.001)
criterion3 = nn.CrossEntropyLoss()

# Train the neural network
num_epochs = 5
for epoch in range(num_epochs):
    model3.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer3.zero_grad()
        outputs = model3(inputs)
        loss = criterion3(outputs, labels)
        loss.backward()
        optimizer3.step()
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model test
model3.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model3(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: { correct / total *100}%')


# Evaluate the model train
model3.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model3(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on train set: { correct / total *100}%')

# Evaluate the model and store predictions
model3.eval()
predictions3 = []
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model3(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions3.extend(predicted.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

import random
import matplotlib.pyplot as plt

# Get a random index for the test data
random_index = random.randint(0, len(test_dataset) - 1)

# Get the image and label at the random index
image, label = test_dataset[random_index]

# Reshape the image tensor to a 28x28 shape
image = image.view(28, 28)

# Convert the image tensor to a numpy array for visualization
image_numpy = image.numpy()

# Assuming 'predictions' contains the predicted labels for the test data
predicted_label1 = predictions3[random_index]

# Display the image with both predicted and actual labels
plt.imshow(image_numpy, cmap='gray')
plt.title(f'Predicted Label: {predicted_label1}, Actual Label: {label}')
plt.axis('off')
plt.show()

"""There is an increase in accuracy of the test data and a slight increase in the accuracy on train data when we use RMSProp optimizer

**3. Model with Dropout**
"""

# Define the neural network architecture
class MLP4(nn.Module):
    def __init__(self):
        super(MLP4, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)  # Dropout layer with 20% dropout rate

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after the first hidden layer
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout after the second hidden layer
        x = self.fc3(x)
        return x

# Initialize the neural network
model4 = MLP4()

# Define the loss function and optimizer
criterion4 = nn.CrossEntropyLoss()
optimizer4 = optim.SGD(model4.parameters(), lr=0.001)

# Train the neural network
num_epochs = 5
for epoch in range(num_epochs):
    model4.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer4.zero_grad()
        outputs = model4(inputs)
        loss = criterion4(outputs, labels)
        loss.backward()
        optimizer4.step()
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model test
model4.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model4(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: { correct / total *100}%')


# Evaluate the model train
model4.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model4(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on train set: { correct / total *100}%')

# Evaluate the model and store predictions
model4.eval()
predictions4 = []
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model4(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions4.extend(predicted.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

import random
import matplotlib.pyplot as plt

# Get a random index for the test data
random_index = random.randint(0, len(test_dataset) - 1)

# Get the image and label at the random index
image, label = test_dataset[random_index]

# Reshape the image tensor to a 28x28 shape
image = image.view(28, 28)

# Convert the image tensor to a numpy array for visualization
image_numpy = image.numpy()

# Assuming 'predictions' contains the predicted labels for the test data
predicted_label1 = predictions4[random_index]

# Display the image with both predicted and actual labels
plt.imshow(image_numpy, cmap='gray')
plt.title(f'Predicted Label: {predicted_label1}, Actual Label: {label}')
plt.axis('off')
plt.show()

"""**4. We increase the number of nodes, use dropout and LeakyRelu**"""

import torch.nn.functional as F

class MLP5(nn.Module):
    def __init__(self):
        super(MLP5, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)  # Add dropout layer with 20% dropout rate

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)  # LeakyReLU activation
        x = self.dropout(x)  # Apply dropout
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.fc4(x)
        return x

# Initialize the neural network
model5 = MLP5()


criterion5 = nn.CrossEntropyLoss()

# Initialize the optimizer (Adam)
optimizer5 = optim.Adam(model5.parameters(), lr=0.01)

# Train the neural network
num_epochs = 5
for epoch in range(num_epochs):
    model5.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer5.zero_grad()
        outputs = model5(inputs)
        loss = criterion5(outputs, labels)
        loss.backward()
        optimizer5.step()
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0

print('Finished Training')

# Evaluate the model test
model5.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model5(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: { correct / total*100}%')


# Evaluate the model train
model5.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = model5(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on train set: { correct / total*100}%')

# Evaluate the model and store predictions
model5.eval()
predictions5 = []
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model5(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions5.extend(predicted.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

import random
import matplotlib.pyplot as plt

# Get a random index for the test data
random_index = random.randint(0, len(test_dataset) - 1)

# Get the image and label at the random index
image, label = test_dataset[random_index]

# Reshape the image tensor to a 28x28 shape
image = image.view(28, 28)

# Convert the image tensor to a numpy array for visualization
image_numpy = image.numpy()

# Assuming 'predictions' contains the predicted labels for the test data
predicted_label1 = predictions5[random_index]

# Display the image with both predicted and actual labels
plt.imshow(image_numpy, cmap='gray')
plt.title(f'Predicted Label: {predicted_label1}, Actual Label: {label}')
plt.axis('off')
plt.show()

"""There are many combinations to experiment with amongst different optimizers, loss functions, dropout, and activation functions. So far we observed that there has been no drastic improvement in the performance of the network.

The original neural network configuration without the additional dense layer performed slightly better on the test data compared to the modified version with an extra layer and SGD optimizer model.

This suggests that while the additional layer increased the model's capacity to learn complex representations, it did not necessarily translate to improved performance on the test data,
"""



"""References:

1. https://theneuralblog.com/forward-pass-backpropagation-example/
2. Tutorial of Data_255
"""
