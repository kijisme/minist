import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='minist', train=True, download=True, transform=transform)
print(len(train_dataset))

test_dataset = datasets.MNIST(root='minist', train=False, download=True, transform=transform)
print(test_dataset.len())