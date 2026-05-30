import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def split_data_iid(trainset, num_clients=10):
    """Simple IID split — each client gets random balanced data"""
    num_samples = len(trainset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    client_size = num_samples // num_clients
    client_data = {}
    for i in range(num_clients):
        client_data[i] = indices[i * client_size:(i + 1) * client_size]
    return client_data

def get_client_dataloader(trainset, indices, batch_size=32):
    subset = torch.utils.data.Subset(trainset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)