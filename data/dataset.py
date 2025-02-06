import torch
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

def get_loader(batch_s):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(32),  # Resize to desired resolution
        transforms.CenterCrop(128),  # Crop to maintain face alignment
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5], [0.5])  # Normalize between -1 and 1
    ])

    # Download and load the dataset
    dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_s, shuffle=True, num_workers=4)
    return dataloader