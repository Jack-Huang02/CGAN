from torchvision import datasets, utils, transforms
from torch.utils.data import Dataset, DataLoader
from config import image_size, batch_size, workers
import matplotlib.pyplot as plt
import torch

def get_minst_loader():
    train_data = datasets.MNIST(root='./data', 
                                train=True, 
                                download=True, 
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5, ))
                                ]))
    test_data = datasets.MNIST(root='./data', 
                                train=False, 
                                download=True, 
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5, ))
                                ]))
    dataset = train_data + test_data # 不区分训练集和测试集
    dataloader = DataLoader(dataset, 
                            batch_size = batch_size, 
                            num_workers = workers,
                            shuffle = True)
    return dataset, dataloader
