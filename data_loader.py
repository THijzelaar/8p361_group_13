import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset


def get_train_valid_loader(data_dir,
                           dataset,
                           batch_size,
                           random_seed,
                           exp='azimuth',
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(0.5),
                 transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                transform=transforms.Compose(trans))
        
    if exp not in VIEWPOINT_EXPS:
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx = indices[split:]
        valid_idx = indices[:split]

        train_set = Subset(dataset, train_idx)
        valid_set = Subset(dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader

def get_test_loader(data_dir,
                    dataset,
                    batch_size,
                    exp='azimuth', # smallnorb only
                    familiar=True, # smallnorb only
                    num_workers=4,
                    pin_memory=False):

    data_dir = data_dir + '/' + dataset

    if dataset == "cifar10":
        trans = [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        dataset = datasets.CIFAR10(data_dir, train=False, download=False,
                transform=transforms.Compose(trans))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10}
}

VIEWPOINT_EXPS = ['azimuth', 'elevation']
